import os
import json
import argparse
import logging
from typing import List, Tuple, Dict, Any

import boto3
import spacy
from spacy.matcher import Matcher
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spacy_text_blob.spacytextblob import SpacyTextBlob

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load settings from environment variables for better security and flexibility
S3_BUCKET_NAME = os.environ.get("DATA_BUCKET", "your-ml-pipeline-bucket")
MODEL_PATH = "models/lead_score_model.h5"
TOKENIZER_PATH = "models/tokenizer.json"
MAX_SEQUENCE_LENGTH = 200  # Must match the length used during model training

# --- Global Model Loading ---
# Load models once to improve performance when processing multiple files
try:
    NLP = spacy.load("en_core_web_sm")
    NLP.add_pipe('spacytextblob') # Add sentiment analysis pipe

    LEAD_SCORE_MODEL = tf.keras.models.load_model(MODEL_PATH)
    
    with open(TOKENIZER_PATH) as f:
        tokenizer_data = json.load(f)
        TOKENIZER = tokenizer_from_json(tokenizer_data)

    WOW_MOMENT_MATCHER = Matcher(NLP.vocab)
    # Define patterns for words indicating excitement or high interest
    wow_pattern = [{"LOWER": {"in": ["wow", "amazing", "incredible", "fantastic", "awesome", "perfect", "great"]}}]
    WOW_MOMENT_MATCHER.add("WOW_MOMENT", [wow_pattern])

except FileNotFoundError as e:
    logging.error(f"Model or tokenizer file not found: {e}. Please ensure models are in the correct directory.")
    exit(1)
except Exception as e:
    logging.error(f"An error occurred during model loading: {e}")
    exit(1)


# --- Helper Functions ---
def download_file_from_s3(bucket: str, key: str) -> str:
    """Downloads a file from S3 and returns its content as a string."""
    logging.info(f"Downloading s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    return s3_object['Body'].read().decode('utf-8')

def upload_file_to_s3(bucket: str, key: str, content: str) -> None:
    """Uploads string content to a file in S3."""
    logging.info(f"Uploading analysis results to s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=content)

def parse_transcription_json(transcription_content: str) -> str:
    """Extracts the full text transcript from the AWS Transcribe JSON output."""
    data = json.loads(transcription_content)
    return data['results']['transcripts'][0]['transcript']

# --- Analysis Functions ---
def analyze_sentiment(doc: spacy.tokens.doc.Doc) -> float:
    """Calculates the overall sentiment polarity of the text."""
    return doc._.blob.polarity

def extract_key_phrases_and_topics(doc: spacy.tokens.doc.Doc) -> Tuple[List[str], List[str]]:
    """Extracts noun chunks as key phrases and nouns as topics."""
    key_phrases = list(set([chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]))
    topics = list(set([token.lemma_ for token in doc if token.pos_ == "NOUN"]))
    return key_phrases, topics[:10]  # Return top 10 unique topics

def find_high_interest_moments(doc: spacy.tokens.doc.Doc) -> List[Dict[str, str]]:
    """Identifies 'wow' moments of high interest using pattern matching."""
    matches = WOW_MOMENT_MATCHER(doc)
    high_interest_moments = []
    for _, start, end in matches:
        span = doc[start:end]
        context_start = max(0, start - 15)
        context_end = min(len(doc), end + 15)
        context = doc[context_start:context_end].text.replace('\n', ' ')
        high_interest_moments.append({"keyword": span.text, "context": f"...{context}..."})
    return high_interest_moments

def predict_lead_score(text: str) -> Tuple[str, float]:
    """Predicts the lead score using the trained TensorFlow model."""
    sequences = TOKENIZER.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = LEAD_SCORE_MODEL.predict(padded_sequences)[0]
    
    score_map = {0: 'Hot', 1: 'Warm', 2: 'Cold'}
    predicted_index = prediction.argmax()
    predicted_class = score_map.get(predicted_index, 'Unknown')
    confidence = float(prediction[predicted_index])
    
    return predicted_class, confidence

# --- Main Pipeline Logic ---
def run_full_analysis(transcription_key: str) -> None:
    """
    Main function to execute the full analysis pipeline for a single transcription file.
    """
    try:
        # 1. Download and process transcription file
        transcription_content = download_file_from_s3(S3_BUCKET_NAME, transcription_key)
        full_text = parse_transcription_json(transcription_content)
        
        # 2. Perform NLP analysis using spaCy
        doc = NLP(full_text)
        
        # 3. Run individual analysis components
        logging.info("Running analysis...")
        sentiment_score = analyze_sentiment(doc)
        key_phrases, topics = extract_key_phrases_and_topics(doc)
        wow_moments = find_high_interest_moments(doc)
        lead_score, confidence = predict_lead_score(full_text)
        
        # 4. Structure the final results
        analysis_results = {
            "fileName": os.path.basename(transcription_key),
            "transcript": full_text,
            "sentiment": sentiment_score,
            "keywords": key_phrases,
            "topics": topics,
            "wowMoments": wow_moments,
            "leadScore": {
                "score": lead_score,
                "confidence": confidence
            }
        }
        
        # 5. Save results to S3
        output_key = f"analysis-results/{os.path.basename(transcription_key)}"
        upload_file_to_s3(
            S3_BUCKET_NAME,
            output_key,
            json.dumps(analysis_results, indent=2)
        )
        logging.info(f"Successfully saved analysis to s3://{S3_BUCKET_NAME}/{output_key}")

    except Exception as e:
        logging.error(f"Failed to process {transcription_key}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML analysis on a transcribed call.")
    parser.add_argument(
        "s3_key",
        type=str,
        help="The S3 key for the transcription JSON file (e.g., 'transcripts/example_call.json')."
    )
    args = parser.parse_args()
    
    run_full_analysis(args.s3_key)
