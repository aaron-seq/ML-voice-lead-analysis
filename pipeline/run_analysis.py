import json
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import boto3
import os
import pandas as pd

# --- CONFIGURATION ---
S3_BUCKET_NAME = os.environ.get("DATA_BUCKET", "your-ml-pipeline-bucket")
MODEL_PATH = "models/lead_score_model.h5"
TOKENIZER_PATH = "models/tokenizer.json"
MAX_SEQUENCE_LENGTH = 200 # Should match the length used during training

# --- LOAD MODELS ---
# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Load TensorFlow model for lead scoring
lead_score_model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# --- HELPER FUNCTIONS ---
def download_from_s3(bucket, key):
    """Downloads a file from S3 and returns its content."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read().decode('utf-8')

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    doc = nlp(text)
    sentiment = doc._.blob.polarity
    return sentiment

def extract_keywords(text):
    """Extracts keywords and topics from the text."""
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    topics = [token.text for token in doc if token.pos_ == "NOUN"]
    return list(set(keywords)), list(set(topics[:10])) # Return top 10 topics

def find_wow_moments(text):
    """Identifies 'wow' moments using spaCy's Matcher."""
    matcher = spacy.matcher.Matcher(nlp.vocab)
    # Define patterns for words indicating excitement
    pattern = [{"LOWER": {"in": ["wow", "amazing", "incredible", "fantastic", "awesome"]}}]
    matcher.add("WOW_MOMENT", [pattern])
    
    doc = nlp(text)
    matches = matcher(doc)
    
    wow_moments = []
    for match_id, start, end in matches:
        span = doc[start:end]
        # Get context around the match
        context_start = max(0, start - 10)
        context_end = min(len(doc), end + 10)
        context = doc[context_start:context_end].text
        wow_moments.append({"keyword": span.text, "context": context})
    return wow_moments

def predict_lead_score(text):
    """Predicts the lead score using the trained TensorFlow model."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = lead_score_model.predict(padded_sequences)
    # Assuming your model outputs probabilities for ['Hot', 'Warm', 'Cold']
    score_map = {0: 'Hot', 1: 'Warm', 2: 'Cold'}
    predicted_class = score_map[prediction.argmax()]
    return predicted_class, float(prediction.max())

def process_transcription(transcription_json):
    """Processes the raw transcription JSON from AWS Transcribe."""
    data = json.loads(transcription_json)
    full_transcript = data['results']['transcripts'][0]['transcript']
    return full_transcript

def main(s3_key):
    """Main function to run the analysis pipeline."""
    # 1. Download transcription from S3
    transcription_content = download_from_s3(S3_BUCKET_NAME, s3_key)
    
    # 2. Process the transcription to get the full text
    full_text = process_transcription(transcription_content)
    
    # 3. Perform analyses
    sentiment = analyze_sentiment(full_text)
    keywords, topics = extract_keywords(full_text)
    wow_moments = find_wow_moments(full_text)
    lead_score, confidence = predict_lead_score(full_text)
    
    # 4. Structure the results
    analysis_results = {
        "fileName": s3_key,
        "transcript": full_text,
        "sentiment": sentiment,
        "keywords": keywords,
        "topics": topics,
        "wowMoments": wow_moments,
        "leadScore": {
            "score": lead_score,
            "confidence": confidence
        }
    }
    
    # 5. Save results to a database (e.g., DynamoDB or RDS)
    # This part would involve using boto3 to write to DynamoDB or a library like
    # psycopg2 to write to a PostgreSQL database.
    # For this example, we'll just print the results.
    print(json.dumps(analysis_results, indent=2))
    
    # Example of saving to a JSON file in S3
    output_key = f"analysis-results/{s3_key.split('/')[-1]}"
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=output_key,
        Body=json.dumps(analysis_results, indent=2)
    )
    print(f"Analysis results saved to s3://{S3_BUCKET_NAME}/{output_key}")


if __name__ == "__main__":
    # This would be triggered by an event, but we can run it manually for a test file
    # You would need to have a transcribed JSON file in your S3 bucket
    test_s3_key = "transcripts/example_call.json" 
    main(test_s3_key)

