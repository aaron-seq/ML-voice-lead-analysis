import boto3
import os
import uuid

# Initialize the Transcribe client
transcribe_client = boto3.client('transcribe')

def lambda_handler(event, context):
    """
    This Lambda function is triggered by an S3 event.
    It starts a transcription job for the uploaded audio file.
    """
    # Get the bucket and key from the S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Generate a unique job name
    job_name = f"transcription-job-{uuid.uuid4()}"
    
    # The S3 URI of the audio file
    job_uri = f"s3://{bucket}/{key}"
    
    # The output bucket for the transcription result
    output_bucket = os.environ.get('TRANSCRIPTION_OUTPUT_BUCKET')
    
    try:
        # Start the transcription job
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            LanguageCode='en-US',  # Adjust language code as needed
            MediaFormat=key.split('.')[-1], # Assumes file extension is the format
            Media={
                'MediaFileUri': job_uri
            },
            OutputBucketName=output_bucket,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2 # Assuming a 2-person call (sales rep and lead)
            }
        )
        
        print(f"Successfully started transcription job: {job_name}")
        return {
            'statusCode': 200,
            'body': f"Started transcription job: {job_name}"
        }

    except Exception as e:
        print(f"Error starting transcription job: {e}")
        return {
            'statusCode': 500,
            'body': f"Error starting transcription job: {e}"
        }

