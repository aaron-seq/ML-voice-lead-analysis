import boto3
import os
import uuid
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')

def start_transcription_job_handler(event, context):
    """
    AWS Lambda handler triggered by an S3 'ObjectCreated' event.
    This function starts an AWS Transcribe job for the newly uploaded audio file.
    """
    try:
        # 1. Get file details from the S3 event record
        s3_record = event['Records'][0]['s3']
        bucket_name = s3_record['bucket']['name']
        object_key = s3_record['object']['key']

        # Ensure the file is an audio file (simple check)
        if not any(object_key.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.m4a']):
            logger.warning(f"Object '{object_key}' is not a supported audio file. Skipping transcription.")
            return {'statusCode': 200, 'body': 'File is not a supported audio format.'}

        # 2. Get environment variables
        output_bucket = os.environ.get('TRANSCRIPTION_OUTPUT_BUCKET')
        if not output_bucket:
            raise ValueError("Environment variable 'TRANSCRIPTION_OUTPUT_BUCKET' is not set.")

        # 3. Prepare Transcription Job parameters
        transcription_job_name = f"transcription-job-{uuid.uuid4()}"
        media_file_uri = f"s3://{bucket_name}/{object_key}"
        media_format = os.path.splitext(object_key)[1].lstrip('.')

        # 4. Start the transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=transcription_job_name,
            LanguageCode='en-US',  # Configurable based on needs
            MediaFormat=media_format,
            Media={'MediaFileUri': media_file_uri},
            OutputBucketName=output_bucket,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2  # Assuming a 2-person call
            }
        )

        logger.info(f"Successfully started transcription job: {transcription_job_name} for file {object_key}")
        return {
            'statusCode': 200,
            'body': f"Started transcription job: {transcription_job_name}"
        }

    except KeyError as e:
        logger.error(f"Invalid S3 event structure: {e}")
        return {'statusCode': 400, 'body': 'Invalid S3 event trigger format.'}
    except Exception as error:
        logger.error(f"Error starting transcription job: {error}")
        # Consider sending a notification (e.g., SNS) on failure
        return {
            'statusCode': 500,
            'body': f"Error starting transcription job: {str(error)}"
        }
