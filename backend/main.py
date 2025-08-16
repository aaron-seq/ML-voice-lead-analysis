from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json

app = FastAPI()

# --- CORS Configuration ---
# This allows your React frontend to communicate with the backend
origins = [
    "http://localhost:3000",
    # Add your deployed frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AWS S3 Configuration ---
S3_BUCKET_NAME = "your-ml-pipeline-bucket" # Use the same bucket as your pipeline output
S3_PREFIX = "analysis-results/"

s3_client = boto3.client('s3')

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Voice Lead Analysis API"}

@app.get("/calls")
def get_all_calls():
    """
    Retrieves a list of all analyzed call files from S3.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        if 'Contents' not in response:
            return []
            
        # Extract just the filenames
        files = [{"fileName": obj['Key'].replace(S3_PREFIX, "")} for obj in response['Contents']]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calls/{file_name}")
def get_call_details(file_name: str):
    """
    Retrieves the detailed analysis for a specific call.
    """
    s3_key = f"{S3_PREFIX}{file_name}"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        content = obj['Body'].read().decode('utf-8')
        return json.loads(content)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Call analysis not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

