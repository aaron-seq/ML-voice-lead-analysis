# ML Voice Lead Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![AWS](https://img.shields.io/badge/AWS-Transcribe-orange.svg)](https://aws.amazon.com/transcribe/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-blue.svg)](https://spacy.io/)

An end-to-end Machine Learning pipeline that transcribes and analyzes sales cold calls to extract actionable buyer insights, identify qualified leads, and visualize key metrics in a real-time dashboard.

## Overview

In a typical sales cycle, hundreds of cold calls are made weekly. These calls contain a wealth of information about customer needs, objections, and interest levels. Manually listening to and analyzing these calls is time-consuming and doesn't scale. This project automates the process by:

1.  **Transcribing** audio recordings of cold calls using **AWS Transcribe**.
2.  **Analyzing** the transcriptions using Natural Language Processing (NLP) with **TensorFlow** and **spaCy**.
3.  **Extracting** key insights, such as sentiment, topics, lead score, and moments of excitement (e.g., usage of words like "wow" or "amazing").
4.  **Visualizing** these insights in an interactive **React.js** dashboard for the sales team.

This pipeline helped us **boost qualified leads by 15%** by enabling the sales team to focus on the most promising prospects and tailor their follow-ups based on deep insights from the calls.

## Project Architecture

The pipeline is designed with a modular and scalable architecture, leveraging cloud services and modern MLOps principles.

```
                            +-----------------------+
                            |   S3 (Audio Files)    |
                            +-----------+-----------+
                                        | (Trigger)
                                        v
+-----------------------+      +-----------------------+      +-----------------------+
|     AWS Lambda        |----->|    AWS Transcribe     |----->|      S3 (JSON)        |
| (File Upload Handler) |      | (Speech-to-Text)      |      |  (Transcripts)        |
+-----------------------+      +-----------------------+      +-----------+-----------+
                                                                         | (Trigger)
                                                                         v
+------------------------------------------------------------------------+
|                                                                        |
|                             ML Processing Pipeline (AWS Step Functions / Airflow) |
|                                                                        |
|   +---------------------+  +-----------------------+  +---------------------+   |
|   |  Data Preprocessing |->| Sentiment & Lead Score|->| Keyword/Topic       |   |
|   |      (spaCy)        |  |  (TensorFlow/Keras)   |  | Extraction (spaCy)  |   |
|   +---------------------+  +-----------------------+  +---------------------+   |
|                                                                        |
+---------------------------------+----------------------------------------+
                                  | (Store Results)
                                  v
                      +-----------------------+
                      |   PostgreSQL/DynamoDB |
                      |   (Analyzed Data)     |
                      +-----------+-----------+
                                  |
                                  v
+-----------------------+      +-----------------------+
|      FastAPI/Flask    |----->|     React.js          |
|      (Backend API)    |      |     (Dashboard)       |
+-----------------------+      +-----------------------+

```

## Key Features

* **Automated Transcription:** Asynchronous transcription of call recordings in various audio formats.
* **Lead Scoring:** A TensorFlow-based classification model to score leads as 'Hot', 'Warm', or 'Cold' based on the language used.
* **Sentiment Analysis:** Track sentiment trends throughout the call to understand the prospect's engagement.
* **"Wow" Moment Detection:** Uses spaCy's pattern matching to identify moments of high interest when specific features are discussed.
* **Topic & Keyword Extraction:** Automatically extracts key topics (e.g., pricing, integration, competitors) and keywords.
* **Interactive Dashboard:** A comprehensive React dashboard with filters, search, and drill-down capabilities to explore the call data.
* **Scalable & Serverless:** Built with cloud-native services to handle a high volume of calls.

## Technology Stack

* **Cloud:** AWS (S3, Lambda, Transcribe, Step Functions, RDS/DynamoDB)
* **Data Processing:** Python, Pandas
* **ML/NLP:** TensorFlow, Keras, Scikit-learn, spaCy
* **Backend:** Python, FastAPI / Flask
* **Frontend:** JavaScript, React.js, D3.js / Recharts
* **CI/CD:** GitHub Actions
* **Infrastructure as Code:** Terraform / AWS CDK (Optional but recommended)

## Getting Started

### Prerequisites

* An AWS Account with appropriate permissions.
* Python 3.9+ and Node.js 16+ installed.
* Docker (for containerizing services).

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ML_Voice_Lead_Analysis.git](https://github.com/your-username/ML_Voice_Lead_Analysis.git)
    cd ML_Voice_Lead_Analysis
    ```

2.  **Setup Backend & ML Pipeline:**
    * Navigate to the `backend` directory.
    * Create a virtual environment and install dependencies:
        ```bash
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        ```
    * Configure your AWS credentials.
    * Set up the necessary environment variables in a `.env` file (see `.env.example`).

3.  **Setup Frontend:**
    * Navigate to the `frontend` directory.
    * Install dependencies:
        ```bash
        npm install
        ```
    * Set up the environment variables in a `.env.local` file to point to your backend API.

### Running the Application

1.  **Start the Backend API:**
    ```bash
    cd backend
    uvicorn main:app --reload
    ```

2.  **Start the React Frontend:**
    ```bash
    cd frontend
    npm start
    ```

3.  **Run the ML Pipeline:**
    The pipeline is triggered by uploading an audio file to the designated S3 bucket. You can also run parts of it manually for testing:
    ```bash
    python pipeline/run_analysis.py --file_path /path/to/your/audio.wav
    ```

## Project Structure

```
ML_Voice_Lead_Analysis/
├── .github/workflows/         # CI/CD pipelines (e.g., test.yml, deploy.yml)
├── backend/                   # FastAPI/Flask application
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── core/              # Configuration, settings
│   │   └── services/          # Business logic
│   ├── models/                # Trained ML models (e.g., .h5, .pkl)
│   ├── main.py                # App entry point
│   └── requirements.txt
├── frontend/                  # React.js dashboard
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   └── ...
├── pipeline/                  # ML pipeline scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── transcribe.py
├── scripts/                   # Helper and deployment scripts
│   └── setup_aws.sh
├── tests/                     # Unit and integration tests
│   ├── test_api.py
│   └── test_pipeline.py
├── README.md
└── .gitignore
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Hat tip to anyone whose code was used as inspiration.
* The teams behind the amazing open-source tools used in this project.
