# Meeting Intelligence System

An NLP-based system that analyzes meeting transcripts and extracts actionable insights such as summaries, action items, suggestions, decisions, and key information.

## Features

- Dialogue act classification using fine-tuned BERT
- Action item extraction using spaCy and rule-based NLP
- Named Entity Recognition (NER)
- Decision and suggestion extraction
- Meeting summarization using DistilBART
- REST API using Flask
- Simple web interface using HTML, CSS and JavaScript

## Architecture

Meeting Transcript
        ↓
BERT Dialogue Act Classification
        ↓
spaCy NLP + Rule Engine
        ↓
Action Items / Suggestions / Decisions
        ↓
DistilBART Summarization
        ↓
Flask API
        ↓
Web Interface

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- BERT
- DistilBART
- spaCy
- Flask
- HTML/CSS/JavaScript

## Dataset

The dialogue-act classifier was trained using the AMI Meeting Corpus.

The raw AMI dataset is not included in this repository.

## Model

The fine-tuned BERT dialogue-act classifier is hosted on Hugging Face:

https://huggingface.co/aditiverma6/meeting-dialogue-act-bert

The application automatically downloads the model from Hugging Face when required.

## Model Performance

Test Accuracy: 66.35%

Test Macro-F1: 49.09%

Test Weighted-F1: 65.31%

## Project Structure

```text
MI1/
│
├── data/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│
├── src/
│   ├── action_items.py
│   ├── analyze_meeting.py
│   ├── app.py
│   ├── decision_extractor.py
│   ├── meeting_pipeline.py
│   ├── meeting_report.py
│   ├── ner.py
│   ├── predict.py
│   └── summarizer.py
│
├── requirements.txt
└── README.md
