# Scam Message NLP Analysis

Developed by **IwazoLab**, this project creates a natural language processing (NLP) pipeline to classify SMS messages as 🔴 High-risk scams, 🟡 Moderate-risk scams, or 🟢 Low-risk/Legit. It aims to protect users by detecting fraud, prioritizing investigations, and enabling smart alerts in apps.

## Dataset

The dataset [scam_detection_labeled.csv](/data/scam_detection_labeled.csv) contains 545 text messages, including legitimate notices (e.g., MPESA payments, rent reminders, Bible verses) and scams (e.g., fake prizes, urgent job offers). It powers the pipeline to identify scam patterns.


## Goals

- Build reliable models for scam detection.
- Enable real-time fraud alerts for users.


## Key Features

- Classifies messages into High-risk, Moderate-risk, or Low-risk/Legit.
- Supports English and Swahili for local SMS.
- Uses advanced models for precise scam detection.
- Offers potential for real-time fraud alerts.

## Findings

The [Findings Report](/reports/findings.md) presents comprehensive results across all stages of the NLP pipeline. It provides metrics, insights, and sample outputs. This report can be used to evaluate pipeline effectiveness, identify bottlenecks, and guide optimizations for robust scam detection.

## Methodologies

The [Methodologies Document](/reports/methodologies.md) outlines the technical approaches used across the NLP pipeline. This resource is designed for developers seeking to understand, replicate, or extend the pipeline, providing a clear roadmap of techniques and tools used.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iwazolab/scam-nlp-pipeline

   cd scam-nlp-pipeline

   pip install -r requirements.txt