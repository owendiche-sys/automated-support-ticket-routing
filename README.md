# Intelligent Support Ticket Routing with NLP

An NLP-powered support ticket routing system that classifies incoming customer requests into the correct support team using TF-IDF, machine learning, and confidence-based human review logic.

---

## Overview

Support teams often spend valuable time manually reviewing incoming tickets before assigning them to the right department. This slows response times, creates inconsistency in triage, and reduces operational efficiency.

This project builds an end-to-end support ticket routing workflow that:

- generates and cleans a realistic synthetic support ticket dataset
- transforms ticket text into machine-learning-ready TF-IDF features
- trains and compares multiple baseline classification models
- selects the best-performing routing model
- analyses prediction confidence and low-certainty cases
- applies confidence-based fallback logic for human review
- deploys the final workflow in a Streamlit app

The final model predicts the most appropriate routed team for each ticket, helping automate first-line support triage.

---

## Business Problem

In many customer support environments, incoming tickets must be manually reviewed and routed to teams such as:

- Technical Support
- Billing
- Account Access
- Orders & Delivery
- Refunds & Returns
- Product Inquiry

When ticket volumes increase, manual routing can become:

- slow
- repetitive
- inconsistent
- difficult to scale

This project addresses that problem by building a machine learning system that automatically classifies incoming support requests into the correct handling team, while still allowing uncertain tickets to be flagged for manual review.

---

## Project Objectives

The main objectives of this project were to:

- build a realistic multi-class NLP classification problem for support ticket routing
- avoid overly easy synthetic data by introducing overlap, ambiguity, and mixed-intent tickets
- compare baseline machine learning models for routing performance
- evaluate model performance using Macro F1, Weighted F1, and Accuracy
- inspect confidence behaviour and misclassifications
- create a deployment-ready Streamlit app for interactive ticket routing

---

## Dataset

This project uses a **synthetic support ticket dataset** designed to resemble real-world service operations data.

### Ticket classes

Each ticket is assigned a dominant routed team:

- Technical Support
- Billing
- Account Access
- Orders & Delivery
- Refunds & Returns
- Product Inquiry

### Why synthetic data was used

A synthetic dataset was created to allow full control over:

- ticket categories
- class balance
- mixed-intent cases
- overlapping issue language
- duplicate-like rows
- missing values
- noisy text patterns

### Realism improvements

The revised version of the dataset was designed to be harder and more realistic by including:

- overlapping vocabulary across teams
- vague subject lines
- secondary issue mentions
- mixed-intent tickets
- text noise and typos
- duplicate-like support cases
- non-standard missing markers

This helped reduce the risk of unrealistically perfect classification performance.

---

## Project Workflow

### 1. Data generation and cleaning

- generated synthetic support tickets with realistic metadata
- introduced ambiguity, overlap, and mixed-intent patterns
- injected duplicate-like rows and missing values
- cleaned and standardised the data
- created a unified `ticket_text` field for NLP modeling

### 2. Text preprocessing and baseline modeling

- loaded the cleaned dataset
- performed a stratified train-test split
- converted ticket text into TF-IDF features
- trained and compared baseline models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - LinearSVC
  - Random Forest

### 3. Explainability and operational review

- evaluated class-level performance
- reviewed confidence scores
- inspected low-confidence cases
- added a confidence-based human review rule
- prepared deployment-ready prediction outputs

### 4. Streamlit deployment

- built an interactive app for routing new tickets
- displayed predicted team and confidence score
- applied auto-route vs human-review logic
- showed ranked class scores for each input

---

## Model Performance

The best-performing model was:

- **Model:** LinearSVC

### Final results

- **Accuracy:** 0.962
- **Macro F1:** 0.962
- **Weighted F1:** 0.9622

These results were achieved on the revised synthetic dataset after introducing more realistic overlap and ambiguity into the ticket text.

---

## Confidence-Based Review Logic

To support safer operational use, the project includes a confidence-based fallback rule:

- **High-confidence tickets** can be auto-routed
- **Lower-confidence tickets** are flagged for human review

This makes the workflow more practical for real support settings, where not every prediction should be trusted equally.

---

## Streamlit App Features

The deployed app allows a user to:

- enter a ticket subject
- enter a ticket description
- predict the most likely routed team
- view the model confidence score
- view the routing decision:
  - Auto-route
  - Human review
- inspect ranked class scores

---

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TF-IDF Vectorization
- Streamlit
- Joblib
- Jupyter Notebook

---

## Project Structure

```bash
intelligent-support-ticket-routing-nlp/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── support_tickets_raw.csv
│   │   └── support_tickets_raw_v2.csv
│   └── processed/
│       ├── support_tickets_clean.csv
│       └── support_tickets_clean_v2.csv
│
├── figures/
│   ├── team_distribution_v2.png
│   ├── priority_distribution_v2.png
│   ├── secondary_issue_distribution_v2.png
│   ├── channel_by_team_v2.png
│   ├── text_length_distribution_v2.png
│   ├── text_length_by_team_v2.png
│   ├── best_model_confusion_matrix.png
│   ├── confidence_distribution.png
│   └── auto_route_vs_review.png
│
├── models/
│   └── best_ticket_routing_pipeline.joblib
│
├── results/
│   ├── baseline_model_comparison.csv
│   ├── best_model_classification_report.csv
│   ├── best_model_test_predictions.csv
│   ├── notebook2_summary.csv
│   ├── notebook3_classification_report.csv
│   ├── notebook3_top_terms.csv
│   ├── notebook3_summary.csv
│   └── ticket_triage_operational_output.csv
│
└── notebooks/
    ├── 01_support_ticket_data_generation_cleaning_eda_v2.ipynb
    ├── 02_support_ticket_tfidf_baseline_models.ipynb
    └── 03_support_ticket_explainability_confidence_operational_readiness.ipynb
```
