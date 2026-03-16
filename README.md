\# Intelligent Support Ticket Routing with NLP



An NLP-powered support ticket routing system that classifies incoming customer requests into the correct support team using TF-IDF, machine learning, and confidence-based human review logic.



\---



\## Overview



Support teams often spend valuable time manually reviewing incoming tickets before assigning them to the right department. This slows response times, creates inconsistency in triage, and reduces operational efficiency.



This project builds an end-to-end support ticket routing workflow that:



\- generates and cleans a realistic synthetic support ticket dataset

\- transforms ticket text into machine-learning-ready TF-IDF features

\- trains and compares multiple baseline classification models

\- selects the best-performing routing model

\- analyses prediction confidence and low-certainty cases

\- applies confidence-based fallback logic for human review

\- deploys the final workflow in a Streamlit app



The final model predicts the most appropriate routed team for each ticket, helping automate first-line support triage.



\---



\## Business Problem



In many customer support environments, incoming tickets must be manually reviewed and routed to teams such as:



\- Technical Support

\- Billing

\- Account Access

\- Orders \& Delivery

\- Refunds \& Returns

\- Product Inquiry



When ticket volumes increase, manual routing can become:



\- slow

\- repetitive

\- inconsistent

\- difficult to scale



This project addresses that problem by building a machine learning system that automatically classifies incoming support requests into the correct handling team, while still allowing uncertain tickets to be flagged for manual review.



\---



\## Project Objectives



The main objectives of this project were to:



\- build a realistic multi-class NLP classification problem for support ticket routing

\- avoid overly easy synthetic data by introducing overlap, ambiguity, and mixed-intent tickets

\- compare baseline machine learning models for routing performance

\- evaluate model performance using Macro F1, Weighted F1, and Accuracy

\- inspect confidence behaviour and misclassifications

\- create a deployment-ready Streamlit app for interactive ticket routing



\---



\## Dataset



This project uses a \*\*synthetic support ticket dataset\*\* designed to resemble real-world service operations data.



\### Ticket classes



Each ticket is assigned a dominant routed team:



\- Technical Support

\- Billing

\- Account Access

\- Orders \& Delivery

\- Refunds \& Returns

\- Product Inquiry



\### Why synthetic data was used



A synthetic dataset was created to allow full control over:



\- ticket categories

\- class balance

\- mixed-intent cases

\- overlapping issue language

\- duplicate-like rows

\- missing values

\- noisy text patterns



\### Realism improvements



The revised version of the dataset was designed to be harder and more realistic by including:



\- overlapping vocabulary across teams

\- vague subject lines

\- secondary issue mentions

\- mixed-intent tickets

\- text noise and typos

\- duplicate-like support cases

\- non-standard missing markers



This helped reduce the risk of unrealistically perfect classification performance.



\---



\## Project Workflow



\### 1. Data generation and cleaning

\- generated synthetic support tickets with realistic metadata

\- introduced ambiguity, overlap, and mixed-intent patterns

\- injected duplicate-like rows and missing values

\- cleaned and standardised the data

\- created a unified `ticket\_text` field for NLP modeling



\### 2. Text preprocessing and baseline modeling

\- loaded the cleaned dataset

\- performed a stratified train-test split

\- converted ticket text into TF-IDF features

\- trained and compared baseline models:

&#x20; - Multinomial Naive Bayes

&#x20; - Logistic Regression

&#x20; - LinearSVC

&#x20; - Random Forest



\### 3. Explainability and operational review

\- evaluated class-level performance

\- reviewed confidence scores

\- inspected low-confidence cases

\- added a confidence-based human review rule

\- prepared deployment-ready prediction outputs



\### 4. Streamlit deployment

\- built an interactive app for routing new tickets

\- displayed predicted team and confidence score

\- applied auto-route vs human-review logic

\- showed ranked class scores for each input



\---



\## Model Performance



The best-performing model was:



\- \*\*Model:\*\* LinearSVC



\### Final results

\- \*\*Accuracy:\*\* 0.962

\- \*\*Macro F1:\*\* 0.962

\- \*\*Weighted F1:\*\* 0.9622



These results were achieved on the revised synthetic dataset after introducing more realistic overlap and ambiguity into the ticket text.



\---



\## Confidence-Based Review Logic



To support safer operational use, the project includes a confidence-based fallback rule:



\- \*\*High-confidence tickets\*\* can be auto-routed

\- \*\*Lower-confidence tickets\*\* are flagged for human review



This makes the workflow more practical for real support settings, where not every prediction should be trusted equally.



\---



\## Streamlit App Features



The deployed app allows a user to:



\- enter a ticket subject

\- enter a ticket description

\- predict the most likely routed team

\- view the model confidence score

\- view the routing decision:

&#x20; - Auto-route

&#x20; - Human review

\- inspect ranked class scores



\---



\## Tools and Technologies



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Scikit-learn

\- TF-IDF Vectorization

\- Streamlit

\- Joblib

\- Jupyter Notebook



\---



\## Project Structure



```bash

intelligent-support-ticket-routing-nlp/

│

├── app.py

├── requirements.txt

├── README.md

│

├── data/

│   ├── raw/

│   │   ├── support\_tickets\_raw.csv

│   │   └── support\_tickets\_raw\_v2.csv

│   └── processed/

│       ├── support\_tickets\_clean.csv

│       └── support\_tickets\_clean\_v2.csv

│

├── figures/

│   ├── team\_distribution\_v2.png

│   ├── priority\_distribution\_v2.png

│   ├── secondary\_issue\_distribution\_v2.png

│   ├── channel\_by\_team\_v2.png

│   ├── text\_length\_distribution\_v2.png

│   ├── text\_length\_by\_team\_v2.png

│   ├── best\_model\_confusion\_matrix.png

│   ├── confidence\_distribution.png

│   └── auto\_route\_vs\_review.png

│

├── models/

│   └── best\_ticket\_routing\_pipeline.joblib

│

├── results/

│   ├── baseline\_model\_comparison.csv

│   ├── best\_model\_classification\_report.csv

│   ├── best\_model\_test\_predictions.csv

│   ├── notebook2\_summary.csv

│   ├── notebook3\_classification\_report.csv

│   ├── notebook3\_top\_terms.csv

│   ├── notebook3\_summary.csv

│   └── ticket\_triage\_operational\_output.csv

│

└── notebooks/

&#x20;   ├── 01\_support\_ticket\_data\_generation\_cleaning\_eda\_v2.ipynb

&#x20;   ├── 02\_support\_ticket\_tfidf\_baseline\_models.ipynb

&#x20;   └── 03\_support\_ticket\_explainability\_confidence\_operational\_readiness.ipynb



```



\## Key Learning Points



This project highlights several important machine learning and NLP lessons:



&#x20;- synthetic datasets can be useful, but overly clean synthetic text can produce unrealistic results

&#x20;- introducing overlap and ambiguity creates a more meaningful classification task

&#x20;- Macro F1 is especially useful for evaluating multi-class support routing problems

&#x20;- confidence-aware workflows are important when deploying classification systems

&#x20;- explainability and error analysis matter just as much as top-line performance



\## Limitations



&#x20;- the dataset is synthetic rather than collected from a live support environment

&#x20;- confidence values from LinearSVC are approximate rather than true probabilities

&#x20;- routing is based mainly on text content, without deeper customer history or operational context

&#x20;- the current system predicts a single dominant team rather than fully multi-label routing



