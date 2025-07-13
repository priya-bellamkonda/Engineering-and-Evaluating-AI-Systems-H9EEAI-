# Engineering-and-Evaluating-AI-H9EEAI

## Overview
This repository contains our implementation of a hierarchical modeling approach for multi-label email classification, developed for the "Engineering and Evaluating AI Systems" course at National College of Ireland. The system classifies customer messages into three labels:
1.	Intent (e.g., Feedback, Support)
2.	Tone (e.g., Positive, Neutral, Negative)
3.	Resolution Type (e.g., Auto-reply, Manual Review)

## Team Members
- Priyanka Bellamkonda (23338849)
- Nithin Billakanti (23423153)

## Repository Structure

```text
.
├── data/                   # Sample datasets
│   ├── AppGallery.csv
│   └── Purchasing.csv
├── diagrams/               # Architecture diagrams
│   ├── Chained Multi-Output Architecture.png
│   └── Hierarchical Modeling Architecture.png
├── model/                  # ML model implementations
│   ├── base.py
│   └── randomforest.py
├── modelling/              # Core classification logic
│   ├── data_model.py       # Data handling class
│   └── modelling.py        # Hierarchical implementation
├── main.py                 # Execution entry point
├── preprocess.py           # Data cleaning pipeline
├── embeddings.py           # Feature engineering
├── Config.py               # Global settings
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Key Features
-	Hierarchical Modeling Architecture (implemented in modelling.py)
-	Modular data preprocessing pipeline
-	Custom Random Forest classifiers with balanced class weighting
-	Consistent data formatting across label dependencies
-	Detailed classification reports for evaluation
## How to Run
1.	Install dependencies:
```bash
    pip install -r requirements.txt
```
3.	Place your CSV data files in the data/ directory
4.	Execute the main pipeline:
    ```bash
    python main.py
    ```
5.	View results:
  - Classification reports appear in console
  - Output predictions saved to out.csv
## Report
The final report (ENGINEERING AND EVALUATING AI FINAL REPORT.pdf) details:
- Architectural comparison (Chained vs. Hierarchical)
- Component/connector analysis
- Implementation decisions
- Evaluation results
- SDG alignment (Goal 9: Industry, Innovation and Infrastructure)


