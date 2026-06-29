XAI Demand Forecasting & Driver Analysis Engine
Overview

This project provides an Explainable AI (XAI) framework for Demand Forecasting and Business Driver Analysis. The solution identifies key factors influencing sales demand and quantifies their impact using statistical modeling and SHAP (SHapley Additive Explanations).

The system enables business users and analysts to:

Forecast demand drivers
Understand feature importance
Quantify business impact of controllable and uncontrollable factors
Generate explainable insights for decision-making
Produce driver analysis reports at Region, Category, and Product levels
Key Features
Demand Driver Analysis
Bivariate regression analysis
Driver coefficient estimation
Feature impact measurement
Relative importance ranking
Explainable AI (XAI)
SHAP-based explainability
Global feature importance
Positive and negative contribution analysis
Driver ranking based on business impact
Automated Data Processing
Automatic feature type detection
Numeric data sanitization
Missing value handling
Feature transformation and normalization
Multi-Level Analysis

The engine performs analysis across:

Region
Category Area
Product

Each combination is independently analyzed and aggregated into final reports.

Project Structure
xai_runner/
│
├── main.py
├── runner.py
├── config.py
├── dataset.csv
├── requirements.txt
│
├── preprocess/
│   ├── type_detection.py
│   └── __init__.py
│
├── utils/
│   ├── loggers.py
│   └── __init__.py
│
├── driver_analysis.py
├── interpretation.py
│
└── outputs/
    ├── cleaned_raw_data.csv
    ├── column_summary.csv
    ├── data_description.csv
    ├── Driver_Analysis_Coefficients.csv
    ├── Shap_Aggregate.csv
    ├── Shap_Values.csv
    └── shap_summary_minmax.png
Dataset Requirements

The input dataset must contain:

Mandatory Columns
Column	Description
Region	Geographic region
category_area	Product category
Product	Product identifier
Sales_TY	Target sales variable
Optional Driver Variables
Price
Promotions
Economic indicators
Weather indicators
Holiday information
Events
Seasonal factors
Calendar variables
Processing Workflow
Step 1: Load Dataset

The system loads:

dataset.csv
Step 2: Segment Data

Data is split by:

Region
    └── Category Area
            └── Product
Step 3: Feature Engineering
Data type detection
Numeric conversion
Encoding
Scaling (Min-Max)
Step 4: Driver Analysis

For each product:

Regression model training
Driver coefficient estimation
Statistical significance analysis
Step 5: SHAP Analysis

The engine calculates:

SHAP values
Driver importance
Positive contribution %
Negative contribution %
Mean impact
Step 6: Generate Outputs

Consolidated reports are created in the outputs folder.

Installation
Clone Repository
git clone <repository-url>
cd xai_runner
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Running the Application

Execute:

python main.py

The system will:

Load dataset
Create metadata
Run analysis by Region → Category → Product
Generate explainability reports
Save outputs
Output Files
Driver_Analysis_Coefficients.csv

Contains:

Driver coefficients
Impact estimates
Business driver rankings
Shap_Aggregate.csv

Contains:

Driver importance scores
Positive contribution %
Negative contribution %
Average SHAP values
Shap_Values.csv

Contains:

Individual record-level SHAP values
Driver contribution details
data_description.csv

Stores metadata including:

Dataset name
Model type
Target variable
Record count
Column count
Business description
shap_summary_minmax.png

Visual SHAP summary chart showing feature importance ranking.
