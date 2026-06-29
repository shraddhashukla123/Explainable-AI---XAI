# XAI Demand Forecasting & Driver Analysis Engine

## Overview

This project provides an Explainable AI (XAI) framework for Demand Forecasting and Business Driver Analysis. The solution identifies key factors influencing sales demand and quantifies their impact using statistical modeling and SHAP (SHapley Additive Explanations).

The system enables business users and analysts to:

* Forecast demand drivers
* Understand feature importance
* Quantify business impact of controllable and uncontrollable factors
* Generate explainable insights for decision-making
* Produce driver analysis reports at Region, Category, and Product levels

---

## Key Features

### Demand Driver Analysis

* Bivariate regression analysis
* Driver coefficient estimation
* Feature impact measurement
* Relative importance ranking

### Explainable AI (XAI)

* SHAP-based explainability
* Global feature importance
* Positive and negative contribution analysis
* Driver ranking based on business impact

### Automated Data Processing

* Automatic feature type detection
* Numeric data sanitization
* Missing value handling
* Feature transformation and normalization

### Multi-Level Analysis

The engine performs analysis across:

* Region
* Category Area
* Product

Each combination is independently analyzed and aggregated into final reports.

---

## Project Structure

```text
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
```

---

## Dataset Requirements

The input dataset must contain the following business dimensions and target variable:

### Mandatory Columns

| Column        | Description           |
| ------------- | --------------------- |
| Region        | Geographic region     |
| category_area | Product category      |
| Product       | Product identifier    |
| Sales_TY      | Target sales variable |

### Optional Driver Variables

The framework can analyze any additional business drivers such as:

* Price
* Promotions
* Discounts
* Weather indicators
* Economic indicators
* Holidays
* Events
* Seasonal factors
* Marketing spend
* Distribution metrics

---

## Processing Workflow

### Step 1: Load Dataset

The system loads the input dataset and validates mandatory columns.

### Step 2: Segment Data

Data is segmented hierarchically:

```text
Region
    └── Category Area
            └── Product
```

Each product group is analyzed independently.

### Step 3: Feature Engineering

The preprocessing layer performs:

* Data type detection
* Numeric conversion
* Missing value handling
* Feature scaling using Min-Max normalization
* Data quality validation

### Step 4: Driver Analysis

For each product segment, the system performs:

* Regression model training
* Driver coefficient estimation
* Feature impact measurement
* Business driver ranking

### Step 5: SHAP Explainability Analysis

The explainability layer calculates:

* SHAP values
* Feature importance
* Positive contribution percentage
* Negative contribution percentage
* Average feature impact

### Step 6: Output Generation

All analysis outputs and visualizations are generated and stored in the output directory.

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd xai_runner
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Execute the application using:

```bash
python main.py
```

The system will:

1. Load and validate the dataset
2. Generate dataset metadata
3. Process data by Region → Category → Product
4. Perform driver analysis
5. Generate SHAP explainability outputs
6. Save reports and visualizations

---

## Output Files

### Driver_Analysis_Coefficients.csv

Contains:

* Driver coefficients
* Driver impact estimates
* Feature ranking
* Business interpretation metrics

---

### Shap_Aggregate.csv

Contains:

* Overall feature importance
* Positive contribution percentage
* Negative contribution percentage
* Mean SHAP values

---

### Shap_Values.csv

Contains:

* Record-level SHAP values
* Feature contribution details
* Explainability outputs for each observation

---

### cleaned_raw_data.csv

Contains:

* Processed dataset after cleaning
* Standardized feature values
* Analysis-ready data

---

### column_summary.csv

Contains:

* Column names
* Data types
* Missing value statistics
* Summary information

---

### data_description.csv

Contains:

* Dataset metadata
* Analysis configuration
* Target variable information
* Record and column counts

---

### shap_summary_minmax.png

Visual representation of:

* Feature importance ranking
* SHAP impact distribution
* Driver comparison summary

---

## Business Interpretation

The framework helps answer key business questions such as:

### Which factors influence sales the most?

Identify the most important drivers affecting demand performance.

### What is the impact of each driver?

Quantify how changes in a driver affect the target KPI.

### Which drivers are positive or negative contributors?

Understand whether a factor increases or decreases sales demand.

### Which business levers should be prioritized?

Rank controllable factors to support strategic decision-making.

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* StatsModels
* Matplotlib

---

## Example Use Cases

### Retail Demand Forecasting

Identify:

* Price sensitivity
* Promotion effectiveness
* Seasonal demand patterns

### Consumer Goods Analytics

Measure the impact of:

* Weather conditions
* Economic indicators
* Holidays and events
* Distribution performance

### Sales & Revenue Planning

Support:

* Demand forecasting
* Inventory planning
* Sales strategy optimization
* Marketing effectiveness analysis

### Business Driver Analysis

Enable stakeholders to:

* Understand key demand drivers
* Explain forecast movements
* Improve planning accuracy
* Support data-driven decision-making

```
```
