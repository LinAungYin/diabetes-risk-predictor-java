# diabetes-risk-predictor-java
PhD Showcase: Pure Java Logistic Regression for Diabetes Risk Prediction.
# Diabetes Risk Predictor: A Machine Learning Showcase for Clinical Applications
This project serves as a technical showcase demonstrating proficiency in developing interpretable machine learning (ML) models in a robust, enterprise-grade environment (Pure Java). The goal is to address the critical clinical need for proactive risk stratification of Type 2 Diabetes Mellitus (T2DM), aligning with the National University Health System (NUHS) and NUS Medicine's focus on computational biology and precision health.

The core implementation uses a foundational ML algorithm, Logistic Regression, to prioritize model transparency and explainability, which is paramount for clinical adoption.

## 1. Methodology: Interpretable Logistic Regression

The model is built from scratch in pure Java, without relying on external ML libraries, to demonstrate a deep understanding of the underlying mathematics. 
Logistic Regression is chosen for its simplicity and inherent interpretability. It predicts the probability of a binary outcome (Diabetes Risk: Yes/No) by calculating a weighted sum of the input features and passing it through the Sigmoid function.

$$P(\text{Diabetes}) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i x_i)}}$$

Where:

$\mathbf{P}$ is the predicted probability.

$\beta_i$ are the weights (feature importance).

$\beta_0$ is the Bias (baseline risk).

$x_i$ are the patient's feature values.

The output of the model explicitly shows the learned weights, indicating the clinical contribution of each input variable. 

## 2. Clinical Data and Evaluation

The model's performance is validated against a representative mock dataset (REAL_WORLD_DATASET) simulating a clinical cohort.

The model uses five widely recognized, non-invasive or easily available clinical factors for prediction:

Glucose: Plasma glucose concentration (2 hours after OGTT).

BMI: Body Mass Index (weight/height$^2$).

Age: Patient's age in years.

Pregnancies: Number of pregnancies (proxy for GDM history).

Diabetes Pedigree Function (DPF): A synthetic score quantifying family history and genetic predisposition.

## 3. Technical Implementation Details

Language: Pure Java (JDK 17+)

Architecture: Single, self-contained DiabetesRiskPredictor.java file.

Dependencies: None (Standard Java only).

Workflow: Demonstrates the full ML lifecycle: data structure, prediction, and validation (confusion matrix and metric calculation).

## 4. Proposed Future Work (PhD Roadmap)

This showcase is a foundational piece. A potential PhD trajectory would involve expanding this model to:

Data Integration: Replace mock data with large-scale, de-identified Electronic Health Record (EHR) data from NUHS to train a geographically specific Asian-population model.

Feature Engineering: Incorporate complex time-series data (e.g., longitudinal blood pressure, lab results, lipid panels) and metabolomic features (as researched by NUS Medicine).

Model Expansion: Implement and benchmark non-linear models (e.g., Random Forests, Neural Networks) against the interpretable Logistic Regression baseline to assess the trade-off between predictive accuracy and clinical explainability.

Deployment: Develop a Java Spring Boot microservice to integrate the model's predictive API into clinical decision support systems.
