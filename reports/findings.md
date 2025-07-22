
# Scam Detection Model Findings

This document summarizes the performance of two machine learning models, **Logistic Regression** and **XGBoost**, used to detect scam messages in a dataset of text communications. The goal is to classify whether a message is **legitimate** or **scam**.

## Dataset Overview

The dataset contains 738 messages categorized into two classes:

* **Legitimate messages** (labeled as 0) - 311
* **Scam messages** (labeled as 1) - 425

Each message was converted into numerical features using **TF-IDF**, resulting in a set of features per message. The dataset was split into training and testing sets to evaluate the models.

## Model Performance

We evaluated the models using several metrics:

* **Accuracy**: Percentage of correct predictions across both classes
* **Precision**: How often predictions for each class are correct
* **Recall**: How well the model catches all instances of each class
* **F1-score**: A balance between precision and recall

### XGBoost

**Overview**: XGBoost is an advanced model that uses decision trees to learn complex patterns in the data, using TF-IDF features.

**Training Results**

* **Accuracy**: **93.88%**
* **Precision (Weighted)**: **94.20%**
* **Recall (Weighted)**: **93.88%**
* **F1-score (Weighted)**: **93.91%**

**Test Results**

* **Accuracy**: **83.78%**
* **Precision (Weighted)**: **83.87%**
* **Recall (Weighted)**: **83.78%**
* **F1-score (Weighted)**: **83.81%**

**Per-Class Performance (Test)**

* **Legitimate Messages**

  * Precision: 80.00%
  * Recall: 82.54%
  * F1-score: 81.25%

* **Scam Messages**

  * Precision: 86.75%
  * Recall: 84.71%
  * F1-score: 85.71%

**Summary**: XGBoost demonstrates strong training performance but sees a performance drop on the test set, indicating some generalization challenges. It performs slightly better on scam messages compared to legitimate messages.

### Logistic Regression

**Overview**: Logistic Regression is a simpler model that predicts message categories based on word patterns captured by TF-IDF features.

**Training Results**

* **Accuracy**: **91.33%**
* **Precision (Weighted)**: **91.57%**
* **Recall (Weighted)**: **91.33%**
* **F1-score (Weighted)**: **91.24%**

**Test Results**

* **Accuracy**: **78.38%**
* **Precision (Weighted)**: **79.44%**
* **Recall (Weighted)**: **78.38%**
* **F1-score (Weighted)**: **77.61%**

**Per-Class Performance (Test)**

* **Legitimate Messages**

  * Precision: 84.44%
  * Recall: 60.32%
  * F1-score: 70.37%

* **Scam Messages**

  * Precision: 75.73%
  * Recall: 91.76%
  * F1-score: 82.98%

**Summary**: Logistic Regression achieves slightly lower test accuracy compared to XGBoost but shows stronger performance in detecting scam messages (high recall). It struggles with legitimate messages due to a lower recall rate.

## Key Observations

1. **Model Comparison**

   * XGBoost outperforms Logistic Regression in overall test accuracy (83.78% vs 78.38%).
   * Logistic Regression captures more scam messages (higher recall for scams) but misclassifies more legitimate messages.

2. **Class Performance**

   * Both models show better detection for scams compared to legitimate messages.
   * Logistic Regression sacrifices precision on scams for higher recall, while XGBoost maintains a better balance.

3. **Training vs Test Performance**

   * Both models show a performance gap between training and test sets, indicating potential overfitting.
   * XGBoost has a higher training accuracy, which aligns with its complexity.

## Conclusion

Both models are effective at distinguishing between scam and legitimate messages, with **XGBoost providing a better overall balance of precision and recall**. Logistic Regression, while slightly less accurate overall, is more aggressive in detecting scams due to its higher recall on scam messages.


