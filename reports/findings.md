# Preprocessing Findings


1. **Empty Messages**: Out of 545 messages, 6 (1.1%) ended up with no usable text after cleaning. These were likely empty or had only things like special characters that got removed.

2. **URLs Removed**: No URLs were left in the cleaned data (0%), meaning all web links were successfully removed.

3. **Token Stats**:
   - On average, each message had about 15 words (tokens) after cleaning.
   - The dataset used 1,794 unique words (vocabulary size).
   - No random symbols or non-alphabetic tokens remained, which is what we wanted.

5. **BERT Processing**:
   - All 545 messages (100%) were properly formatted for BERT with special markers (CLS and SEP), so they’re ready for analysis.
   - Sample numeric IDs for words (input IDs) were created correctly, as shown below.

6. **Sample Messages**:
   
   - **Message 1**:
     - **Original**: CONGRATULATION!\nYOUR ACCOUNT 254757547986 HAS BEEN CREDITED WITH KES 62,950\n\nNew BONUS Balance: KES 62,950 \n\nLOGIN>wekelea.com\n \n DEPOSIT&PLAY
     - **Cleaned**: congratulation account credit kes new bonus balance kes login deposit play
     - **Tokens**: ['congratulation', 'account', 'credit', 'kes', 'new', 'bonus', 'balance', 'kes', 'login', 'deposit', 'play']
     - **BERT Input IDs**: [101, 26478, 8609, 9513, 4070, 4923, 17710, 2015, 2047, 6781, 5703, 17710, 2015, 8833, 2378, 12816, 2377, 102, 0, 0, ...]
   - **Message 2**:
     - **Original**: 🙏🙏 I can do all this through him who gives me strength. Phil 4;13 Reply with 20 To stop this message reply with STOP
     - **Cleaned**: give strength phil reply stop message reply stop
     - **Tokens**: ['give', 'strength', 'phil', 'reply', 'stop', 'message', 'reply', 'stop']
     - **BERT Input IDs**: [101, 2507, 3997, 6316, 7514, 2644, 4471, 7514, 2644, 102, 0, 0, ...]
   
   - **Message 3**:
     - **Original**: TAL6FH2DN CONFIRMED, YOU HAVE RECEIVED KES. 70,350\n\nUSIPITWE NA HII CHEPKORIR INGIA> pepea.ke\n\nSHARE YAKO INAKUNGOJA LEO\n\nDEPOSIT 99/-PLAY&WIN
     - **Cleaned**: receive kes usipitwe hii chepkorir ingia pepea share yako leo deposit play win
     - **Tokens**: ['receive', 'kes', 'usipitwe', 'hii', 'chepkorir', 'ingia', 'pepea', 'share', 'yako', 'leo', 'deposit', 'play', 'win']
     - **BERT Input IDs**: [101, 4449, 17710, 2015, 1057, 28036, 23737, 2571, 1045, 16655, 26240, 2386, 13699, 10730, 10893, 12816, 2377, 2663, 102, 0, 0, ...]

In summary, the preprocessing worked well to clean the data, remove URLs, and prepare messages for BERT analysis. Some names remained like Chepkorir, but the dataset is now organized and ready.





# Scam Detection Model Findings

This document summarizes the performance of three machine learning models, **Logistic Regression**, **XGBoost**, and **BERT**, used to detect scam messages in a dataset of text communications. The goal is to identify whether a message is a scam or legit (legitimate).

## Dataset Overview

The dataset contains **545 messages**, with:

* **348 scams** (labeled as 1).
* **197 legitimate** messages (labeled as 0).

Each message was converted into numerical features using **TF-IDF** (a method that captures important words) for Logistic Regression and XGBoost, resulting in **3447 features per message**. BERT uses its own internal tokenization and embedding process. The dataset was split into **80% training** and **20% testing** (**109 test messages**) to evaluate the models.

## Model Performance

We evaluated the models using five metrics:

* **Accuracy**: Percentage of correct predictions (scam or legit).
* **Precision**: How often a "scam" prediction is correct.
* **Recall**: How well the model catches all actual scams.
* **F1-score**: A balance between precision and recall.
* **ROC-AUC**: How well the model distinguishes scams from legit messages (higher is better).

### Logistic Regression

**Overview**: Logistic Regression is a simple model that predicts whether a message is a scam based on word patterns captured by TF-IDF features.

**Results**:

* **Accuracy**: **82.57%** (90 out of 109 test messages correctly classified).
* **Precision**: **78.65%** (some legit messages were incorrectly flagged as scams).
* **Recall**: **100%** (caught every actual scam).
* **F1-score**: **88.05%** (good balance, but precision could improve).
* **ROC-AUC**: **98.06%** (excellent at separating scams from legit messages).

**Confusion Matrix**:

* 20 legit messages correctly identified.
* 70 scam messages correctly identified.
* 19 legit messages incorrectly flagged as scams.
* 0 scams missed (perfect recall).

**Summary**: Logistic Regression is highly effective at catching all scams (**100% recall**), but it flags some legitimate messages as scams (lower precision). It’s a good starting point but may need tuning to reduce false positives.

**Raw Output**:

```
Loaded data shape: (545, 20)
TF-IDF matrix shape: (545, 3447)
Label distribution:
label
1    348
0    197
Name: count, dtype: int64
Training Logistic Regression...
Metrics: {'Accuracy': 0.8256880733944955, 'Precision': 0.7865168539325843, 'Recall': 1.0, 'F1-score': 0.8805031446540881, 'ROC-AUC': 0.9805860805860805}
Confusion Matrix:
 [[20 19]
 [ 0 70]]
Logistic Regression complete. Model and metrics saved.
```

### XGBoost

**Overview**: XGBoost is a more advanced model that uses decision trees to learn complex patterns in the data, using TF-IDF features.

**Results**:

* **Accuracy**: **93.58%** (102 out of 109 test messages correctly classified).
* **Precision**: **94.37%** (most "scam" predictions were correct).
* **Recall**: **95.71%** (missed only a few scams).
* **F1-score**: **95.04%** (excellent balance of precision and recall).
* **ROC-AUC**: **94.80%** (very good at distinguishing scams from legit messages).

**Confusion Matrix**:

* 35 legit messages correctly identified.
* 67 scam messages correctly identified.
* 4 legit messages incorrectly flagged as scams.
* 3 scams missed.

**Summary**: XGBoost performs exceptionally well, with high accuracy, precision, and recall. It misses very few scams and has fewer false positives than Logistic Regression, making it a strong candidate for scam detection.

**Raw Output**:

```
Loaded data shape: (545, 20)
TF-IDF matrix shape: (545, 3447)
Label distribution:
label
1    348
0    197
Name: count, dtype: int64
Training XGBoost...
Metrics: {'Accuracy': 0.9357798165137615, 'Precision': 0.9436619718309859, 'Recall': 0.9571428571428572, 'F1-score': 0.950354609929078, 'ROC-AUC': 0.947985347985348}
Confusion Matrix:
 [[35  4]
 [ 3 67]]
XGBoost complete. Model and metrics saved.
```

### BERT

**Overview**: BERT is a deep learning model specifically designed for understanding text context and nuances, trained here to classify messages.

**Results**:

* **Accuracy**: **85.32%** (93 out of 109 test messages correctly classified - calculated from confusion matrix: 25+68=93).
* **Precision**: **82.93%** (better than Logistic Regression, lower than XGBoost).
* **Recall**: **97.14%** (better than XGBoost, lower than Logistic Regression).
* **F1-score**: **89.47%** (better than Logistic Regression, lower than XGBoost).
* **ROC-AUC**: **89.23%** (lower than both Logistic Regression and XGBoost).

**Confusion Matrix**:

* 25 legit messages correctly identified.
* 68 scam messages correctly identified.
* 14 legit messages incorrectly flagged as scams.
* 2 scams missed.

**Summary**: BERT shows good performance, achieving higher recall than XGBoost but lower precision and overall metrics like F1-score and ROC-AUC. It has fewer false positives than Logistic Regression but more than XGBoost.

**Raw Output**:

```
Metrics: {'Accuracy': 0.8532110091743119, 'Precision': 0.8292682926829268, 'Recall': 0.9714285714285714, 'F1-score': 0.8947368421052632, 'ROC-AUC': 0.8923076923076922}
Confusion Matrix:
 [[25 14]
 [ 2 68]]
```

## Key Observations

* **XGBoost** remains the highest performing model overall, leading in Accuracy, Precision, F1-score, and ROC-AUC.
* **Logistic Regression** achieved perfect Recall (100%), meaning it missed no scams, but at the cost of the lowest Precision (most false positives).
* **BERT** showed strong Recall (97.14%), better than XGBoost, but lagged behind XGBoost in Precision, F1-score, and ROC-AUC. It sits between Logistic Regression and XGBoost in terms of false positives and missed scams.
    * XGBoost: 4 false positives, 3 missed scams.
    * BERT: 14 false positives, 2 missed scams.
    * Logistic Regression: 19 false positives, 0 missed scams.

* **Dataset Imbalance**: The dataset has more scams (348) than legit messages (197). The models' varying trade-offs between Precision (avoiding false positives) and Recall (catching all scams) reflect how they handle this imbalance.


## Conclusion

**XGBoost** is currently the best-performing model, offering the strongest balance across most key metrics (Accuracy, Precision, F1-score, ROC-AUC) and minimizing false positives while still catching most scams. **Logistic Regression** is best if the absolute priority is catching every single scam, provided a higher rate of false positives is acceptable. **BERT** performs well, especially in terms of Recall, but does not surpass XGBoost on this dataset with the current training. 

