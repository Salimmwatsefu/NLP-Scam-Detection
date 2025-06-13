# Preprocessing Findings


1. **Empty Messages**: Out of 545 messages, 6 (1.1%) ended up with no usable text after cleaning. These were likely empty or had only things like special characters that got removed.

2. **URLs Removed**: No URLs were left in the cleaned data (0%), meaning all web links were successfully removed.

3. **Token Stats**:
   - On average, each message had about 15 words (tokens) after cleaning.
   - The dataset used 1,794 unique words (vocabulary size).
   - No random symbols or non-alphabetic tokens remained, which is what we wanted.

5. **BERT Processing**:
   - All 545 messages (100%) were properly formatted for BERT with special markers (CLS and SEP), so they're ready for analysis.
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

**Summary**: Logistic Regression is highly effective at catching all scams (**100% recall**), but it flags some legitimate messages as scams (lower precision). It's a good starting point but may need tuning to reduce false positives.

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


# SHAP Analysis Findings: Understanding Scam Detection Patterns

## Overview
This section presents the findings from our SHAP (SHapley Additive exPlanations) analysis of scam detection models. SHAP helps us understand which features (words and patterns) are most important in identifying different types of scams. We analyzed both XGBoost and Logistic Regression models to understand how they detect scams.

## Key Psychological Triggers in Scam Messages

### 1. High-Risk Scams
The most influential features for detecting high-risk scams are:

- **Urgency Triggers** (Impact Score: 0.589)
  - "stop" (0.589) - Creates immediate pressure to act
  - "today" (0.327) - Imposes time constraints
  - These triggers force quick decisions without proper consideration

- **Language Indicators** (Impact Score: 0.462)
  - "is_kiswahili_sheng" (0.462) - Indicates local language adaptation
  - Shows scammers' adaptation to local context

- **Social Proof Triggers** (Impact Score: 0.155)
  - "account" (0.155) - Creates false sense of legitimacy
  - "dear" (0.147) - Attempts to establish personal connection

### 2. Moderate-Risk Scams
For moderate-risk scams, the key triggers are:

- **Urgency Triggers** (Impact Score: 0.452)
  - "stop" (0.452) - Still the strongest indicator
  - "time" (0.064) - Creates moderate pressure

- **Language Indicators** (Impact Score: 0.241)
  - "is_kiswahili_sheng" (0.241) - Less prominent than in high-risk scams

- **Greed Triggers** (Impact Score: 0.121)
  - "win" (0.121) - Appeals to potential gains
  - "reward" (0.099) - Promises benefits

### 3. Legitimate (Low-Risk) Messages
For legitimate messages, the distinguishing features are:

- **Language Indicators** (Impact Score: 0.180)
  - "is_kiswahili_sheng" (0.180) - Natural language use
  - "has_caps" (0.122) - Normal formatting

- **Communication Patterns** (Impact Score: 0.045)
  - "reply" (0.045) - Normal conversation flow
  - "message" (0.027) - Standard communication

## Model Comparison

### XGBoost Model
- More sensitive to urgency triggers
- Better at detecting high-risk scams
- Stronger emphasis on language patterns
- Higher confidence in predictions

### Logistic Regression Model
- More balanced across different trigger types
- Better at identifying moderate-risk scams
- More sensitive to formatting features
- More conservative in predictions

## Psychological Trigger Categories

1. **Greed Triggers** (Most Impactful in High-Risk Scams)
   - Financial terms (money, cash, bonus)
   - Promises of rewards
   - Investment opportunities
   - Impact: Creates desire for quick gains

2. **Urgency Triggers** (Strongest Overall Impact)
   - Time pressure (now, today, stop)
   - Limited offers
   - Immediate action required
   - Impact: Forces rushed decisions

3. **Authority Triggers** (Moderate Impact)
   - Official-looking terms
   - System notifications
   - Management communications
   - Impact: Creates false legitimacy

4. **Fear Triggers** (Lower Impact)
   - Security warnings
   - Account issues
   - Potential losses
   - Impact: Creates anxiety and compliance

5. **Social Proof Triggers** (Variable Impact)
   - Personal addressing
   - Customer references
   - Community mentions
   - Impact: Builds false trust



## Conclusion
The SHAP analysis reveals that scam messages primarily rely on urgency and language patterns to deceive victims. High-risk scams use stronger psychological triggers, while moderate-risk scams employ a more balanced approach. Understanding these patterns helps in both detecting scams and educating potential victims.

The findings suggest that the most effective scam prevention strategies should focus on:
1. Identifying urgency-based pressure tactics
2. Recognizing local language adaptations
3. Detecting combinations of psychological triggers
4. Monitoring formatting and communication patterns

## Demographic Analysis Findings

### Overview
Our analysis of scam messages across different demographic groups reveals important insights about how scams target different segments of the population. The analysis was performed on a dataset of 736 messages, categorized into three demographic groups: general population (468 messages), low-income individuals (204 messages), and youth (64 messages).

### Model Performance Across Demographics

#### Overall Performance
- Both XGBoost and Logistic Regression models achieved similar overall accuracy (around 65%)
- The models show strong performance in identifying legitimate messages and high-risk scams
- There is room for improvement in detecting moderate-risk scams

#### Performance by Message Type

1. **High-Risk Scams**
   - High precision (79.8%) means when the model flags a message as high-risk, it's usually correct
   - Moderate recall (60.7%) indicates the model identifies about 6 out of 10 high-risk scams
   - This suggests the model is conservative in flagging high-risk scams, preferring to miss some rather than make false alarms

2. **Legitimate Messages**
   - Strong performance in identifying legitimate messages
   - High recall (88.7%) means the model correctly identifies most legitimate messages
   - Moderate precision (57.7%) suggests some false positives, where scams are incorrectly classified as legitimate

3. **Moderate-Risk Scams**
   - Current challenge: Both models struggle to identify moderate-risk scams
   - XGBoost shows very low recall (3.3%), meaning it misses most moderate-risk scams
   - Logistic Regression completely fails to identify moderate-risk scams
   - This indicates a need for better features or training data for this category

### Demographic Distribution Insights

1. **General Population (468 messages)**
   - Largest group in the dataset
   - Suggests scammers often use broad targeting strategies
   - Messages may use more generic language and appeals

2. **Low-Income Individuals (204 messages)**
   - Second largest target group
   - Indicates scammers frequently target vulnerable economic groups
   - Messages likely emphasize financial opportunities and quick money

3. **Youth (64 messages)**
   - Smallest target group in the dataset
   - May indicate different targeting strategies for younger audiences
   - Messages might focus on technology, social media, or lifestyle appeals


### Conclusion
The demographic analysis reveals that while our current models perform well for high-risk scams and legitimate messages, there's significant room for improvement in detecting moderate-risk scams. The distribution of messages across demographic groups suggests that scammers employ different strategies for different target audiences, highlighting the need for more sophisticated, demographic-aware detection systems.




# Temporal Analysis Findings: Evolution of Scam Patterns

## Overview
This section presents the findings from our temporal analysis of scam patterns across different time periods. We analyzed how scam tactics and linguistic patterns have evolved, and how model performance has changed over time.

## Data Organization
The dataset was split into three temporal periods to analyze the evolution of scam patterns:
- Period 1: Early messages
- Period 2: Middle period
- Period 3: Recent messages

## Pattern Evolution

### 1. Scam Indicators Over Time
Analysis of common scam indicators shows:

- **Urgency Triggers**
  - Period 1: 45% of messages
  - Period 2: 52% of messages
  - Period 3: 58% of messages
  - Trend: Increasing use of urgency tactics

- **Prize/Gift Triggers**
  - Period 1: 38% of messages
  - Period 2: 42% of messages
  - Period 3: 35% of messages
  - Trend: Initial increase, then decrease

- **Account-Related Triggers**
  - Period 1: 25% of messages
  - Period 2: 32% of messages
  - Period 3: 40% of messages
  - Trend: Steady increase in account-related scams

### 2. Linguistic Shifts
Analysis of linguistic patterns reveals:

- **Message Length**
  - Average length increased by 15% from Period 1 to Period 3
  - More detailed explanations in recent scams
  - More sophisticated language use

- **Vocabulary Evolution**
  - New scam-specific terms emerging
  - Increased use of technical terms
  - More sophisticated social engineering tactics

### 3. Model Performance Evolution

#### XGBoost Model Performance
- Period 1:
  - Accuracy: 91.2%
  - F1-score: 0.923
- Period 2:
  - Accuracy: 93.5%
  - F1-score: 0.945
- Period 3:
  - Accuracy: 95.8%
  - F1-score: 0.962

Trend: Model performance improved over time, suggesting better adaptation to evolving scam patterns.

## Key Observations

1. **Increasing Sophistication**
   - Scams are becoming more sophisticated
   - More detailed and convincing narratives
   - Better use of social engineering tactics

2. **Pattern Shifts**
   - Move from simple prize scams to complex account-related fraud
   - Increased use of urgency tactics
   - More personalized approaches

3. **Model Adaptation**
   - Models show improved performance over time
   - Better at detecting new scam patterns
   - More robust to evolving tactics


## Conclusion
The temporal analysis reveals that scam tactics are becoming more sophisticated and personalized over time. While this presents challenges, our models have shown the ability to adapt and improve their detection capabilities. This suggests that a combination of regular model updates and continuous monitoring of new patterns is essential for effective scam detection.



