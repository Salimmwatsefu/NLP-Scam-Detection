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



