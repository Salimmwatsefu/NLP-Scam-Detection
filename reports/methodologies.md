# Methods used and explanations. It is personal and will be deleted afterwards

## 1. Step-1 Data Labelling

Used Label Studio for labelling ( correcting ) the prelabeled data.
Includes a prelabel column which is either 0: Legit (Low-risk), 1:Moderate risk Scam, 2: High risk scam. The the label columns classifies them as either legit or scam.
Used the prelabel.py and label.py scripts under the scripts folder.

## 2. Text preprocessing

Transforms raw sms messages from scam_detection_labeled.csv dataset into clean structured features suitable for traditional methods like TF-IDF and deep learning like BERT embeddings.

preprocessing.ipynb has 3 main components:

### 1.  def preprocess_text - function ( cell 4)

##### Text cleaning and tokenization
Steps:
1. Lowercasing
2. Noise Removal
    - replace newlines ((\n, \\n)) with spaces
    - remove non-ASCII characters to handle emojis or special symbols.
    - remove urls
    - remove emails
    - remove phone numbers ( included also kenyan formats, with & without +)
    - remove transaction codes
    - replace special characters ( >, -) with spaces

3. Name Removal - applied spaCy's en_core_web_sb NER twice to detect PERSON entities ( eg. Christa) and remove them

4. Tokenization and lemmatization
    - Generate tokens using spaCy
    - Remove stopwords
    - require length of tokens > 1
    - Lemmatize tokens to their base forms
    - Joined tokens together into cleaned_text for TF-IDF

 


### 2.  def batch_bert_tokenize - function ( cell 5)
Input is the cleaned_text

1. Calculate Max_Length
Compute the longest tokenized sequence across all texts using BERT's bert-base-uncased tokenizer, capped at 200 tokens.
Skip empty texts to avoid errors, using only valid texts (filtered with text.strip()).

2. Batch Processing

Split texts into batches (size=32) for memory efficiency.
Replace empty texts with "placeholder" to ensure valid tokenization.
Tokenize each batch with:
padding="max_length": Pad sequences to max_len (200) with zeros.
truncation=True: Truncate longer texts to max_len.
return_tensors="np": Output NumPy arrays for input_ids.

Output - Concatenate batch input-ids into a single array
       - Each rows starts with [101] (CLS), ends with [102] (SEP) padded with zeros

###### Reason for this
1. BERT models require input_ids with fixed lengths starting with [101] and ending with [102], to generate contextual embeddings for scam classification.

2. Handles large datasets (545 rows) without memory overload, as BERT tokenization is resource-intensive.
Batch size 32 is a practical default.

3. NumPy Output -  Matches tools (TensorFlow) for BERT modeling.

Goal - Produce bert_input_ids ready for BERT embeddings, capturing scam patterns in a format models can process.



### 3.  application step ( cell 6)

1. Apply Preprocessing
2. Run batch_bert_tokenizer on cleaned_text, storing berch_input_ids as DF column
3. Save into a pickle file (.pkl)

###### Why a .pkl file
1. Saves preprocessed data as a binary file for fast reloading in feature engineering (e.g., TF-IDF, BERT training).
2. Preserves complex structures (tokens lists, bert_input_ids arrays) vs. CSV.

Goal - Validate preprocessing, ensure data is model-ready, and enable quick iteration in later steps (e.g., classification).

## SHAP Analysis Methodology

### Overview
SHAP (SHapley Additive exPlanations) analysis is used to understand how different features contribute to the model's predictions. This helps us identify which psychological triggers and patterns are most influential in scam detection.

### Implementation Steps
1. **Feature Importance Calculation**
   - Use SHAP values to quantify the impact of each feature
   - Calculate both global (overall) and local (per-instance) importance
   - Focus on psychological trigger features

2. **Trigger Taxonomy Mapping**
   - Map features to psychological trigger categories:
     - Greed triggers (e.g., money, win, bonus)
     - Urgency triggers (e.g., now, today, limited)
     - Authority triggers (e.g., official, secure, verify)
     - Fear triggers (e.g., account, security, warning)
     - Social proof triggers (e.g., others, people, users)
   - Calculate aggregate impact for each trigger category

3. **Analysis Approach**
   - Compare trigger importance across different scam types
   - Identify patterns in how triggers are used together
   - Analyze the relationship between trigger combinations and scam risk levels

## Demographic Analysis Methodology

### Overview
The demographic analysis approach aims to understand how scam patterns vary across different demographic groups and improve detection by considering demographic-specific characteristics.

### Implementation Steps
1. **Demographic Inference**
   - Analyze message content for demographic indicators
   - Categorize messages into demographic groups:
     - General population
     - Low-income individuals
     - Youth
   - Use linguistic patterns and content markers for classification

2. **Demographic-Aware Model Training**
   - Train separate models for each demographic group
   - Incorporate demographic features into the model
   - Apply demographic-specific weights to features
   - Use both XGBoost and Logistic Regression models

3. **Model Evaluation**
   - Evaluate performance across demographic groups
   - Calculate metrics for each message type:
     - High-risk scams
     - Moderate-risk scams
     - Legitimate messages
   - Compare model performance between demographic groups

### Technical Implementation
1. **Feature Engineering**
   - Extract demographic indicators from text
   - Create demographic-specific feature sets
   - Normalize features across demographic groups

2. **Model Training**
   - Split data by demographic groups
   - Train models with demographic weights
   - Implement cross-validation for each group

3. **Performance Metrics**
   - Calculate accuracy, precision, recall, and F1-score
   - Evaluate performance per demographic group
   - Compare results across different message types



