# Quora Duplicate Question Classifier System 


**A lightweight, interpretable duplicate question detection system using hand-crafted lexical & fuzzy features**

**Live Demo** → (Deploy your Streamlit app and paste the link here)  
**GitHub Repository** → (Paste your repo link here)  
**Dataset** → [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)  

---

### Objective
Detect whether two questions are duplicates using only lexical patterns and string similarity — **no embeddings, no transformers, no LLMs**.  
Real-world goal: Help Q&A platforms like Quora automatically identify and merge near-duplicate questions to reduce redundancy and improve user experience.

---

### Key Features Engineered

| Category                  | Features                                                                                           |
|---------------------------|----------------------------------------------------------------------------------------------------|
| Token Overlap             | `cwc_min`, `cwc_max`, `ctc_min`, `ctc_max`, `csc_min`, `csc_max`, `first_word_eq`, `last_word_eq` |
| Length & Structure        | `abs_len_diff`, `mean_len`, `token_set_ratio`, `token_sort_ratio`                                  |
| Longest Substring Ratio   | `longest_substr_ratio` = Length of longest common substring / Avg length of both questions        |
| Fuzzy Matching (fuzzywuzzy) | `fuzz_ratio`, `fuzz_partial_ratio`, `fuzz_token_sort_ratio`, `fuzz_token_set_ratio`                |
| Others                    | Question mark count, special character count, numbers presence, etc.                               |

---

### Text Preprocessing Pipeline
- Lowercase + strip whitespace  
- Expand contractions (`don't` → `do not`, etc.)  
- Replace currency/symbols with words (`$` → `dollar`, `₹` → `rupee`)  
- Remove/replace noisy patterns: `[math]`, HTML tags, excessive punctuation  
- Smart handling of apostrophes and repeated characters  

---

### Exploratory Data Analysis & Visualization

#### Pairplot Insights (ctc_min, cwc_min, csc_min)
- Strong separation on `ctc_min` & `cwc_min` → duplicates share both total and meaningful words  
- `csc_min` (stopwords) adds moderate signal but overlaps heavily with non-duplicates  
- Best discrimination comes from combining content-word overlap features  

#### t-SNE 2D Projection
- Duplicate pairs form tight, well-defined clusters  
- Non-duplicates are widely scattered  
- Clear visual proof that hand-crafted features capture similarity effectively  

(Insert your t-SNE plot here)  
(Insert your pairplot screenshot here)

---

### Modeling & Results

| Model             | Accuracy | Log Loss | Notes                                      |
|-------------------|----------|----------|--------------------------------------------|
| XGBoost           | 79.2%    | 0.482    | Slightly higher accuracy                   |
| **Random Forest** | **78.4%**| **0.491**| **Chosen** — significantly fewer false positives |

**Final Model**: Random Forest (prioritizes lower false positive rate for real-world use)

![Alt text](https://github.com/Rustam-Shrestha/quora_lexical_duplicate_classifier/tree/main/assets)


![Screenshot 1](assets/Screenshot%20from%202025-11-21%2018-37-15.png)
![Screenshot 2](assets/Screenshot%20from%202025-11-21%2018-37-41.png)
![Screenshot 3](assets/Screenshot%20from%202025-11-21%2018-38-01.png)
![Screenshot 4](assets/Screenshot%20from%202025-11-21%2018-38-23.png)
![Screenshot 5](assets/Screenshot%20from%202025-11-21%2018-38-42.png)
![Screenshot 6](assets/Screenshot%20from%202025-11-21%2018-38-59.png)
![Screenshot 7](assets/Screenshot%20from%202025-11-21%2018-39-11.png)

# diagram
https://github.com/Rustam-Shrestha/quora_lexical_duplicate_classifier/tree/main/assets



---

## Tasks Completed

### 1. Token-Based Feature Extraction

We reviewed and interpreted key lexical overlap features used in NLP tasks like Quora question pair classification:

| Feature           | Description |
|------------------|-------------|
| `cwc_min/max`    | Common word count (excluding stopwords) |
| `csc_min/max`    | Common stopword count |
| `ctc_min/max`    | Common token count (including stopwords) |
| `first_word_eq`  | Boolean: whether both questions start with the same word |
| `last_word_eq`   | Boolean: whether both questions end with the same word |

These features were extracted using a custom function and mapped into the dataframe for modeling.

---

### 2. Longest Substring Ratio

Defined and explained the **Longest Substring Ratio**, a metric that captures the longest continuous sequence of shared words or characters between two questions:



\[
\text{LSR} = \frac{\text{Length of Longest Common Substring}}{\text{Average Length of Both Sentences}}
\]



Used for semantic similarity and duplicate detection.

---

### 3. Fuzzy Matching Metrics

Explored four key fuzzy string similarity metrics:

| Metric               | Description |
|----------------------|-------------|
| `fuzz.ratio`         | Levenshtein similarity |
| `fuzz.partial_ratio` | Best substring match |
| `token_sort_ratio`   | Sorts tokens before comparing |
| `token_set_ratio`    | Compares shared and unique tokens using set logic |

These were recommended for feature engineering in text similarity tasks.

---

### 4. Preprocessing Pipeline

Reviewed and enhanced a robust preprocessing function that:

- Lowercases and strips whitespace
- Replaces special characters (`%`, `$`, `₹`, `€`, `@`) with semantic equivalents
- Removes `[math]` tags and compresses large numbers (`k`, `m`, `b`)
- Expands contractions using a comprehensive dictionary
- Removes HTML tags and punctuation
- Handles apostrophes and stopwords



```python
pattern = re.compile(r'\W')  # Use raw string literal

# quora_semantic_duplicate_classifier_system-
# quora_semantic_duplicate_classifier_system-

# Duplicate Question Detection Workflow — Daily Exploration Summary

This document captures the full exploration of feature engineering, visualization, modeling, and semantic interpretation for duplicate question detection. It includes pairplot analysis, t-SNE interpretation, model pipelines, confusion matrix evaluation, and a flowchart of the prediction system.

---

##  Pairplot Analysis (ctc_min, cwc_min, csc_min)

**Features:**
- `ctc_min`: Common token count (includes all words)
- `cwc_min`: Common word count (excludes stopwords)
- `csc_min`: Common stopword count (only filler words)

**Observations:**
- `ctc_min` vs `cwc_min`: Strong separation — duplicates share both total and meaningful words.
- `ctc_min` vs `csc_min`: Moderate signal — duplicates share stopwords, but overlap with non-duplicates.
- `cwc_min` vs `csc_min`: Weak alone — some rising pattern, but not fully separable.

**Takeaways:**
- Prioritize `ctc_min` and `cwc_min` for classification.
- Use `csc_min` as a supporting feature.
- Combine features to boost signal.

---

## t-SNE Interpretation (2D)

**Goal:** Visualize high-dimensional question pair features in 2D.

**Insights:**
- Duplicates (blue) form tight clusters → strong semantic similarity.
- Non-duplicates (red) are more scattered → diverse phrasing.
- Overlap zones indicate ambiguous cases.
- Outliers may be noisy or rare patterns.

**Takeaways:**
- Features capture structure well.
- Use t-SNE to justify feature selection and model strategy.

---

## Model Pipelines (Random Forest & XGBoost)

###  Data Preparation
```python
X = final_df.iloc[:, 1:].values
y = final_df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




---

### Model Training Pipeline (Placeholder)
```python
# X = features, y = is_duplicate
# X_train, X_test, y_train, y_test = train_test_split(...)
# model = RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# → Evaluate accuracy, log loss, confusion matrix
