# Quora Duplicate Question Classifier System 

![Alt text](https://github.com/santosh-ai/duplicate-detector/blob/main/assets/diagram.png)


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

Also resolved a `SyntaxWarning` by correcting regex usage:

```python
pattern = re.compile(r'\W')  # Use raw string literal

# quora_semantic_duplicate_classifier_system-
# quora_semantic_duplicate_classifier_system-
