# Recruitment Data Optimization Pipeline

## 📌 Project Overview
This project addresses a common data science bottleneck: processing large datasets for machine learning models. Working with a dataset from *Training Data Ltd.*, the goal was to optimize the storage efficiency of student data used to predict job-seeking behavior. By downcasting numerical data and converting object types to categorical structures, the pipeline significantly reduces memory overhead while preparing the data for predictive modeling.

## 🚀 Key Objectives
* **Memory Optimization:** Reduce the memory footprint of the customer dataset by converting standard `object`, `int64`, and `float64` types to more efficient formats.
* **Data Cleaning:** Handle null values and enforce strict data typing for ordinal and nominal variables.
* **Targeted Filtering:** Isolate a specific subset of candidates—those with extensive experience (>10 years) working in large enterprises (1000+ employees)—for a proof-of-concept model.

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Library:** Pandas (CategoricalDtype, astype conversions, conditional filtering)
* **Input Data:** `customer_train.csv` (Anonymized student recruitment data)

## 📊 Methodology & Optimization Strategy

### 1. Data Type Conversion
Standard Pandas types were converted to optimize memory usage:
* **Booleans:** `relevant_experience` and `job_change` were mapped to `True/False`.
* **Integers:** `student_id` and `training_hours` were downcasted to `int32`.
* **Floats:** `city_development_index` was downcasted to `float16`.

### 2. Categorical Encoding
To retain the logical order of specific variables, `CategoricalDtype` was applied with defined ordering:
* **Ordered Categories:** `education_level`, `experience`, `company_size`, and `enrolled_university`.
    * *Example Order:* `Primary School` < `High School` < `Graduate` < `Masters` < `Phd`.
* **Nominal Categories:** Remaining object columns (e.g., `city`, `gender`) were converted to standard categories.

### 3. Strategic Filtering
The dataset was filtered to identify high-value candidates for the recruitment drive:
* **Experience:** Students with **10 or more years** of experience.
* **Company Size:** Students currently employed at companies with **1000+ employees**.

## 💻 Code Highlight: Efficient Loop Processing
The project utilizes a loop to dynamically assign data types based on a pre-defined schema, ensuring scalability if new columns are added.

```python
# Efficiently looping through columns to apply memory-saving transformations
for col in ds_jobs_transformed:
    
    # Convert two-factor categories to bool
    if col in ['relevant_experience', 'job_change']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].map(two_factor_cats[col])
    
    # Downcast integers for memory savings
    elif col in ['student_id', 'training_hours']:
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype('int32')
    
    # Apply ordered categories
    elif col in ordered_cats.keys():
        category = pd.CategoricalDtype(ordered_cats[col], ordered=True)
        ds_jobs_transformed[col] = ds_jobs_transformed[col].astype(category)
