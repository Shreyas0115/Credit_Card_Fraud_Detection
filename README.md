# 💳 Credit Card Fraud Detection 

## 📂 Input
- A **CSV file** with credit card transactions.  
- Target column (fraud indicator) guessed automatically: `Class`, `class`, `Fraud`, `fraud`, `is_fraud`.  
- Fraud = 1, Legit = 0.  

---

## 🚀 Steps Performed in the Code

### 1. Load & Identify Target Column
- Reads dataset with **pandas**.  
- Detects target column automatically if not specified.  
- Splits features into:  
  - **Numeric columns** (amount, time, V1–V28 PCA components, etc.)  
  - **Categorical columns** (if present, e.g., transaction type).  

**What I learned**: Auto-detecting target helps with messy real-world datasets. Separating numeric vs categorical features allows flexible preprocessing.

---

### 2. Preprocessing with ColumnTransformer
- **Numeric pipeline**:  
  - Fill missing values with median (`SimpleImputer`).  
  - Standardize scale with `StandardScaler`.  
- **Categorical pipeline**:  
  - Fill missing values with most frequent.  
  - Encode categories via `OneHotEncoder`.  
- Combine both using `ColumnTransformer`.

**What I learned**: Scaling numeric values stabilizes Logistic Regression. Encoding categorical features allows models to use all information.

---

### 3. Train & Evaluate Models
Two classifiers are trained (with **class imbalance handling**):  
- **Logistic Regression** (`class_weight="balanced"`)  
- **Random Forest** (400 trees, `class_weight="balanced_subsample"`)  

For each model:  
- Train on training set.  
- Predict probabilities on test set.  
- Compute:  
  - **ROC-AUC** (measures ranking quality)  
  - **PR-AUC** (precision-recall area; more sensitive for imbalanced fraud detection).  

**What I learned**: Fraud datasets are highly imbalanced — PR-AUC is a better metric than accuracy or even ROC-AUC.

---

### 4. Model Selection
- Selects the **best model** by **PR-AUC** (preferred) or ROC-AUC (fallback).  
- Prints side-by-side comparison for both metrics.  

**What I learned**: Selecting by PR-AUC ensures the chosen model is good at detecting rare fraud cases without too many false alarms.

---

### 5. Detailed Evaluation
- For the **best model**:  
  - Print **classification report** (precision, recall, F1).  
  - Show **confusion matrix**.  
  - Report **ROC-AUC** and **PR-AUC** again for clarity.  

**What I learned**: Precision and recall give insight into trade-offs — e.g., recall is critical in fraud detection (don’t miss fraud cases).  

---

### 6. Save Best Model
- Save pipeline (preprocessing + model) using **joblib**.  
- Metadata stored includes:  
  - Best model type  
  - Target column name  
  - Numeric + categorical feature lists  
  - Test split size  

**What I learned**: Saving preprocessing + model together ensures predictions work seamlessly in deployment.

---
