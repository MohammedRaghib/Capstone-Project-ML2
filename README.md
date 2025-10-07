# 📰 Fake News Detection using NLP (Flask Deployment)

📌 **Problem Statement**
The rapid spread of fake news across digital platforms threatens public trust and decision-making. Detecting misinformation automatically is challenging due to subtle linguistic cues and the strategic use of misleading or biased narratives.

This project builds a machine learning model that classifies news statements into six truthfulness categories using the LIAR dataset. The objective is to deploy a Flask API so users can submit text and receive predictions in real time.

🎯 **Objectives**
* Preprocess and clean the dataset, including handling missing data.
* Engineer highly predictive features based on speaker history.
* Develop a robust baseline, specialized ensemble models (Gradient Boosting), and deep learning models.
* Conduct comprehensive experimentation across ML and Deep Learning models.
* Evaluate final performance using the **Macro F1-Score**.
* Deploy a Flask web application for real-time predictions.

📂 **Dataset**
Name: LIAR Dataset
Source: William Yang Wang – UCSB
Files:
* `train.tsv` → training set
* `valid.tsv` → validation set
* `test.tsv` → test set
Labels: **pants-fire**, **false**, **barely-true**, **half-true**, **mostly-true**, **true**

***

## 🛠️ Methodology (Four-Phase Workflow)

The project followed a structured machine learning workflow to build and evaluate a multi-class classification model, with the Macro F1-Score as the primary metric.

### 1. Exploratory Data Analysis (EDA) and Feature Engineering
This phase focused on data preparation and creating predictive variables.

* **Data Cleaning and Preprocessing:** The text feature (**Statement**) underwent standardization (lowercasing, punctuation removal) and normalization (tokenization, stopword removal, and lemmatization). Missing values in categorical columns were handled.
* **Feature Engineering:** Features were engineered based on speaker history, including raw **Count Features** and normalized **Ratio Features** (e.g., proportion of true/mostly-true statements vs. false/pants-on-fire/barely-true statements). Additional text features like statement length were also created.
* **Analysis:** Univariate and bivariate analyses were conducted to understand the distribution of the target variable and the correlation of features with the truthfulness labels.

### 2. Model Making (Baseline Pipeline Construction)
A simple, robust benchmark model was established to provide an initial performance metric.

* **Feature Pipeline Construction:** A scikit-learn **ColumnTransformer** was defined to manage different feature types:
    * **Text Features:** Processed using a vectorization technique (e.g., TF-IDF Vectorizer).
    * **Categorical Features:** Handled with One-Hot Encoding.
    * **Numerical Features:** Scaled using a StandardScaler.
* **Baseline Selection:** **Logistic Regression** was chosen as the initial model, integrated within the feature pipeline.
* **Training and Evaluation:** The full pipeline was trained on the training data, and its performance (Macro F1-Score) on the validation set established the initial benchmark.

### 3. Experimentation (Advanced Modeling & Optimization)
This phase involved exploring a wider array of algorithms and refining model performance.

* **Model Exploration:** A range of models was tested against the baseline, including:
    * Machine Learning Models: **Linear Support Vector Machines (LinearSVC)** and **Multinomial Naive Bayes**.
    * Ensemble Methods: **Random Forest Classifier**.
    * Deep Learning Models: **LSTM** and **GRU** networks (trained on text embeddings) to capture sequence information.
* **Hyperparameter Tuning:** **Grid Search** or **Randomized Search** was employed on the most promising models to find optimal parameters for both the classifiers and the pipeline components (e.g., TF-IDF parameters, regularization strength).
* **Comparative Analysis:** All models were evaluated on the validation set, and a summary table of performance metrics was compiled to identify the best-performing candidates.

### 4. Gradient Boosting (Specialized Ensemble Model)
A dedicated investigation into high-performance ensemble techniques was conducted to potentially leverage the full, mixed feature set.

* **Model Selection and Configuration:** A **Gradient Boosting Classifier (GBC)** or similar advanced library (e.g., XGBoost) was trained on the features produced by the optimal ColumnTransformer.
* **Optimization:** Targeted hyperparameter tuning was performed for the GBC, focusing on parameters like `n_estimators`, `learning_rate`, and tree complexity controls (`max_depth`).
* **Final Model Selection and Testing:** The best-performing Gradient Boosting model was compared with the top models from Phase 3. The overall best model (based on the highest Macro F1-Score on the validation set) was selected and applied to the untouched **Test Set** to report the unbiased final metrics (Accuracy, Classification Report, and Confusion Matrix), confirming generalization capability.

### 5. Deployment (Flask)
The final, best-performing model pipeline is saved using `joblib`. A Flask API is created with two primary routes:
* `/predict` → Accepts a POST request with a news statement and returns the predicted truthfulness label.
* `/` → Serves a simple frontend form for users to submit text and view predictions.

***

📊 **Expected Results**
* **Logistic Regression (Baseline):** Achieved a Macro F1-Score in the range of 0.43-0.45 on the validation set, demonstrating a strong foundation.
* **Final Model Performance:** The best-performing model achieved similar F1-scores, indicating the challenge of the multi-class task.

🚀 **Future Enhancements**
* Hyperparameter optimization (e.g., using GridSearchCV) for the best-performing models (Logistic Regression and GBC).
* Integration of advanced contextual embeddings (e.g., BERT, not used in the final comparative run) for state-of-the-art performance.
* Extending the application with live data from external APIs (e.g., PolitiFact).

📦 **How to Run**
Clone repo & install dependencies
```bash
git clone [https://github.com/MohammedRaghib/Capstone-Project-ML2.git](https://github.com/MohammedRaghib/Capstone-Project-ML2.git)
cd Capstone-Project-ML2
pip install -r requirements.txt