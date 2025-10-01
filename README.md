# 📰 Fake News Detection using NLP (Flask Deployment)

## 📌 Problem Statement

The rapid spread of fake news across digital platforms threatens public trust and decision-making. Detecting misinformation automatically is challenging due to subtle linguistic cues and biased narratives.

This project builds a **machine learning model that classifies news statements into six truthfulness categories** using the **LIAR dataset**. A Flask API is deployed so users can submit text and receive predictions in real time.

---

## 🎯 Objectives

1. Preprocess and clean the dataset.
2. Engineer text features using TF-IDF and embeddings.
3. Train baseline and advanced ML/NLP models.
4. Evaluate performance with proper metrics.
5. Deploy a **Flask web application** for real-time predictions.

---

## 📂 Dataset

* **Name**: LIAR Dataset
* **Source**: [William Yang Wang – UCSB](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
* **Files**:

  * `train.tsv` → training set (~10k samples)
  * `valid.tsv` → validation set (~1.2k samples)
  * `test.tsv` → test set (~1.2k samples)
* **Labels**: `pants-fire, false, barely-true, half-true, mostly-true, true`

---

## 🛠️ Methodology

### 1. Data Preprocessing

* Lowercasing, tokenization, stopword removal, lemmatization.
* Handling missing values & imbalanced labels.

### 2. Feature Engineering

* **Baseline**: TF-IDF (n-grams).
* **Advanced**: Word2Vec, GloVe embeddings.
* **Deep Learning**: Fine-tuned BERT embeddings.

### 3. Model Training

* Logistic Regression, Naive Bayes, XGBoost (baselines).
* LSTM / BiLSTM (sequence models).
* BERT fine-tuning (advanced).

### 4. Model Evaluation

* Use **validation set** for hyperparameter tuning.
* Final evaluation on **test set**.
* Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

### 5. Deployment (Flask)

* Train model, save with `pickle` or `joblib`.
* Flask API with routes:

  * `/predict` → POST a news statement, returns label.
  * `/` → simple frontend form for users.
* Host on **Heroku, Render, or PythonAnywhere**.

---

## 📊 Expected Results

* TF-IDF + Logistic Regression → ~25–30% accuracy (multi-class baseline).
* BERT fine-tuning → ~40–45% accuracy (state-of-the-art on LIAR).

---

## 🚀 Future Enhancements

* Add speaker metadata (party, context, state) into the model.
* Extend with live data from PolitiFact API/scraping.
* Multi-lingual fake news detection.

---

## 📦 How to Run

1. **Clone repo & install dependencies**

   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```bash
   python train.py
   ```

3. **Run Flask app**

   ```bash
   python app.py
   ```

4. **Access locally**

   * Open browser → `http://127.0.0.1:5000/`
   * Enter a statement → Get prediction.

---

## 👨‍💻 Author

Mohammed Raghib – Capstone Project 2025

---
