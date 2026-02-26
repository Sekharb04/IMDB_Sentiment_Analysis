# 🎬 IMDB Movie Review Sentiment Analysis

A Machine Learning web application that classifies movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP) techniques and Logistic Regression.

---

## 📌 Project Overview

This project builds a complete end-to-end sentiment analysis system:

- Text preprocessing (cleaning, stopword removal, stemming)
- TF-IDF vectorization
- Model training & comparison
- Hyperparameter tuning
- Flask web deployment
- Confidence score prediction

The application provides real-time sentiment prediction through a clean web interface.

---

## 🧠 Machine Learning Pipeline

1. **Data Source**  
   IMDB Movie Reviews Dataset (50,000 reviews)

2. **Preprocessing**
   - Lowercasing
   - HTML tag removal
   - Special character removal
   - Stopword removal
   - Stemming (PorterStemmer)

3. **Feature Engineering**
   - TF-IDF Vectorization (max_features=5000)

4. **Models Evaluated**
   - Logistic Regression
   - Multinomial Naive Bayes
   - Random Forest

5. **Model Selection**
   Logistic Regression selected based on highest F1-Score and stable cross-validation performance.

---

## 🖥️ Web Application Features

- Modern responsive UI
- Real-time prediction
- Sentiment confidence score
- Clean project structure
- Lightweight deployment setup

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- NLTK
- Flask
- HTML / CSS

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python app.py
