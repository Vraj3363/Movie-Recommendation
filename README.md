# 🎬 Movie Recommendation System

**Author:** Vraj Patoliya | B.Tech Information Technology  
**College:** Gandhinagar University, Gujarat  
**CGPA:** 9.59 | Internship: NSDBytes Technologies

---

## 📌 Project Overview

A content-based movie recommendation system that suggests similar movies based on genres, plot keywords, cast, and director using **CountVectorizer** and **Cosine Similarity**.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| ML Algorithm | Scikit-learn (CountVectorizer, Cosine Similarity) |
| Text Processing | NLTK (PorterStemmer) |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |

## 📁 Project Structure

```
movie-recommender/
├── movie_recommendation.ipynb   # Main Jupyter Notebook
├── app.py                       # Streamlit web app
├── requirements.txt             # Dependencies
├── movies.pkl                   # Saved movie dataframe
├── similarity.pkl               # Cosine similarity matrix
├── vectorizer.pkl               # Fitted CountVectorizer
└── README.md                    # This file
```

## 🚀 How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download dataset
Download from Kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata  
Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project folder.

### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook movie_recommendation.ipynb
```
Run all cells — this builds and saves the model (movies.pkl, similarity.pkl).

### Step 4: Launch the Web App
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## 🔍 How It Works

```
Movie Input
    ↓
Extract Features (Overview + Genres + Keywords + Cast + Director)
    ↓
Text Preprocessing (Stemming, Lowercase)
    ↓
CountVectorizer (5000 features)
    ↓
Cosine Similarity Matrix (4800 x 4800)
    ↓
Top-N Most Similar Movies
```

## 📊 Results

- Input: `Avatar`
- Recommendations: Guardians of the Galaxy, Aliens, Star Wars, Titan A.E., Independence Day

- Input: `The Dark Knight`
- Recommendations: Batman Begins, The Dark Knight Rises, Joker, Batman, Superman

## 📸 Screenshots

> Run the notebook to generate:
> - `genre_distribution.png` — top genres chart
> - `similarity_heatmap.png` — movie similarity heatmap
> - `reco_Avatar.png` — recommendation bar chart

## 🧠 Algorithm Details

**CountVectorizer:**  
Converts each movie's tags into a bag-of-words vector (5000 most frequent words, English stop words removed).

**Cosine Similarity:**  
Measures the cosine of the angle between two vectors.  
Score = 1.0 → identical movies  
Score = 0.0 → completely different movies  
Typical good match: 0.15 – 0.40

## 📦 Requirements

```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
streamlit
```

---

*This project was built as part of Data Science placement preparation at Gandhinagar University.*
