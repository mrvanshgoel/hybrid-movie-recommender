# 🎬 Hybrid Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![PySpark](https://img.shields.io/badge/PySpark-ALS%20Collaborative%20Filtering-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A state-of-the-art Hybrid Movie Recommendation Engine built for a BCA Machine Learning final project. This engine seamlessly blends **Content-Based NLP** and **PySpark Collaborative Filtering** to deliver highly accurate movie suggestions across both Hollywood and Bollywood datasets.

---

## 🌟 Key Features

* **Massive Index:** Over 89,000 global movies indexed, including a dedicated injection of 1,600+ regional Bollywood titles.
* **Hybrid Algorithm:** Combines TF-IDF vectorization (Content-Based) and Alternating Least Squares (ALS - Collaborative Filtering).
* **Live Metadata Fetching:** Retrieves high-quality movie posters in real-time using the TMDB API.
* **Premium UI/UX:** A stunning, dark-cinema themed interface built entirely in Streamlit with custom CSS, micro-animations, and responsive glassmorphic cards.
* **Instant Inference:** Optimized sparse matrices and pre-computed features allow for lightning-fast cosine similarity lookups.

---

## 🧠 Machine Learning Architecture

### 1. Data Sources (The "Golden Bridge")
* **MovieLens 32M:** Provided the backbone of 32 million user rating events for Collaborative Filtering.
* **TMDB / IMDb:** Provided rich text metadata (genres, overviews, cast, and directors).
* **Bollywood Regional Dataset:** Appended vertically and mapped to synthetic IDs for titles lacking global identifiers.

### 2. Content-Based Filtering (NLP)
* **Text Processing:** Combined overviews, genres, cast, and crew into a single "tag" per movie. Applied regex cleaning and NLTK PorterStemmer.
* **Vectorization:** Extracted 5,000 top features using `TfidfVectorizer`.
* **Similarity:** Computes Cosine Similarity on the fly when a user queries a movie.

### 3. Collaborative Filtering (ALS)
* **PySpark ALS:** Trained an Alternating Least Squares model on the 32M ratings dataset.
* **Performance:** Achieved an excellent RMSE (Root Mean Square Error) of ~0.81.

### 4. The Hybrid Engine
The final recommendation score for a given movie query is calculated using a weighted formula:
`Hybrid Score = (Cosine Similarity × 0.8) + (Normalized Average Rating × 0.2)`

---

## 🚀 Running the Project Locally

### Prerequisites
Make sure you have Python 3.9+ installed. You will also need Git and Git LFS if you plan on modifying the dataset.

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/mrvanshgoel/hybrid-movie-recommender.git
   cd hybrid-movie-recommender
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open in your browser at `http://localhost:8501`.*

---

## 👨‍💻 Developers
**Vansh Goel** (24SCSE1720002) & **Shiva Tyagi**
*BCA Machine Learning Course Project*
