import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
import base64

# TMDB API Key
DEFAULT_TMDB_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Pre-compute the SVG fallback ONCE at module level so it can be safely
# injected into inline onerror attributes without re-encoding on every card.
_SVG_PLACEHOLDER_SRC = (lambda: (
    lambda b64: f"data:image/svg+xml;base64,{b64}"
)(base64.b64encode(
    '<svg width="500" height="750" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="500" height="750" fill="#1e293b"/>'
    '<text x="50%" y="45%" font-family="sans-serif" font-size="80" fill="#64748b" text-anchor="middle">🎬</text>'
    '<text x="50%" y="55%" font-family="sans-serif" font-size="26" fill="#475569" text-anchor="middle">No Poster</text>'
    '</svg>'.encode('utf-8')
).decode('utf-8')))()

def get_base64_svg():
    return _SVG_PLACEHOLDER_SRC

@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id, imdb_id, api_key):
    if not pd.isna(imdb_id):
        formatted_imdb = f"tt{int(imdb_id):07d}"
        return f"https://images.metahub.space/poster/medium/{formatted_imdb}/img"
    return _SVG_PLACEHOLDER_SRC

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🍿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&family=Bebas+Neue&display=swap');

    /* ── Base ── */
    .stApp {
        background: #0a0a0f;
        color: #e8e6e1;
        font-family: 'DM Sans', sans-serif;
    }
    h3 { color: #a8a49e; font-weight: 400; text-align: center; margin-top: 5px; margin-bottom: 30px; }

    /* ── Hero: dark cinema with a single warm amber underline accent ── */
    .hero-section {
        background: #0f0e17;
        border: 1px solid #1e1d2e;
        border-top: 3px solid #e8a020;        /* single amber top-rule — like a marquee */
        padding: 48px 32px 40px;
        border-radius: 4px 4px 16px 16px;
        text-align: center;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    /* Film-strip perforation pattern along the sides */
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 28px;
        background-image: repeating-linear-gradient(
            to bottom,
            transparent 0px,
            transparent 10px,
            #1a1928 10px,
            #1a1928 22px
        );
        border-right: 1px solid #1e1d2e;
    }
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0;
        width: 28px;
        background-image: repeating-linear-gradient(
            to bottom,
            transparent 0px,
            transparent 10px,
            #1a1928 10px,
            #1a1928 22px
        );
        border-left: 1px solid #1e1d2e;
    }

    /* Glow behind the title */
    .hero-glow {
        position: absolute;
        top: -60px; left: 50%;
        transform: translateX(-50%);
        width: 500px; height: 200px;
        background: radial-gradient(ellipse, rgba(232,160,32,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        color: #f0ece4;
        font-size: 3.6em;
        font-weight: 400;          /* Bebas is inherently bold */
        letter-spacing: 3px;
        margin-bottom: 4px;
        position: relative;
        line-height: 1;
    }
    .hero-title span { color: #e8a020; }   /* amber accent on keyword */

    .hero-subtitle {
        color: #6b6860;
        font-size: 0.9em;
        font-weight: 300;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 36px;
        position: relative;
    }

    /* ── Stat Cards: dark glass, amber top border ── */
    .stats-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        position: relative;
    }
    .stat-pill {
        background: #16151f;
        border: 1px solid #252338;
        border-top: 2px solid #e8a020;
        border-radius: 0 0 10px 10px;
        padding: 14px 20px 12px;
        min-width: 118px;
        text-align: center;
        transition: background 0.2s ease, transform 0.2s ease;
    }
    .stat-pill:hover {
        background: #1c1b2a;
        transform: translateY(-3px);
    }
    .stat-number {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.9em;
        font-weight: 400;
        color: #e8a020;
        line-height: 1;
        display: block;
        letter-spacing: 1px;
    }
    .stat-label {
        font-size: 0.68em;
        color: #5a5760;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        display: block;
        margin-top: 5px;
    }

    /* ── Dataset Strip ── */
    .dataset-strip {
        background: #0f0e17;
        border: 1px solid #1e1d2e;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 28px;
        display: flex;
        align-items: flex-start;
        gap: 0;
        flex-wrap: wrap;
    }
    .ds-block {
        flex: 1;
        min-width: 190px;
        padding: 0 20px;
        border-right: 1px solid #1e1d2e;
    }
    .ds-block:first-child { padding-left: 4px; }
    .ds-block:last-child  { border-right: none; }
    .ds-head {
        font-size: 0.68em;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #e8a020;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .ds-body { font-size: 0.84em; color: #5a5760; line-height: 1.7; }
    .ds-body b { color: #a8a49e; font-weight: 500; }
    .ds-body code {
        background: #1c1b2a;
        color: #9d8fcc;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.92em;
    }

    /* ── Movie Cards ── */
    .movie-card {
        background: #0f0e17;
        border-radius: 12px;
        padding: 14px;
        border: 1px solid #1e1d2e;
        transition: all 0.25s ease;
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        transform: translateY(-8px);
        border-color: #e8a020;
        box-shadow: 0 12px 32px rgba(232,160,32,0.12);
    }
    .movie-img { border-radius: 8px; width: 100%; box-shadow: 0 4px 14px rgba(0,0,0,0.5); margin-bottom: 12px; }
    .movie-title { font-size: 1rem; font-weight: 600; color: #e8e6e1; margin-bottom: 4px; line-height: 1.3; }
    .movie-rating { font-size: 0.9rem; font-weight: 600; color: #e8a020; margin-bottom: 4px; }
    .movie-genres { font-size: 0.76rem; color: #4a4855; font-style: italic; margin-bottom: 10px; }

    /* ── Controls ── */
    .stButton>button {
        background: #e8a020;
        color: #0a0a0f;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #f5b733;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(232,160,32,0.35);
    }

    .imdb-btn {
        background-color: #f5c518;
        color: #000000 !important;
        padding: 7px 14px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 800;
        font-size: 0.82em;
        display: inline-block;
        transition: background 0.2s;
        margin-top: auto;
    }
    .imdb-btn:hover { background-color: #d4a914; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-glow"></div>
    <div class="hero-title">🎬 Hybrid <span>Movie</span> Recommendation Engine</div>
    <div class="hero-subtitle">
        Content-Based NLP &nbsp;·&nbsp; PySpark Collaborative Filtering &nbsp;·&nbsp; Hollywood &amp; Bollywood<br>
        Vansh Goel (24SCSE1720002) &nbsp;&amp;&nbsp; Shiva Tyagi &nbsp;·&nbsp; BCA Machine Learning Project
    </div>
    <div class="stats-row">
        <div class="stat-pill">
            <span class="stat-number">89K+</span>
            <span class="stat-label">Movies Indexed</span>
        </div>
        <div class="stat-pill">
            <span class="stat-number">32M</span>
            <span class="stat-label">Rating Events</span>
        </div>
        <div class="stat-pill">
            <span class="stat-number">5,000</span>
            <span class="stat-label">TF-IDF Features</span>
        </div>
        <div class="stat-pill">
            <span class="stat-number">~0.81</span>
            <span class="stat-label">ALS RMSE</span>
        </div>
        <div class="stat-pill">
            <span class="stat-number">1,600+</span>
            <span class="stat-label">Bollywood Titles</span>
        </div>
        <div class="stat-pill">
            <span class="stat-number">0.8 / 0.2</span>
            <span class="stat-label">Hybrid Weights</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Dataset details strip ────────────────────────────────────────────────────
st.markdown("""
<div class="dataset-strip">
    <div class="ds-block">
        <div class="ds-head">📂 Data Sources</div>
        <div class="ds-body">
            <b>MovieLens 32M</b> — 32M ratings backbone for collaborative filtering<br>
            <b>TMDB &amp; IMDb Non-Commercial</b> — genres, overviews, cast &amp; crew<br>
            <b>Bollywood Regional Dataset</b> — 1,600+ Hindi films injected via vertical concat
        </div>
    </div>
    <div class="ds-block">
        <div class="ds-head">🔗 Integration ("Golden Bridge")</div>
        <div class="ds-body">
            Datasets unified via <b>imdbId / tmdbId</b> mapping from <code>links.csv</code><br>
            Regional titles without global IDs got <b>synthetic IDs</b> generated<br>
            Deduplication handled dynamically at index-build time
        </div>
    </div>
    <div class="ds-block">
        <div class="ds-head">🧹 Data Treatment</div>
        <div class="ds-body">
            Fields merged into a <b>single text tag</b> per movie (overview + genres + cast + director)<br>
            Cleaned with <b>regex</b> (strip noise) → <b>NLTK PorterStemmer</b> (root forms)<br>
            Missing ratings imputed to <b>5.0</b>; missing posters fall back to SVG placeholder
        </div>
    </div>
    <div class="ds-block">
        <div class="ds-head">⚙️ ML Pipeline</div>
        <div class="ds-body">
            <b>PySpark ALS</b> on 32M rows → RMSE ≈ 0.81<br>
            <b>TF-IDF</b> (5,000 features) → on-the-fly cosine similarity at inference<br>
            Final score = <b>0.8 × cosine sim + 0.2 × norm. IMDb rating</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#e8a020;'>🔑 API Settings</h2>", unsafe_allow_html=True)
    user_api_key = st.text_input("TMDB API Key (For Posters)", type="password",
                                 help="The default public key is often rate-limited. Get your own free key from themoviedb.org!")
    active_api_key = user_api_key if user_api_key else DEFAULT_TMDB_KEY

    st.markdown("<h2 style='text-align:center;color:#e8a020;'>👨‍💻 Developers</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f0e17;border:1px solid #1e1d2e;padding:15px;border-radius:10px;text-align:center;color:#a8a49e;'>
        <b style='font-size:1.1em;color:#e8e6e1;'>Vansh Goel</b><br>
        <b style='font-size:1.1em;color:#e8e6e1;'>Shiva Tyagi</b><br>
        <hr style='border-color:#1e1d2e;margin:10px 0;'>
        <i style='font-size:0.85em;'>BCA Machine Learning Course Project</i>
    </div>
    """, unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Engine Core...")
def load_data():
    movies = pd.read_pickle('movies_data.pkl')
    tfidf = scipy.sparse.load_npz('tfidf_matrix.npz')

    def clean_display_title(title):
        title = str(title).strip()
        title = re.sub(r'"+', '"', title)
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        return title

    movies['display_title'] = movies['title'].apply(clean_display_title)
    movies['search_title'] = movies['display_title'].apply(lambda t: str(t).lower().strip())
    indices = pd.Series(movies.index, index=movies['search_title']).drop_duplicates()
    return movies, tfidf, indices

movies, tfidf_matrix, indices = load_data()
movie_list = sorted(movies['display_title'].dropna().astype(str).unique())

# ── Search UI ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_movie = st.selectbox("Search for a Masterpiece:", movie_list)
    search_clicked = st.button('Discover Similar Movies', use_container_width=True)

# ── Recommendation logic ──────────────────────────────────────────────────────
if search_clicked:
    with st.spinner('Analyzing neural pathways & metadata...'):
        movie_title_clean = str(selected_movie).lower().strip()

        if movie_title_clean in indices:
            idx = indices[movie_title_clean]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

            movie_vector = tfidf_matrix[idx]
            sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()

            scores_df = pd.DataFrame({
                'index': range(len(sim_scores)),
                'similarity': sim_scores,
                'rating': movies['averageRating'].fillna(5.0)
            })
            scores_df = scores_df[scores_df['index'] != idx]
            scores_df['norm_rating'] = scores_df['rating'] / 10.0
            scores_df['hybrid_score'] = (scores_df['similarity'] * 0.8) + (scores_df['norm_rating'] * 0.2)

            top_movies = scores_df.sort_values(
                by=['hybrid_score', 'similarity'], ascending=[False, False]
            ).head(5)

            st.markdown("---")
            st.markdown(
                f"<h3 style='text-align:left;color:#a8a49e;font-family:\"DM Sans\",sans-serif;'>Top Matches for: "
                f"<span style='color:#e8a020;font-weight:700;'>{selected_movie}</span></h3>",
                unsafe_allow_html=True
            )

            cols = st.columns(5)
            for i, col in enumerate(cols):
                movie_idx = int(top_movies.iloc[i]['index'])

                rec_title  = movies.iloc[movie_idx]['display_title']
                rec_tmdbId = movies.iloc[movie_idx]['tmdbId']
                rec_imdbId = movies.iloc[movie_idx]['imdbId']
                rec_rating = movies.iloc[movie_idx]['averageRating']
                rec_rating_str = "Unrated" if pd.isna(rec_rating) else f"{rec_rating}/10"

                raw_genres = str(movies.iloc[movie_idx]['final_genres']).split()[:3]
                rec_genres = " • ".join([g.capitalize() for g in raw_genres]) if raw_genres else "Unknown"

                poster_url = fetch_poster(rec_tmdbId, rec_imdbId, active_api_key)

                imdb_link   = f"https://www.imdb.com/title/tt{int(rec_imdbId):07d}/" if not pd.isna(rec_imdbId) else "#"
                imdb_button = f'<a href="{imdb_link}" target="_blank" class="imdb-btn">🔗 View IMDb</a>' if not pd.isna(rec_imdbId) else ""

                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster_url}" class="movie-img"
                             onerror="this.onerror=null; this.src='{get_base64_svg()}';">
                        <div class="movie-title">{rec_title}</div>
                        <div class="movie-rating">⭐ {rec_rating_str}</div>
                        <div class="movie-genres">{rec_genres}</div>
                        {imdb_button}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Movie not found in index. Please try another title.")