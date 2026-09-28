# ============================================================
#  Movie Recommendation System — Streamlit Web App
#  Author : Vraj Patoliya | Gandhinagar University | B.Tech IT
# ============================================================

import streamlit as st
import pickle
import pandas as pd

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender — Vraj Patoliya",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.4rem 0;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .score-badge {
        background: rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        float: right;
    }
    .author-tag {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    .stSelectbox label { font-weight: 600; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    movies     = pickle.load(open('movies.pkl',     'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

try:
    movies, similarity = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Recommend Function ───────────────────────────────────────
def recommend(movie_title, n=5):
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    distances = similarity[idx]
    sorted_movies = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:n+1]
    return [(movies.iloc[i].title, round(score * 100, 1)) for i, score in sorted_movies]


# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Content-based filtering using Cosine Similarity | Vraj Patoliya — Gandhinagar University</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Please run the notebook first to generate movies.pkl and similarity.pkl")
    st.code("# Run in Jupyter notebook:\npickle.dump(new_df, open('movies.pkl', 'wb'))\npickle.dump(similarity, open('similarity.pkl', 'wb'))")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
    1. Movie tags created from:
       - Plot overview
       - Genres
       - Keywords
       - Top 3 cast
       - Director
    2. Tags vectorized with **CountVectorizer**
    3. Similarity computed via **Cosine Similarity**
    4. Top-N most similar movies returned
    """)
    st.divider()
    st.markdown(f"📽️ **{len(movies)} movies** in database")


# ── Main Area ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Find Similar Movies")
    all_titles = sorted(movies['title'].values.tolist())
    selected_movie = st.selectbox(
        "Select a movie you like:",
        all_titles,
        index=all_titles.index('Avatar') if 'Avatar' in all_titles else 0
    )

    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            results = recommend(selected_movie, n=num_recs)

        if results:
            st.success(f"✅ Top {len(results)} movies similar to **{selected_movie}**")
            st.divider()
            for rank, (title, score) in enumerate(results, 1):
                st.markdown(f"""
                <div class="movie-card">
                    {rank}. {title}
                    <span class="score-badge">Match: {score}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Movie not found in the database.")

with col2:
    st.subheader("📊 Quick Stats")
    st.metric("Total Movies", len(movies))
    st.metric("Algorithm", "Cosine Similarity")
    st.metric("Features", "5000 words")
    st.metric("Vectorizer", "CountVectorizer")
    st.divider()
    st.subheader("🔥 Try These")
    quick_movies = ["Avatar", "Inception", "The Dark Knight", "Interstellar", "The Godfather"]
    for m in quick_movies:
        if m in movies['title'].values:
            if st.button(f"▶ {m}", use_container_width=True):
                results = recommend(m, n=num_recs)
                if results:
                    st.success(f"Similar to {m}:")
                    for rank, (title, score) in enumerate(results, 1):
                        st.write(f"{rank}. {title} ({score}%)")


# ── About Section ────────────────────────────────────────────
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    ### Movie Recommendation System
    
    **Type:** Content-Based Filtering  
    **Dataset:** TMDB 5000 Movie Dataset (Kaggle)  
    **Algorithm:** Cosine Similarity  
    **Libraries:** Python, Pandas, Scikit-learn, Streamlit  
    
    #### How Content-Based Filtering Works:
    Each movie is represented as a **tag vector** combining:
    - Plot overview words
    - Genre names (Action, Drama, etc.)
    - Thematic keywords
    - Top 3 cast members
    - Director name
    
    **CountVectorizer** converts these tags into numerical vectors.  
    **Cosine Similarity** measures the angle between vectors — movies with smaller angles are more similar.
    
    #### Author
    **Vraj Patoliya** | B.Tech Information Technology  
    Gandhinagar University, Gujarat, India
    """)

st.markdown('<div class="author-tag">Built by Vraj Patoliya — Gandhinagar University | B.Tech IT | 2024</div>', unsafe_allow_html=True)
