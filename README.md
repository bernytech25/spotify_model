# 🎵 Spotify Recommender System

Music recommendation system based on **acoustic similarity** using FAISS (vector search), with song clustering and popularity prediction. Complete project from data analysis to production-ready API.

## 📊 Dataset

- **114,000 songs** from Spotify
- **20+ acoustic features** (danceability, energy, valence, tempo, etc.)
- **Diverse music genres** (acoustic, rock, pop, reggaeton, etc.)

## 🚀 Tech Stack

| Area | Technologies |
|------|--------------|
| **ML & Data** | Python, Pandas, NumPy, Scikit-learn |
| **Recommendation** | FAISS (vector search), Cosine Similarity |
| **Clustering** | K-Means, t-SNE |
| **API** | FastAPI, Uvicorn |
| **Container** | Docker |
| **Versioning** | Git, GitHub |

## 🎯 Model Approach

**Acoustic similarity > Genre labels**

The model recommends songs based on **how they sound** (energy, danceability, tempo), not on arbitrary genre labels. This allows discovering musical connections that traditional genre tags hide.

### Real example:

| Original Song | Genre | Recommendation | Genre | Why? |
|---------------|-------|----------------|-------|------|
| Comedy | Acoustic | JAMAICA | Reggaeton | Both have medium energy and similar tempo |

## 📈 Model Evaluation

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Latency** | 1.22 ms | ⚡ Excellent for production |
| **Average Similarity** | 0.99 | ✅ Highly similar recommendations |
| **Coverage** | 80% | 📚 8 out of 10 songs are recommendable |
| **Genre Precision** | 17% | 🎯 Intentional: prioritizes sound over labels |

## 🔧 Installation & Usage

### Local

```bash
# Clone repository
git clone https://github.com/bernytech25/spotify_model.git
cd spotify_model

# Install dependencies
pip install -r requirements.txt

# Run API
python app.py
