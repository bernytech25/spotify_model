# ============================================
# API SPOTIFY RECOMMENDER - CON POPULARIDAD REAL
# ============================================

from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import pickle
import faiss
from typing import List, Dict, Optional

app = FastAPI(
    title="Spotify Recommender API",
    description="Recomendación de canciones basada en similitud de características",
    version="1.0.0"
)

# ============================================
# CARGAR MODELOS
# ============================================

print("🔄 Cargando modelos...")

# Cargar embeddings
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Cargar índice FAISS
index = faiss.read_index('indice_faiss.bin')

# Cargar metadatos
with open('df_meta.pkl', 'rb') as f:
    df_meta = pickle.load(f)

print(f"✅ Modelos cargados: {len(df_meta)} canciones listas")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Endpoint raíz - verifica que la API funciona"""
    return {
        "message": "Spotify Recommender API",
        "status": "online",
        "total_canciones": len(df_meta)
    }

@app.get("/canciones/{idx}")
def get_cancion(idx: int):
    """Obtiene los datos de una canción por su índice"""
    if idx < 0 or idx >= len(df_meta):
        raise HTTPException(status_code=404, detail="Canción no encontrada")
    
    cancion = df_meta.iloc[idx]
    return {
        "idx": idx,
        "track_name": cancion['track_name'],
        "artists": cancion['artists'],
        "track_genre": cancion['track_genre'],
        "popularity": int(cancion['popularity'])
    }

@app.get("/populares")
def get_populares(limit: int = 20, min_popularity: int = 80):
    """Obtiene las canciones más populares"""
    populares = df_meta[df_meta['popularity'] >= min_popularity].head(limit)
    
    return [
        {
            "idx": idx,
            "track_name": row['track_name'],
            "artists": row['artists'],
            "popularity": int(row['popularity'])
        }
        for idx, row in populares.iterrows()
    ]

@app.post("/recomendar")
def recomendar(idx_cancion: int, top_n: int = 5):
    """Recomienda canciones similares a una canción dada"""
    if idx_cancion < 0 or idx_cancion >= len(embeddings):
        raise HTTPException(status_code=404, detail="Canción no encontrada")
    
    # Buscar similares con FAISS
    query = embeddings[idx_cancion:idx_cancion+1]
    similitudes, indices = index.search(query, top_n + 5)  # Pedimos extras para filtrar
    
    resultados = []
    nombre_original = df_meta.iloc[idx_cancion]['track_name']
    
    for i in range(len(indices[0])):
        idx = indices[0][i]
        
        # Evitar recomendar la misma canción
        if idx == idx_cancion:
            continue
            
        cancion = df_meta.iloc[idx]
        
        resultados.append({
            "idx": int(idx),
            "track_name": cancion['track_name'],
            "artists": cancion['artists'],
            "track_genre": cancion['track_genre'],
            "popularity": int(cancion['popularity']),
            "similitud": float(similitudes[0][i])
        })
        
        if len(resultados) >= top_n:
            break
    
    return {"canciones_similares": resultados}

@app.get("/buscar")
def buscar_por_nombre(q: str, limit: int = 10):
    """Busca canciones por nombre (búsqueda parcial)"""
    mascara = df_meta['track_name'].str.contains(q, case=False, na=False)
    resultados = df_meta[mascara].head(limit)
    
    return [
        {
            "idx": idx,
            "track_name": row['track_name'],
            "artists": row['artists']
        }
        for idx, row in resultados.iterrows()
    ]

@app.post("/recomendar_por_nombre")
def recomendar_por_nombre(track_name: str, top_n: int = 5):
    """Recomienda canciones similares por NOMBRE de canción"""
    # Buscar la canción
    mascara = df_meta['track_name'].str.lower() == track_name.lower()
    if not mascara.any():
        raise HTTPException(status_code=404, detail=f"Canción '{track_name}' no encontrada")
    
    idx_cancion = mascara.idxmax()
    return recomendar(idx_cancion, top_n)

@app.get("/cluster/{idx}")
def get_cluster_info(idx: int):
    """Obtiene información del cluster de una canción"""
    if idx < 0 or idx >= len(df_meta):
        raise HTTPException(status_code=404, detail="Canción no encontrada")
    
    cluster_id = int(df_meta.iloc[idx]['cluster'])
    mismo_cluster = df_meta[df_meta['cluster'] == cluster_id]
    
    return {
        "idx": idx,
        "track_name": df_meta.iloc[idx]['track_name'],
        "cluster": cluster_id,
        "total_en_cluster": len(mismo_cluster),
        "canciones_del_cluster": [
            {"idx": int(i), "track_name": row['track_name'], "artists": row['artists']}
            for i, row in mismo_cluster.head(10).iterrows()
        ]
    }

# ============================================
# EJECUTAR
# ============================================

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Iniciando API en http://127.0.0.1:8000")
    print("📖 Documentación interactiva en http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)