import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_indices_cosine(embeddings_dict):
    """Build FAISS indices using cosine similarity"""
    indices = {}

    for level, embeds in embeddings_dict.items():
        embed_array = np.array(embeds).astype('float32')
        faiss.normalize_L2(embed_array)

        dimension = embed_array.shape[1]
        index = faiss.IndexFlatIP(dimension)

        index.add(embed_array)
        indices[level] = index

        print(f"Built cosine similarity index for {level} level with {index.ntotal} vectors")

    return indices

def create_multilevel_embeddings(text_levels, model):
    """Create embeddings at different granularity levels"""

    embeddings = {}

    for level_name, texts in text_levels.items():
        if level_name == 'metadata':
            continue

        print(f"Creating embeddings for {level_name} level...")
        level_embeddings = model.encode(texts)
        embeddings[level_name] = level_embeddings
        print(f"Created {len(level_embeddings)} embeddings at {level_name} level")

    return embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')