import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

def build_indices_cosine(embeddings_dict):
    """Build FAISS indices using cosine similarity with detailed logging"""
    print(f"\nBuilding search indices for all levels...")
    
    indices = {}
    start_time = time.time()
    
    # Track total vectors
    total_vectors = 0
    

    for level, embeds in embeddings_dict.items():
        level_start = time.time()
        print(f"Building {level} level index...")
        
        # Convert to numpy array
        embed_array = np.array(embeds).astype('float32')
        print(f"  Array shape: {embed_array.shape}, dtype: {embed_array.dtype}")
        print(f"  Memory: {embed_array.nbytes / (1024*1024):.2f} MB")
        
        # Normalize for cosine similarity
        norm_start = time.time()
        print(f"  Normalizing vectors...")
        faiss.normalize_L2(embed_array)
        norm_time = time.time() - norm_start
        print(f"  Normalization completed in {norm_time*1000:.2f} ms")

        # Build index
        dimension = embed_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        add_start = time.time()
        print(f"  Adding {len(embed_array)} vectors to index...")
        index.add(embed_array)
        add_time = time.time() - add_start
        print(f"  Vectors added in {add_time*1000:.2f} ms")
        
        indices[level] = index
        total_vectors += index.ntotal
        
        level_time = time.time() - level_start
        print(f"Built cosine similarity index for {level} level with {index.ntotal} vectors in {level_time*1000:.2f} ms")

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"\nIndex building summary:")
    print(f"  Built {len(indices)} indices with total {total_vectors} vectors")
    print(f"  Time: {elapsed_ms:.2f} ms")
    print(f"  Speed: {total_vectors/(elapsed_ms/1000):.1f} vectors/sec")

    return indices


def create_multilevel_embeddings(text_levels, model):
    """Create embeddings at different granularity levels with detailed logging"""
    print(f"\nCreating embeddings for all hierarchical levels...")
    
    embeddings = {}
    total_embedding_count = 0
    start_time = time.time()

    # Process each level
    for level_name, texts in text_levels.items():
        if level_name == 'metadata':
            continue
            
        level_start = time.time()
        print(f"Creating embeddings for {level_name} level...")
        
        # Create embeddings with batch processing for better visibility
        batch_size = 32
        level_embeddings = []
        texts_length = len(texts)
        
        for i in range(0, texts_length, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch_texts)
            level_embeddings.extend(batch_embeddings)
            
            # Print progress for paragraph level (which can be large)
            if level_name == 'paragraphs' and texts_length > 100 and i % 100 == 0:
                print(f"  Progress: {min(i+batch_size, texts_length)}/{texts_length} ({min(i+batch_size, texts_length)/texts_length*100:.1f}%)")
            
        embeddings[level_name] = level_embeddings
        total_embedding_count += len(level_embeddings)
        
        level_time = time.time() - level_start
        embed_per_sec = len(texts) / level_time if level_time > 0 else 0
        
        print(f"Created {len(level_embeddings)} embeddings at {level_name} level in {level_time:.2f}s ({embed_per_sec:.1f} embeddings/sec)")
        
        # Show embedding dimensions
        if level_embeddings:
            print(f"  Embedding dimensions: {level_embeddings[0].shape}")
            print(f"  Memory per embedding: {level_embeddings[0].nbytes / 1024:.2f} KB")
            
        # Calculate memory for this level
        level_memory_mb = sum(e.nbytes for e in level_embeddings) / (1024 * 1024)
        print(f"  Memory used for {level_name} level: {level_memory_mb:.2f} MB")


    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    elapsed_sec = elapsed_ms / 1000
    
    print(f"\nTotal embedding creation summary:")
    print(f"  Created {total_embedding_count} total embeddings")
    print(f"  Time: {elapsed_sec:.2f} seconds ({elapsed_ms:.2f} ms)")
    print(f"  Speed: {total_embedding_count/elapsed_sec:.1f} embeddings/sec")
    
    # Calculate total embedding memory
    total_memory_mb = sum(sum(e.nbytes for e in emb_list) for emb_list in embeddings.values()) / (1024 * 1024)
    print(f"  Total embedding memory: {total_memory_mb:.2f} MB")

    return embeddings