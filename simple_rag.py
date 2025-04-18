import os
import time
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Any
from logs import append_to_json_file

# Global variables for model and embeddings
MODEL = None
EMBEDDINGS = None
CHUNKS = None

def initialize_model_and_embeddings(directory_path: str):
    """Initialize the SentenceTransformer model and precompute document embeddings."""
    global MODEL, EMBEDDINGS, CHUNKS

    print(f"\n{'='*50}")
    print(f"INITIALIZATION PROCESS")
    print(f"{'='*50}")

    # Load documents
    documents = load_documents_from_directory(directory_path)
    if not documents:
        print("No documents found. Exiting initialization.")
        return False

    # Chunk documents
    CHUNKS = chunk_documents(documents)

    # Initialize embedding model
    print(f"\n{'='*50}")
    print(f"MODEL INITIALIZATION")
    print(f"{'='*50}")
    model_start = time.time()
    print(f"Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    model_time = time.time() - model_start
    print(f"✓ Model loaded in {model_time:.3f}s")

    # Generate embeddings
    print(f"\n{'='*50}")
    print(f"EMBEDDING GENERATION")
    print(f"{'='*50}")
    embedding_start = time.time()
    print(f"Generating embeddings for {len(CHUNKS)} chunks...")

    batch_size = 32
    all_embeddings = []

    for i in range(0, len(CHUNKS), batch_size):
        batch = CHUNKS[i:i+batch_size]
        batch_texts = [chunk['text'] for chunk in batch]
        batch_embeddings = MODEL.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)

        # Print progress
        progress = min(i+batch_size, len(CHUNKS))
        print(f"  Progress: {progress}/{len(CHUNKS)} ({progress/len(CHUNKS)*100:.1f}%)")

    embedding_time = time.time() - embedding_start
    embed_per_sec = len(CHUNKS) / embedding_time if embedding_time > 0 else 0
    print(f"\n✓ Generated {len(all_embeddings)} embeddings in {embedding_time:.3f}s")
    print(f"  Embedding speed: {embed_per_sec:.1f} embeddings/sec")

    EMBEDDINGS = np.array(all_embeddings).astype('float32')
    print(f"  Embedding dimensions: {EMBEDDINGS[0].shape}")
    print(f"  Total embedding size: {EMBEDDINGS.nbytes / (1024 * 1024):.2f} MB")

    return True

def load_documents_from_directory(directory_path: str) -> List[Dict]:
    """Load all documents from a directory with detailed logging"""
    print(f"\n{'='*50}")
    print(f"DOCUMENT LOADING PROCESS - TRADITIONAL RAG")
    print(f"{'='*50}")
    
    load_start = time.time()
    documents = []
    total_chars = 0
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return []
        
    print(f"Scanning directory: {directory_path}")
    
    file_paths = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in ['.txt', '.md', '.docx', '.pdf']:
            file_paths.append(file_path)
    
    print(f"Found {len(file_paths)} compatible documents")
            
    for file_path in file_paths:
        file_start = time.time()
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"Loading: {filename} ({file_size:.2f} KB)")
        
        try:
            if file_ext in ['.txt', '.md']:
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            text = file.read()
                            char_count = len(text)
                            total_chars += char_count
                            documents.append({
                                'id': filename,
                                'text': text
                            })
                        print(f"  ✓ Loaded {char_count:,} chars using {encoding} in {time.time()-file_start:.3f}s")
                        break
                    except UnicodeDecodeError:
                        continue
            
            elif file_ext == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    char_count = len(text)
                    total_chars += char_count
                    documents.append({
                        'id': filename,
                        'text': text
                    })
                    print(f"  ✓ Loaded {char_count:,} chars using python-docx in {time.time()-file_start:.3f}s")
                except ImportError:
                    print("  ✗ python-docx not installed. Run: pip install python-docx")
                    
            elif file_ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n\n"
                        char_count = len(text)
                        total_chars += char_count
                        documents.append({
                            'id': filename,
                            'text': text
                        })
                    print(f"  ✓ Loaded {char_count:,} chars using PyPDF2 in {time.time()-file_start:.3f}s")
                except ImportError:
                    print("  ✗ PyPDF2 not installed. Run: pip install PyPDF2")
                    
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
    
    load_time = time.time() - load_start
    print(f"\n✓ Loaded {len(documents)} documents ({total_chars:,} total chars) in {load_time:.3f}s")
    print(f"  Average loading speed: {total_chars/load_time:.0f} chars/sec")
    
    return documents


def chunk_documents(documents: List[Dict], chunk_size: int = 1000) -> List[Dict]:
    """Split documents into chunks with detailed logging"""
    print(f"\n{'='*50}")
    print(f"DOCUMENT CHUNKING PROCESS")
    print(f"{'='*50}")
    chunk_start = time.time()
    
    chunks = []
    chunk_sizes = []
    total_input_chars = sum(len(doc['text']) for doc in documents)
    
    for doc_idx, doc in enumerate(documents):
        doc_start = time.time()
        print(f"Processing document {doc_idx+1}/{len(documents)}: {doc['id']} ({len(doc['text']):,} chars)")
        
        paragraphs = re.split(r'\n\n+', doc['text'])
        doc_chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                doc_chunks.append({
                    'id': f"{doc['id']}_chunk_{i}",
                    'text': para.strip(),
                    'document_id': doc['id']
                })
                chunk_sizes.append(len(para.strip()))
        
        chunks.extend(doc_chunks)
        print(f"  ✓ Created {len(doc_chunks)} chunks in {time.time()-doc_start:.3f}s")
    
    # Chunk statistics
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    
    chunk_time = time.time() - chunk_start
    print(f"\n✓ Created {len(chunks)} total chunks in {chunk_time:.3f}s")
    print(f"  Chunk sizes: min={min_chunk_size}, avg={avg_chunk_size:.1f}, max={max_chunk_size} chars")
    print(f"  Input: {total_input_chars:,} chars → Output: {sum(chunk_sizes):,} chars")
    print(f"  Chunking speed: {total_input_chars/chunk_time:.0f} chars/sec")
    
    return chunks

def traditional_faiss_search(query: str, top_k: int = 3):
    """Perform FAISS search using precomputed embeddings."""
    global MODEL, EMBEDDINGS, CHUNKS

    if MODEL is None or EMBEDDINGS is None or CHUNKS is None:
        print("Error: Model and embeddings are not initialized. Please initialize first.")
        return [], {"error": "Model and embeddings not initialized"}

    print(f"\n{'='*50}")
    print(f"SEARCH EXECUTION")
    print(f"{'='*50}")
    search_start = time.time()

    # Normalize embeddings for cosine similarity
    print("Normalizing embeddings for cosine similarity...")
    norm_start = time.time()
    faiss.normalize_L2(EMBEDDINGS)
    norm_time = time.time() - norm_start
    print(f"✓ Normalization complete in {norm_time:.3f}s")

    # Build FAISS index
    print(f"Creating FAISS IndexFlatIP with {EMBEDDINGS.shape[1]} dimensions...")
    index = faiss.IndexFlatIP(EMBEDDINGS.shape[1])

    add_start = time.time()
    print(f"Adding {len(EMBEDDINGS)} vectors to index...")
    index.add(EMBEDDINGS)
    add_time = time.time() - add_start
    print(f"✓ Added vectors in {add_time:.3f}s")

    # Embed query
    print(f"Embedding query: '{query}'")
    query_embedding = MODEL.encode([query])[0].astype('float32').reshape(1, -1)
    print(f"Query embedding shape: {query_embedding.shape}")

    print("Normalizing query vector...")
    faiss.normalize_L2(query_embedding)

    print(f"Searching against ALL {len(CHUNKS)} vectors in corpus...")
    actual_search_start = time.time()
    D, I = index.search(query_embedding, min(top_k, len(CHUNKS)))
    actual_search_time = time.time() - actual_search_start
    print(f"✓ Vector search completed in {actual_search_time*1000:.2f} ms")

    # Get results
    print("\nAssembling results...")
    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(CHUNKS):
            results.append({
                'document_id': CHUNKS[idx]['document_id'],
                'chunk_id': CHUNKS[idx]['id'],
                'text': CHUNKS[idx]['text'],
                'similarity': float(D[0][i])
            })
            print(f"  Result {i+1}: {CHUNKS[idx]['document_id']} (Score: {D[0][i]:.4f})")

    search_time = time.time() - search_start
    print(f"\n✓ Search process completed in {search_time:.3f}s")
    print(f"  Found {len(results)} results")

    return results, {
        'search_time': search_time,
        'actual_search_time': actual_search_time,
        'vector_comparisons': len(CHUNKS)
    }


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"TRADITIONAL RAG SYSTEM - DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*70}")

    data = {}
    file_path = "rag_data.json"

    docs_dir = input("Enter documents directory path: ")

    # Initialize model and embeddings
    if initialize_model_and_embeddings(docs_dir):
        while True:
            query = input("\nEnter search query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            data["Query"] = query
            start_time = time.time()
            results, metrics = traditional_faiss_search(query)
            overall_time = time.time() - start_time

            print(f"\n{'='*50}")
            print(f"SEARCH RESULTS")
            print(f"{'='*50}")
            for i, result in enumerate(results):
                j = i+1
                print(f"{j}. [{result['document_id']}] (Score: {result['similarity']:.4f})")
                print(f"   {result['text']}\n")
                data[f"{j} Document"] = f"[{result['document_id']}] (Score: {result['similarity']:.4f})"
                data[f"{j} Document Text"] = result['text']

            print("Total Time : ", overall_time)
            data["OverAll Time : "] = overall_time
            append_to_json_file(data, file_path)