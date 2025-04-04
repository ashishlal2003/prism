import faiss
import numpy as np

def progressive_search_cosine(query, indices, embeddings_dict, text_levels, model, top_k=3):
    """
    Search through embeddings at multiple levels with progressive filtering using cosine similarity
    """

    query_embedding = model.encode([query])[0].astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    results = {}

    section_to_doc = text_levels['metadata']['section_to_doc']
    paragraph_to_section = text_levels['metadata']['paragraph_to_section']

    # Step 1: Document level search
    D, I = indices['document'].search(query_embedding, k=min(top_k, indices['document'].ntotal))
    results['document'] = {
        'indices': I[0].tolist(),
        'similarities': D[0].tolist(),
        'texts': [text_levels['document'][i] for i in I[0]],
    }

    # Step 2: Filter sections based on top documents
    relevant_section_indices = []
    for doc_idx in results['document']['indices']:
        for sec_idx, doc_of_sec in section_to_doc.items():
            if doc_of_sec == doc_idx:
                relevant_section_indices.append(sec_idx)

    if not relevant_section_indices:
        relevant_section_indices = list(range(len(text_levels['sections'])))

    filtered_section_embeddings = np.array([embeddings_dict['sections'][i] for i in relevant_section_indices]).astype('float32')
    faiss.normalize_L2(filtered_section_embeddings)

    section_dim = filtered_section_embeddings.shape[1]
    filtered_section_index = faiss.IndexFlatIP(section_dim)
    filtered_section_index.add(filtered_section_embeddings)

    D, I = filtered_section_index.search(query_embedding, k=min(top_k, filtered_section_index.ntotal))

    original_section_indices = [relevant_section_indices[i] for i in I[0]]

    results['section'] = {
        'indices': original_section_indices,
        'similarities': D[0].tolist(),
        'texts': [text_levels['sections'][i] for i in original_section_indices],
    }

    # Step 3: Filter paragraphs based on top sections (CORRECTED)
    relevant_paragraph_indices = []
    for sec_idx in results['section']['indices']:
        for para_idx, section_of_para in paragraph_to_section.items():
            if section_of_para == sec_idx:
                relevant_paragraph_indices.append(para_idx)

    if not relevant_paragraph_indices:
        relevant_paragraph_indices = list(range(len(text_levels['paragraphs'])))

    filtered_paragraph_embeddings = np.array([embeddings_dict['paragraphs'][i] for i in relevant_paragraph_indices]).astype('float32')
    faiss.normalize_L2(filtered_paragraph_embeddings)

    paragraph_dim = filtered_paragraph_embeddings.shape[1]
    filtered_paragraph_index = faiss.IndexFlatIP(paragraph_dim)
    filtered_paragraph_index.add(filtered_paragraph_embeddings)

    D, I = filtered_paragraph_index.search(query_embedding, k=min(top_k, filtered_paragraph_index.ntotal))

    original_paragraph_indices = [relevant_paragraph_indices[i] for i in I[0]]

    results['paragraph'] = {
        'indices': original_paragraph_indices,
        'similarities': D[0].tolist(),
        'texts': [text_levels['paragraphs'][i] for i in original_paragraph_indices],
    }

    return results