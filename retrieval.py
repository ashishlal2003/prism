import time
import faiss
import numpy as np

def progressive_search_cosine(query, indices, embeddings_dict, text_levels, model, data, top_k=3):
    """
    Search through embeddings at multiple levels with progressive filtering using cosine similarity
    """
    print(f"\n{'='*50}")
    print(f"PROGRESSIVE FILTERING SEARCH")
    print(f"{'='*50}")
    print(f"Query: '{query}'")
    print(f"Collection size: {len(text_levels['document'])} documents, {len(text_levels['sections'])} sections, {len(text_levels['paragraphs'])} paragraphs")
    
    overall_start = time.time()
    
    query_embedding = model.encode([query])[0].astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    print(f"Query embedded to vector dimension: {query_embedding.shape[1]}")

    results = {}

    section_to_doc = text_levels['metadata']['section_to_doc']
    paragraph_to_section = text_levels['metadata']['paragraph_to_section']

    # Step 1: Document level search
    print(f"\n{'='*50}")
    print(f"STEP 1: DOCUMENT LEVEL SEARCH")
    print(f"{'='*50}")
    doc_search_start = time.time()
    print(f"Searching across {indices['document'].ntotal} document vectors...")
    
    D, I = indices['document'].search(query_embedding, k=min(top_k, indices['document'].ntotal))
    doc_search_time = time.time() - doc_search_start
    data["Document Search Time"] = doc_search_time

    results['document'] = {
        'indices': I[0].tolist(),
        'similarities': D[0].tolist(),
        'texts': [text_levels['document'][i] for i in I[0]],
    }
    
    print(f"\n✓ Document search complete in {doc_search_time:.6f}s")
    print(f"  Selected {len(results['document']['indices'])} documents")
    for i, (idx, sim) in enumerate(zip(results['document']['indices'], results['document']['similarities'])):
        doc_name = text_levels['metadata']['doc_names'][idx]
        print(f"  Document {i+1}: {doc_name} (Similarity: {sim:.4f})")

    # Step 2: Filter sections based on top documents
    print(f"\n{'='*50}")
    print(f"STEP 2: SECTION LEVEL SEARCH")
    print(f"{'='*50}")
    section_search_start = time.time()
    
    relevant_section_indices = []
    for doc_idx in results['document']['indices']:
        for sec_idx, doc_of_sec in section_to_doc.items():
            if doc_of_sec == doc_idx:
                relevant_section_indices.append(sec_idx)

    print(f"Filtering sections from {len(results['document']['indices'])} selected documents...")
    print(f"Found {len(relevant_section_indices)} relevant sections out of {len(text_levels['sections'])} total sections")
    print(f"Filtering ratio: {len(relevant_section_indices) / len(text_levels['sections']):.2%}")
    
    if not relevant_section_indices:
        print("No relevant sections found. Falling back to all sections.")
        relevant_section_indices = list(range(len(text_levels['sections'])))

    filtered_section_embeddings = np.array([embeddings_dict['sections'][i] for i in relevant_section_indices]).astype('float32')
    faiss.normalize_L2(filtered_section_embeddings)

    section_dim = filtered_section_embeddings.shape[1]
    filtered_section_index = faiss.IndexFlatIP(section_dim)
    filtered_section_index.add(filtered_section_embeddings)
    
    print(f"Searching across {filtered_section_index.ntotal} filtered section vectors...")
    D, I = filtered_section_index.search(query_embedding, k=min(top_k, filtered_section_index.ntotal))

    original_section_indices = [relevant_section_indices[i] for i in I[0]]

    results['section'] = {
        'indices': original_section_indices,
        'similarities': D[0].tolist(),
        'texts': [text_levels['sections'][i] for i in original_section_indices],
    }
    
    section_search_time = time.time() - section_search_start
    data["Section Search Time"] = section_search_time

    print(f"\n✓ Section search complete in {section_search_time:.6f}s")
    print(f"  Selected {len(results['section']['indices'])} sections")
    for i, (idx, sim) in enumerate(zip(results['section']['indices'], results['section']['similarities'])):
        doc_idx = text_levels['metadata']['section_to_doc'][idx]
        doc_name = text_levels['metadata']['doc_names'][doc_idx]
        section_preview = text_levels['sections'][idx][:50].replace('\n', ' ') + "..."
        print(f"  Section {i+1}: from {doc_name} (Similarity: {sim:.4f}) - {section_preview}")

    # Step 3: Filter paragraphs based on top sections
    print(f"\n{'='*50}")
    print(f"STEP 3: PARAGRAPH LEVEL SEARCH")
    print(f"{'='*50}")
    para_search_start = time.time()
    
    relevant_paragraph_indices = []
    for sec_idx in results['section']['indices']:
        for para_idx, section_of_para in paragraph_to_section.items():
            if section_of_para == sec_idx:
                relevant_paragraph_indices.append(para_idx)

    print(f"Filtering paragraphs from {len(results['section']['indices'])} selected sections...")
    print(f"Found {len(relevant_paragraph_indices)} relevant paragraphs out of {len(text_levels['paragraphs'])} total paragraphs")
    print(f"Filtering ratio: {len(relevant_paragraph_indices) / len(text_levels['paragraphs']):.2%}")
    
    if not relevant_paragraph_indices:
        print("No relevant paragraphs found. Falling back to all paragraphs.")
        relevant_paragraph_indices = list(range(len(text_levels['paragraphs'])))

    filtered_paragraph_embeddings = np.array([embeddings_dict['paragraphs'][i] for i in relevant_paragraph_indices]).astype('float32')
    faiss.normalize_L2(filtered_paragraph_embeddings)

    paragraph_dim = filtered_paragraph_embeddings.shape[1]
    filtered_paragraph_index = faiss.IndexFlatIP(paragraph_dim)
    filtered_paragraph_index.add(filtered_paragraph_embeddings)

    print(f"Searching across {filtered_paragraph_index.ntotal} filtered paragraph vectors...")
    D, I = filtered_paragraph_index.search(query_embedding, k=min(top_k, filtered_paragraph_index.ntotal))

    original_paragraph_indices = [relevant_paragraph_indices[i] for i in I[0]]

    results['paragraph'] = {
        'indices': original_paragraph_indices,
        'similarities': D[0].tolist(),
        'texts': [text_levels['paragraphs'][i] for i in original_paragraph_indices],
    }
    
    para_search_time = time.time() - para_search_start
    data["Paragraph Search Time"] = para_search_time

    total_time = para_search_time + section_search_time + doc_search_time
    data["Total Retrieval Time"] = total_time

    print(f"\n✓ Paragraph search complete in {para_search_time:.6f}s")
    print(f"  Selected {len(results['paragraph']['indices'])} paragraphs")
    for i, (idx, sim) in enumerate(zip(results['paragraph']['indices'], results['paragraph']['similarities'])):
        section_idx = text_levels['metadata']['paragraph_to_section'][idx]
        doc_idx = text_levels['metadata']['section_to_doc'][section_idx]
        doc_name = text_levels['metadata']['doc_names'][doc_idx]
        para_preview = text_levels['paragraphs'][idx][:50].replace('\n', ' ') + "..."
        print(f"  Paragraph {i+1}: from {doc_name} (Similarity: {sim:.4f}) - {para_preview}")

    # Performance summary
    # print(f"\n{'='*50}")
    # print(f"PERFORMANCE SUMMARY - PRISM APPROACH")
    # print(f"{'='*50}")
    # print(f"Total search time: {total_time:.6f}s")
    # print(f"  - Document search: {doc_search_time:.6f}s ({doc_search_time/total_time*100:.1f}%)")
    # print(f"  - Section search: {section_search_time:.6f}s ({section_search_time/total_time*100:.1f}%)")
    # print(f"  - Paragraph search: {para_search_time:.6f}s ({para_search_time/total_time*100:.1f}%)")
    # print(f"Vector comparisons:")
    # print(f"  - Document level: {indices['document'].ntotal} vectors (100% of documents)")
    # print(f"  - Section level: {filtered_section_index.ntotal} vectors ({filtered_section_index.ntotal/len(text_levels['sections'])*100:.1f}% of all sections)")
    # print(f"  - Paragraph level: {filtered_paragraph_index.ntotal} vectors ({filtered_paragraph_index.ntotal/len(text_levels['paragraphs'])*100:.1f}% of all paragraphs)")
    # print(f"Total vector comparisons: {indices['document'].ntotal + filtered_section_index.ntotal + filtered_paragraph_index.ntotal}")
    # print(f"Flat comparison would require: {len(text_levels['paragraphs'])} vector comparisons")
    # print(f"Vector comparison reduction: {(1 - (indices['document'].ntotal + filtered_section_index.ntotal + filtered_paragraph_index.ntotal)/len(text_levels['paragraphs']))*100:.1f}%")

    return results