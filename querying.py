from sentence_transformers import SentenceTransformer
from document import get_multiple_documents, process_multiple_documents
from retrieval import progressive_search_cosine

def multi_document_query_interface(all_text_levels, multilevel_embeddings, indices, model):
    """Query interface for multiple documents"""

    print("\n" + "="*50)
    print("Multi-Document Query System")
    print(f"Documents loaded: {len(all_text_levels['document'])}")
    for i, name in enumerate(all_text_levels['metadata']['doc_names']):
        print(f"  {i+1}. {name}")
    print("="*50)
    print("Type your query or 'quit' to exit")
    print("="*50)

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Multi-Document Query System!")
            break

        if not query.strip():
            print("Please enter a valid query")
            continue

        print("\nSearching across all documents...")

        # Use our progressive search function with cosine similarity
        results = progressive_search_cosine(query, indices, multilevel_embeddings, all_text_levels, model)

        # Display results
        print("\n" + "-"*50)
        print(f"Results for: {query}")
        print("-"*50)

        print("\n#### Document Level Matches")
        for i, (idx, sim) in enumerate(zip(results['document']['indices'], results['document']['similarities'])):
            doc_name = all_text_levels['metadata']['doc_names'][idx]
            print(f"\nDocument {i+1}: {doc_name} (Similarity: {sim:.4f})")

        print("\n#### Top Section Matches")
        for i, (idx, sim) in enumerate(zip(results['section']['indices'], results['section']['similarities'])):
            # Get document this section belongs to
            doc_idx = all_text_levels['metadata']['section_to_doc'][idx]
            doc_name = all_text_levels['metadata']['doc_names'][doc_idx]

            section_text = all_text_levels['sections'][idx]
            print(f"\nSection {i+1} from {doc_name} (Similarity: {sim:.4f})")

            # Limit length for display clarity
            if len(section_text) > 500:
                print(section_text[:500] + "...")
            else:
                print(section_text)

        print("\n#### Top Paragraph Matches")
        for i, (idx, sim) in enumerate(zip(results['paragraph']['indices'], results['paragraph']['similarities'])):
            # Get section this paragraph belongs to
            section_idx = all_text_levels['metadata']['paragraph_to_section'][idx]
            # Get document this section belongs to
            doc_idx = all_text_levels['metadata']['section_to_doc'][section_idx]
            doc_name = all_text_levels['metadata']['doc_names'][doc_idx]

            paragraph_text = all_text_levels['paragraphs'][idx]
            print(f"\nParagraph {i+1} from {doc_name} (Similarity: {sim:.4f})")
            print(paragraph_text)

        print("\n" + "-"*50)

def run_multi_document_rag_system():
    """Run the complete multi-document RAG system"""

    # Load the model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")

    # Get multiple documents
    documents = get_multiple_documents()

    if not documents:
        print("No documents were processed successfully. Exiting.")
        return

    # Process all documents
    all_text_levels, multilevel_embeddings, indices = process_multiple_documents(documents, model)

    # Start the multi-document query interface
    multi_document_query_interface(all_text_levels, multilevel_embeddings, indices, model)