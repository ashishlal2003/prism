from sentence_transformers import SentenceTransformer
from document import get_documents_from_folder, process_multiple_documents
from retrieval import progressive_search_cosine
import time
from logs import append_to_json_file


def multi_document_query_interface(
    all_text_levels, multilevel_embeddings, indices, model
):
    """Query interface for multiple documents with performance tracking"""

    data = {}
    file_path = "prism_data.json"
    print("\n" + "=" * 50)
    print("Multi-Document Query System")
    print(f"Documents loaded: {len(all_text_levels['document'])}")
    for i, name in enumerate(all_text_levels["metadata"]["doc_names"]):
        print(f"  {i+1}. {name}")
    print("=" * 50)

    # Collection statistics
    print(f"Collection statistics:")
    print(f"  • Documents: {len(all_text_levels['document'])}")
    print(f"  • Sections: {len(all_text_levels['sections'])}")
    print(f"  • Paragraphs: {len(all_text_levels['paragraphs'])}")

    print("=" * 50)
    print("Type your query or 'quit' to exit")
    print("=" * 50)

    # Track session performance
    session_stats = {"queries": 0, "total_time": 0, "total_comparisons": 0}

    while True:
        
        query = input("\nEnter your query: ")
        data["query"] = query

        if query.lower() in ["quit", "exit", "q"]:
            # Show session summary
            if session_stats["queries"] > 0:
                print("\n" + "=" * 50)
                print("SESSION PERFORMANCE SUMMARY")
                print("=" * 50)
                print(f"Queries executed: {session_stats['queries']}")
                print(
                    f"Average search time: {session_stats['total_time']/session_stats['queries']:.4f} seconds"
                )
                print(
                    f"Average vector comparisons: {session_stats['total_comparisons']/session_stats['queries']:.1f}"
                )
                print(
                    f"Comparison reduction: {(1 - session_stats['total_comparisons']/(session_stats['queries']*len(all_text_levels['paragraphs'])))*100:.1f}% vs flat"
                )

                # logging
                data["Queries executed"] = session_stats["queries"]
                data["Average search time"] = (
                    session_stats["total_time"] / session_stats["queries"]
                )
                data["Average vector comparisons"] = (
                    session_stats["total_comparisons"] / session_stats["queries"]
                )
                data["Comparison reduction"] = (
                    1
                    - session_stats["total_comparisons"]
                    / (session_stats["queries"] * len(all_text_levels["paragraphs"]))
                ) * 100

            print("\nThank you for using the Multi-Document Query System!")
            break

        if not query.strip():
            print("Please enter a valid query")
            continue

        print("\nSearching across all documents...")

        start = time.time()

        # Use our progressive search function with cosine similarity
        results = progressive_search_cosine(
            query, indices, multilevel_embeddings, all_text_levels, model, data
        )
        search_time = time.time() - start
        data["Search Time"] = search_time

        # Update session stats (extract comparison count from results if needed)
        session_stats["queries"] += 1
        session_stats["total_time"] += search_time

        # Extract comparison count from output or estimate it
        # (This assumes progressive_search_cosine has been modified to return this info)
        est_comparisons = indices["document"].ntotal
        if "filtered_section_count" in results:
            est_comparisons += results["filtered_section_count"]
        if "filtered_paragraph_count" in results:
            est_comparisons += results["filtered_paragraph_count"]
        else:
            # Rough estimate if not available
            est_comparisons += len(results["section"]["indices"]) * 10

        session_stats["total_comparisons"] += est_comparisons

        print(f"\nSearch completed in {search_time:.2f} seconds")

        # Display results
        print("\n" + "-" * 50)
        print(f"Results for: {query}")
        print("-" * 50)

        print("\n#### Document Level Matches")
        for i, (idx, sim) in enumerate(
            zip(results["document"]["indices"], results["document"]["similarities"])
        ):
            doc_name = all_text_levels["metadata"]["doc_names"][idx]
            print(f"\nDocument {i+1}: {doc_name} (Similarity: {sim:.4f})")
            data[f"Document {i+1}"] = f"{doc_name} (Similarity: {sim:.4f})"

        print("\n#### Top Section Matches")
        for i, (idx, sim) in enumerate(
            zip(results["section"]["indices"], results["section"]["similarities"])
        ):
            # Get document this section belongs to
            doc_idx = all_text_levels["metadata"]["section_to_doc"][idx]
            doc_name = all_text_levels["metadata"]["doc_names"][doc_idx]

            section_text = all_text_levels["sections"][idx]
            print(f"\nSection {i+1} from {doc_name} (Similarity: {sim:.4f})")
            data[f"Section {i+1}"] = f"{doc_name} (Similarity: {sim:.4f})"
            data[f"Section {i+1} Text"] = section_text

            # Limit length for display clarity
            if len(section_text) > 500:
                print(section_text[:500] + "...")
            else:
                print(section_text)

        print("\n#### Top Paragraph Matches")
        for i, (idx, sim) in enumerate(
            zip(results["paragraph"]["indices"], results["paragraph"]["similarities"])
        ):
            # Get section this paragraph belongs to
            section_idx = all_text_levels["metadata"]["paragraph_to_section"][idx]
            # Get document this section belongs to
            doc_idx = all_text_levels["metadata"]["section_to_doc"][section_idx]
            doc_name = all_text_levels["metadata"]["doc_names"][doc_idx]

            paragraph_text = all_text_levels["paragraphs"][idx]
            print(f"\nParagraph {i+1} from {doc_name} (Similarity: {sim:.4f})")
            data[f"Paragraph {i+1}"] = f"{doc_name} (Similarity: {sim:.4f})"
            data[f"Paragraph {i+1} Text"] = paragraph_text

            print(paragraph_text)
        
        duration = time.time() - start
        data["Total Duration"] = duration
        print("\n" + "-" * 50)
    
        print("calling append function")
        append_to_json_file(data, file_path)
        print("DATA APPENDED SUCESSFULLY!!!!!!!!!!!!!!!")


# Update your run_multi_document_rag_system function to use this new function
def run_multi_document_rag_system():
    """Run the complete multi-document RAG system with folder loading"""

    # Load the model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully!")

    # Get folder path
    folder_path = input("Enter folder path containing documents: ")

    # Load all documents from folder
    documents = get_documents_from_folder(folder_path)

    if not documents:
        print("No documents were processed successfully. Exiting.")
        return

    # Process all documents
    all_text_levels, multilevel_embeddings, indices = process_multiple_documents(
        documents, model
    )

    # Start the multi-document query interface
    multi_document_query_interface(
        all_text_levels, multilevel_embeddings, indices, model
    )
