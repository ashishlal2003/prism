import os
from extraction import extract_text_from_pdf, extract_text_from_docx
from splitting import split_into_levels
from embeddings import create_multilevel_embeddings
from retrieval import build_indices_cosine

def get_multiple_documents():
    """Get paths for multiple document files"""

    documents = []
    document_paths = []

    print("\n" + "="*50)
    print("Multiple Document Processing")
    print("="*50)

    while True:
        file_path = input("\nEnter the path to a document file (PDF or DOCX) or 'done' when finished: ")

        if file_path.lower() == 'done':
            if not document_paths:
                print("Please add at least one document before proceeding.")
                continue
            else:
                break

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.pdf', '.docx', '.doc']:
            print(f"Unsupported file format: {file_ext}. Please use PDF or DOCX files.")
            continue

        document_paths.append(file_path)
        print(f"Added: {os.path.basename(file_path)}")

    # Process all documents
    for file_path in document_paths:
        file_ext = os.path.splitext(file_path)[1].lower()

        # Extract text based on file type
        if file_ext == '.pdf':
            document_text = extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            document_text = extract_text_from_docx(file_path)

        if document_text and document_text.strip():
            documents.append({
                'path': file_path,
                'name': os.path.basename(file_path),
                'text': document_text
            })
        else:
            print(f"Warning: Could not extract text from {os.path.basename(file_path)}")

    print(f"\nSuccessfully processed {len(documents)} documents")
    return documents

def process_multiple_documents(documents, model):
    """Process multiple documents into a unified multi-level structure"""

    all_text_levels = {
        'document': [],
        'sections': [],
        'paragraphs': [],
        'metadata': {
            'section_to_doc': {},
            'paragraph_to_section': {},
            'doc_names': []
        }
    }

    section_idx = 0
    paragraph_idx = 0

    # Process each document
    for doc_idx, doc in enumerate(documents):
        print(f"\nProcessing document {doc_idx+1}/{len(documents)}: {doc['name']}")

        # Process this document into levels
        doc_text_levels = split_into_levels(doc['text'])

        # Add document
        all_text_levels['document'].append(doc['text'])
        all_text_levels['metadata']['doc_names'].append(doc['name'])

        # Track sections for this document
        doc_section_start = section_idx

        # Add sections and track relationships
        for local_sec_idx, section in enumerate(doc_text_levels['sections']):
            all_text_levels['sections'].append(section)
            all_text_levels['metadata']['section_to_doc'][section_idx] = doc_idx
            section_idx += 1

        # Add paragraphs and track relationships
        for local_para_idx, paragraph in enumerate(doc_text_levels['paragraphs']):
            all_text_levels['paragraphs'].append(paragraph)

            # Get the section this paragraph belongs to
            local_section_idx = doc_text_levels['metadata']['paragraph_to_section'][local_para_idx]

            # Map to global section index
            global_section_idx = doc_section_start + local_section_idx

            all_text_levels['metadata']['paragraph_to_section'][paragraph_idx] = global_section_idx
            paragraph_idx += 1

    # Create embeddings
    print("\nCreating embeddings for all documents...")
    multilevel_embeddings = create_multilevel_embeddings(all_text_levels, model)

    # Build indices
    print("\nBuilding search indices...")
    indices = build_indices_cosine(multilevel_embeddings)

    return all_text_levels, multilevel_embeddings, indices

