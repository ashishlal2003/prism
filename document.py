import os
from extraction import extract_text_from_pdf, extract_text_from_docx
from splitting import split_into_levels
from embeddings import create_multilevel_embeddings, build_indices_cosine
import time

def get_documents_from_folder(folder_path):
    """Automatically load all compatible documents from a folder"""
    
    documents = []
    
    print("\n" + "="*50)
    print("DOCUMENT LOADING PROCESS - PRISM")
    print("="*50)
    
    load_start = time.time()
    total_chars = 0
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return []
        
    print(f"Scanning directory: {folder_path}")
    
    # Find all compatible files
    file_paths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in ['.pdf', '.docx', '.doc']:
            file_paths.append(file_path)
    
    print(f"Found {len(file_paths)} compatible documents")
            
    # Process each file
    for file_path in file_paths:
        file_start = time.time()
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"Loading: {filename} ({file_size:.2f} KB)")
        
        try:
            # Extract text based on file type
            if file_ext in ['.docx', '.doc']:
                document_text = extract_text_from_docx(file_path)
            elif file_ext == '.pdf':
                document_text = extract_text_from_pdf(file_path)
            
            if document_text and document_text.strip():
                char_count = len(document_text)
                total_chars += char_count
                documents.append({
                    'path': file_path,
                    'name': filename,
                    'text': document_text
                })
                extraction_time = time.time() - file_start
                print(f"  ✓ Extracted {char_count:,} characters in {extraction_time:.2f}s ({char_count/extraction_time:.0f} chars/sec)")
            else:
                print(f"  ⚠ Warning: Could not extract text from {filename}")
                
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
    
    load_time = time.time() - load_start
    print(f"\nSuccessfully processed {len(documents)} documents")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total processing time: {load_time:.2f}s")
    print(f"  Average processing speed: {total_chars/load_time:.0f} chars/sec")
    
    return documents



def process_multiple_documents(documents, model):
    """Process multiple documents into a unified multi-level structure with detailed logging"""
    overall_start = time.time()
    
    print(f"\n{'='*50}")
    print(f"DOCUMENT STRUCTURAL ANALYSIS")
    print(f"{'='*50}")

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
    total_chars = sum(len(doc['text']) for doc in documents)
    
    # Process each document
    for doc_idx, doc in enumerate(documents):
        doc_start = time.time()
        print(f"\nProcessing document {doc_idx+1}/{len(documents)}: {doc['name']} ({len(doc['text']):,} chars)")

        # Process this document into levels
        level_start = time.time()
        doc_text_levels = split_into_levels(doc['text'])
        level_time = time.time() - level_start
        
        section_count = len(doc_text_levels['sections'])
        paragraph_count = len(doc_text_levels['paragraphs'])
        
        print(f"  ✓ Split into {section_count} sections and {paragraph_count} paragraphs in {level_time:.2f}s")
        print(f"    Avg section size: {sum(len(s) for s in doc_text_levels['sections'])/section_count:.0f} chars")
        print(f"    Avg paragraph size: {sum(len(p) for p in doc_text_levels['paragraphs'])/paragraph_count:.0f} chars")

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
            
        doc_time = time.time() - doc_start
        print(f"  ✓ Document processing completed in {doc_time:.2f}s")

    # Structural analysis summary
    analysis_time = time.time() - overall_start
    print(f"\n{'='*50}")
    print(f"STRUCTURAL ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Processed {len(documents)} documents with {len(all_text_levels['sections']):,} sections and {len(all_text_levels['paragraphs']):,} paragraphs")
    print(f"  Documents: {len(all_text_levels['document'])}")
    print(f"  Sections: {len(all_text_levels['sections'])}")
    print(f"  Paragraphs: {len(all_text_levels['paragraphs'])}")
    print(f"  Analysis time: {analysis_time:.2f}s ({total_chars/analysis_time:.0f} chars/sec)")

    # Create embeddings (this already has timing in the function)
    print(f"\n{'='*50}")
    print(f"MULTI-LEVEL EMBEDDING CREATION")
    print(f"{'='*50}")
    embedding_start = time.time()
    multilevel_embeddings = create_multilevel_embeddings(all_text_levels, model)
    embedding_time = time.time() - embedding_start

    # Build indices (this already has timing in the function)
    print(f"\n{'='*50}")
    print(f"SEARCH INDEX CONSTRUCTION")
    print(f"{'='*50}")
    index_start = time.time()
    indices = build_indices_cosine(multilevel_embeddings)
    index_time = time.time() - index_start
    
    # Overall processing summary
    total_time = time.time() - overall_start
    print(f"\n{'='*50}")
    print(f"PRISM PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"  - Structural analysis: {analysis_time:.2f}s ({analysis_time/total_time*100:.1f}%)")
    print(f"  - Embedding creation: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
    print(f"  - Index building: {index_time:.2f}s ({index_time/total_time*100:.1f}%)")
    
    # Storage metrics
    embedding_bytes = sum(e.nbytes for emb_list in multilevel_embeddings.values() for e in emb_list) if multilevel_embeddings else 0
    embedding_mb = embedding_bytes / (1024 * 1024)
    print(f"Storage requirements:")
    print(f"  - Embeddings: {embedding_mb:.2f} MB")
    print(f"  - Document count: {len(all_text_levels['document'])}")
    print(f"  - Section count: {len(all_text_levels['sections'])}")
    print(f"  - Paragraph count: {len(all_text_levels['paragraphs'])}")

    return all_text_levels, multilevel_embeddings, indices