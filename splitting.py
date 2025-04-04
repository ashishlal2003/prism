import re
import numpy as np
from sentence_transformers import SentenceTransformer

def split_into_levels(document):
    """Split document into hierarchical levels based on Markdown heading syntax"""
    # Document level stays the same
    document_level = document
    
    # Section splitting based on level 1 and 2 headings (# and ##)
    section_pattern = re.compile(r'^#{1,2}\s+.*?(?=^#{1,2}\s+|\Z)', re.MULTILINE | re.DOTALL)
    sections = []
    
    # Find all sections based on heading patterns
    for match in section_pattern.finditer(document):
        section_content = match.group(0).strip()
        if section_content:
            sections.append(section_content)
    
    # If no sections found, use the whole document
    if not sections:
        sections = [document]
    
    # Extract paragraphs and track relationships
    paragraphs = []
    paragraph_to_section = {}
    section_to_doc = {}
    
    # Map all sections to document ID 0 (single document case)
    for section_idx in range(len(sections)):
        section_to_doc[section_idx] = 0
    
    # Process each section to extract paragraphs
    for section_idx, section in enumerate(sections):
        # Split section into paragraphs (blank line separation)
        section_paragraphs = re.split(r'\n\n+', section)
        section_paragraphs = [p.strip() for p in section_paragraphs if p.strip()]
        
        # Add paragraphs and track their relationship to parent section
        for paragraph in section_paragraphs:
            paragraph_idx = len(paragraphs)
            paragraphs.append(paragraph)
            paragraph_to_section[paragraph_idx] = section_idx
    
    return {
        'document': [document_level],
        'sections': sections,
        'paragraphs': paragraphs,
        'metadata': {
            'section_to_doc': section_to_doc,
            'paragraph_to_section': paragraph_to_section
        }
    }