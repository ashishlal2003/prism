import re
import numpy as np
from sentence_transformers import SentenceTransformer

def split_into_levels(document):
    # Document level stays the same
    document_level = document

    # Better section splitting - find all headings and extract content between them
    heading_pattern = re.compile(r'^#{1,3}\s+.*$', re.MULTILINE)
    heading_matches = list(heading_pattern.finditer(document))

    sections = []
    for i, match in enumerate(heading_matches):
        start_pos = match.start()
        # If last heading, take until end of document
        if i == len(heading_matches) - 1:
            end_pos = len(document)
        else:
            end_pos = heading_matches[i+1].start()

        section_content = document[start_pos:end_pos].strip()
        sections.append(section_content)

    # If no headings found, treat the whole document as one section
    if not sections:
        sections = [document]

    paragraphs = []

    paragraph_to_section = {}
    section_to_doc = {}

    for section_idx in range(len(sections)):
        section_to_doc[section_idx] = 0

    paragraph_idx = 0
    for section_idx, section in enumerate(sections):
        section_paragraphs = re.split(r'\n\n+', section)
        section_paragraphs = [p.strip() for p in section_paragraphs if p.strip()]

        for paragraph in section_paragraphs:
            paragraphs.append(paragraph)
            paragraph_to_section[paragraph_idx] = section_idx
            paragraph_idx += 1

    return {
        'document': [document_level],
        'sections': sections,
        'paragraphs': paragraphs,
        'metadata': {
            'section_to_doc': section_to_doc,
            'paragraph_to_section': paragraph_to_section
        }
    }