# src/chunker.py

def chunk_text(text, chunk_size=200, overlap=40):
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text       : the full resume string
        chunk_size : how many words per chunk
        overlap    : how many words to repeat between consecutive chunks
    
    Returns:
        List of string chunks
    """
    words = text.split()
    if not words:
        return []
        
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap) # Fix: prevent infinite loop
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step
    
    return chunks


def chunk_by_sections(text):
    """
    Alternative: split resume by natural sections (Experience, Skills, etc.)
    Falls back to word chunking if natural sections cannot be determined.
    """
    if not text.strip():
        return []

    # Common resume section headers
    section_keywords = [
        "experience", "education", "skills", "projects",
        "certifications", "summary", "objective", "achievements"
    ]
    
    lines = text.split("\n")
    sections = []
    current_section = []
    
    for line in lines:
        stripped_line = line.strip().lower()
        # Fix: Check if line is short and strictly starts with/equals a keyword
        is_header = len(stripped_line) < 50 and any(
            stripped_line.startswith(kw) or stripped_line == kw 
            for kw in section_keywords
        )
        
        if is_header and current_section:
            sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append("\n".join(current_section))
    
    # Fix: Only return sections if we actually found distinct sections, else fallback
    return sections if len(sections) > 1 else chunk_text(text)

if __name__ == "__main__":
    sample = """
    John Doe - Data Analyst
    Skills: Python, SQL, Excel, Power BI, Machine Learning
    Experience: 2 years as Medical Representative at ABC Pharma
    Built daily sales automation tool using Python
    Education: B.Sc Computer Science
    """
    
    chunks = chunk_text(sample, chunk_size=20, overlap=5)
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}]: {chunk}")
