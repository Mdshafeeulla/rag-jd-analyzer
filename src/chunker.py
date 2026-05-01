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
    # Split the text into individual words
    words = text.split()
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Take 'chunk_size' words starting from 'start'
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap)
        # This means the next chunk starts 'overlap' words before the current end
        start += chunk_size - overlap
    
    return chunks


def chunk_by_sections(text):
    """
    Alternative: split resume by natural sections (Experience, Skills, etc.
    """
    # Common resume section headers
    section_keywords = [
        "experience", "education", "skills", "projects",
        "certifications", "summary", "objective", "achievements"
    ]
    
    lines = text.split("\n")
    sections = []
    current_section = []
    
    for line in lines:
        # Check if this line is a section header
        is_header = any(kw in line.lower() for kw in section_keywords)
        
        if is_header and current_section:
            # Save previous section, start new one
            sections.append("\n".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    # Don't forget the last section
    if current_section:
        sections.append("\n".join(current_section))
    
    return sections if sections else chunk_text(text)  # fallback to word chunks

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
