# src/prompt_builder.py


def build_analysis_prompt(retrieved_chunks, job_description):
    """
    Build the RAG prompt by injecting retrieved resume context
    into a structured analysis template.
    
    This is the key RAG step:
      retrieved_chunks = what was found in the knowledge base (resume)
      job_description  = the query we searched against
    
    The LLM is instructed to use ONLY the context — not its own
    training knowledge — to prevent hallucination.
    """
    
    # Format chunks with their relevance scores
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Resume Section {i} | Relevance: {score}]\n{chunk}")
    
    context = "\n\n".join(context_blocks)
    
    prompt = f"""You are a professional career analyst performing a RAG-based resume analysis.

You have been given RETRIEVED RESUME SECTIONS (most relevant to the job description) and a JOB DESCRIPTION.
Use ONLY the information in the resume sections below — do not assume or invent any skills.

════════════════════════════════════════
RETRIEVED RESUME SECTIONS:
════════════════════════════════════════
{context}

════════════════════════════════════════
JOB DESCRIPTION:
════════════════════════════════════════
{job_description}

════════════════════════════════════════
YOUR ANALYSIS TASK:
════════════════════════════════════════

Please provide a structured analysis with these exact sections:

MATCH SCORE: [Give a number 0-100 based on how well the resume matches]

MATCHED SKILLS: [List skills/experiences found in BOTH the resume and JD]

MISSING SKILLS: [List skills required in JD but NOT found in resume sections]

TOP STRENGTH: [The single most compelling thing about this candidate for this role]

KEY GAP: [The most critical missing requirement]

RESUME IMPROVEMENT: [One specific bullet point the candidate should add to their resume]

INTERVIEW TIP: [One likely interview question based on the JD and a tip to answer it]

Be specific. Use details from the resume sections provided. Do not fabricate experience.
"""
    return prompt


# ── Test it standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    sample_chunks = [
        ("Python developer with experience in data analysis and automation", 0.91),
        ("Built daily sales reporting tool using pandas and openpyxl", 0.85),
    ]
    sample_jd = "We are looking for a Data Analyst with Python and SQL skills."
    
    prompt = build_analysis_prompt(sample_chunks, sample_jd)
    print(prompt)
    print(f"\nPrompt length: {len(prompt.split())} words")