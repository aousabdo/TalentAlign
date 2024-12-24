import sys
import re
import string
import numpy as np
import PyPDF2
import concurrent.futures
import os

from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from openai import OpenAI, APIConnectionError, APIStatusError

###############################################################################
# GLOBAL PARAMETERS
###############################################################################
ALPHA = 0.3            # Weight of doc-level embedding vs. TF-IDF
BULLET_WEIGHT = 0.5    # Weight for bullet-based lines
SECTION_WEIGHT = 0.5   # Weight for section-level alignment
CLAMP_THRESHOLD = 0.7  # If doc-sim < 0.7 => clamp to 0.0
TOP_N_KEYWORDS = 10

# Hard requirement markers
HARD_REQ_MARKERS = {
    "required", "must", "mandatory", "doctor", "residency", "board certification",
    "10+ years",  # example marker for product manager, etc.
}

###############################################################################
# 1. PDF Extraction
###############################################################################
def extract_text_from_pdf(pdf_path):
    text_content = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

###############################################################################
# 2. Parse Resume Sections
###############################################################################
def parse_resume_sections(resume_text):
    """
    Attempt to locate typical resume headings:
    'Key Skills', 'Skills', 'Education', 'Experience', 'Publications',
    'Honors', 'Awards', etc.
    
    Store them as {section_name -> text}, with 'Misc' for unknown.
    """
    headings = [
        "skills", "key skills", "experience", "professional experience",
        "education", "publications", "honors", "awards", "certifications"
    ]

    lines = resume_text.splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    sections = {}
    current_section = "Misc"
    sections[current_section] = []

    for line in lines:
        lower_line = line.lower()
        matched_heading = None
        for hd in headings:
            # If 'hd' is found near start of line
            if hd in lower_line and lower_line.index(hd) < 5:
                matched_heading = hd.title()
                break

        if matched_heading:
            current_section = matched_heading
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    # Convert lists to strings
    for sec in sections:
        sections[sec] = "\n".join(sections[sec]).strip()

    return sections

###############################################################################
# 3. Parse JD Sections
###############################################################################
def parse_jd_sections(jd_text):
    """
    Captures typical headings like 'Responsibilities', 'Requirements', 'Preferred', etc.
    Everything else goes to 'Misc'.
    """
    lines = jd_text.splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    known_headings = {
        "responsibilities", "requirements", "preferred", "benefits",
        "how to apply", "job description", "about us", "education", "notes"
    }

    sections = {}
    current_section = "Misc"
    sections[current_section] = []

    for line in lines:
        lower_line = line.lower()
        matched = None
        for hd in known_headings:
            if lower_line.startswith(hd):
                matched = hd.title()
                break

        if matched:
            current_section = matched
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    for sec in sections:
        sections[sec] = "\n".join(sections[sec]).strip()

    return sections

###############################################################################
# 4. Must-Have Requirements
###############################################################################
def check_hard_requirements(requirements_text, resume_text):
    """
    If 'requirements_text' has lines containing any HARD_REQ_MARKERS,
    we ensure the resume also has them. If missing => final=0.
    """
    req_lower = requirements_text.lower()
    res_lower = resume_text.lower()

    for marker in HARD_REQ_MARKERS:
        if marker in req_lower and marker not in res_lower:
            return False
    return True

###############################################################################
# 5. Embedding & Cosine
###############################################################################
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def embed_text(client, text):
    """
    Single chunk embedding. A small function used by parallel calls as well.
    """
    if not text.strip():
        return None
    try:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        return resp.data[0].embedding
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error embedding chunk: {str(e)}")
        return None

###############################################################################
# 6. Section-Level Matching
###############################################################################
def match_jd_sections_with_resume(client, jd_sections, resume_sections):
    """
    For each JD section, embed it, then find the best matching resume section by cos sim.
    Return average of best-match scores across all JD sections.
    """
    # Pre-embed resume sections (in parallel for speed)
    resume_embs = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_map = {}
        for r_sec_name, r_sec_text in resume_sections.items():
            future = executor.submit(embed_text, client, r_sec_text)
            future_map[future] = r_sec_name

        for fut in concurrent.futures.as_completed(future_map):
            r_name = future_map[fut]
            resume_embs[r_name] = fut.result()

    sims = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_jd = {}
        # For each JD section, embed in parallel
        for jd_sec_name, jd_sec_text in jd_sections.items():
            future = executor.submit(embed_text, client, jd_sec_text)
            future_jd[future] = jd_sec_name

        jd_embs = {}
        for fut in concurrent.futures.as_completed(future_jd):
            j_name = future_jd[fut]
            jd_embs[j_name] = fut.result()

        # Now compute best match for each JD section
        for jd_sec_name, jd_emb in jd_embs.items():
            if jd_emb is None:
                continue
            best_score = 0.0
            for r_sec_name, r_emb in resume_embs.items():
                if r_emb is None:
                    continue
                score = cosine_similarity(jd_emb, r_emb)
                if score > best_score:
                    best_score = score
            sims.append(best_score)

    if sims:
        return sum(sims)/len(sims)
    return 1.0  # If no meaningful JD sections, return 1 to not penalize

###############################################################################
# 7. Bullet-Based Matching
###############################################################################
def parse_bullet_sections_for_jd(jd_text):
    sections = parse_jd_sections(jd_text)
    resp_text = sections.get("Responsibilities", "")
    req_text  = sections.get("Requirements", "")

    resp_lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
    req_lines  = [ln.strip() for ln in req_text.splitlines() if ln.strip()]

    return resp_lines, req_lines

def average_bullet_similarity(client, lines, resume_emb):
    if not lines:
        return 1.0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_map = {}
        for line in lines:
            if line.strip():
                future = executor.submit(embed_text, client, line)
                future_map[future] = line

        sims = []
        for fut in concurrent.futures.as_completed(future_map):
            bullet_emb = fut.result()
            if bullet_emb is not None and resume_emb is not None:
                sim = cosine_similarity(bullet_emb, resume_emb)
                sims.append(sim)
        if sims:
            return sum(sims)/len(sims)
        return 0.0

###############################################################################
# 8. TF-IDF
###############################################################################
def extract_top_keywords_from_text(text, top_n=TOP_N_KEYWORDS):
    text = text.strip()
    if not text:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,1))
    matrix = vec.fit_transform([text])
    feats = vec.get_feature_names_out()
    scores = matrix.toarray()[0]
    pairs = list(zip(feats, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_s in pairs[:top_n]]

def keyword_match_score(resume_text, jd_keywords):
    if not jd_keywords:
        return 0.0
    r_lower = resume_text.lower()
    matched = sum(kw.lower() in r_lower for kw in jd_keywords)
    return matched / len(jd_keywords)

###############################################################################
# 9. Final Score Combination
###############################################################################
def final_score(doc_level, bullet_avg, section_avg):
    """
    doc_level = alpha*doc_sim + (1-alpha)*tfidf
    bullet_avg = bullet-based alignment
    section_avg = section-level alignment

    1) doc_bullet = geometric combination of doc_level & bullet_avg
        doc_bullet = (doc_level^(1-BULLET_WEIGHT)) * (bullet_avg^(BULLET_WEIGHT))
    2) final = geometric combination of doc_bullet & section_avg
        final = (doc_bullet^(1-SECTION_WEIGHT)) * (section_avg^(SECTION_WEIGHT))
    """
    doc_bullet = (doc_level ** (1-BULLET_WEIGHT)) * (bullet_avg ** BULLET_WEIGHT)
    final_val  = (doc_bullet ** (1-SECTION_WEIGHT)) * (section_avg ** SECTION_WEIGHT)
    return final_val

###############################################################################
# MAIN
###############################################################################
def main():
    if len(sys.argv) != 3:
        print("Usage: python match_resume_jd.py <resume.pdf> <jd.pdf>")
        sys.exit(1)

    resume_pdf = sys.argv[1]
    jd_pdf     = sys.argv[2]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1) Extract text
    resume_text = extract_text_from_pdf(resume_pdf)
    jd_text     = extract_text_from_pdf(jd_pdf)

    # 2) Parse JD sections
    jd_sections = parse_jd_sections(jd_text)
    # If there's a "Requirements" section
    req_text = jd_sections.get("Requirements", "")
    # Must-have check
    if not check_hard_requirements(req_text, resume_text):
        print("\nResume fails must-have/hard requirements!")
        print("Final Combined Score: 0.0000")
        return

    # 3) Doc-level similarity
    try:
        emb_resume = client.embeddings.create(model="text-embedding-ada-002", input=[resume_text])
        emb_jd     = client.embeddings.create(model="text-embedding-ada-002", input=[jd_text])
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error: {str(e)}")
        sys.exit(1)

    r_emb = emb_resume.data[0].embedding
    j_emb = emb_jd.data[0].embedding

    doc_sim = cosine_similarity(r_emb, j_emb)
    if CLAMP_THRESHOLD and doc_sim < CLAMP_THRESHOLD:
        doc_sim = 0.0

    # 4) TF-IDF
    jd_keywords = extract_top_keywords_from_text(jd_text, TOP_N_KEYWORDS)
    tfidf_sc = keyword_match_score(resume_text, jd_keywords)

    doc_level = ALPHA*doc_sim + (1-ALPHA)*tfidf_sc

    # 5) Bullet-based from step #1
    resp_lines, req_lines = parse_bullet_sections_for_jd(jd_text)
    bullet_sc_resp = average_bullet_similarity(client, resp_lines, r_emb)
    bullet_sc_req  = average_bullet_similarity(client, req_lines,  r_emb)
    bullet_avg     = (bullet_sc_resp + bullet_sc_req)/2.0

    # 6) Section-level approach
    resume_sections = parse_resume_sections(resume_text)
    section_avg     = match_jd_sections_with_resume(client, jd_sections, resume_sections)

    # 7) final
    final_val = final_score(doc_level, bullet_avg, section_avg)

    # Print results
    print("\n=== SECTION-LEVEL PARSING RESULTS ===")
    print("Resume Sections Found:")
    for k,v in resume_sections.items():
        print(f"  [{k}] => {len(v)} chars")

    print("JD Sections Found:")
    for k,v in jd_sections.items():
        print(f"  [{k}] => {len(v)} chars")

    print("\n=== MATCH RESULTS ===")
    print(f"Document Embedding Similarity: {doc_sim:.4f}")
    print(f"TF-IDF Keyword Score:          {tfidf_sc:.4f}")
    print(f"Bullet (Resp) Score:           {bullet_sc_resp:.4f}")
    print(f"Bullet (Req) Score:            {bullet_sc_req:.4f}")
    print(f"Bullet Average:                {bullet_avg:.4f}")
    print(f"Section-level alignment:       {section_avg:.4f}")
    print(f"Final Combined Score:          {final_val:.4f}")

if __name__ == "__main__":
    main()
