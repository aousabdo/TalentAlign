import sys
import re
import string
import numpy as np
import PyPDF2
import concurrent.futures
import os
import hashlib
import json
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from openai import OpenAI, APIConnectionError, APIStatusError

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

###############################################################################
# GLOBAL PARAMETERS
###############################################################################
ALPHA = 0.3            # Weight of doc-level embedding vs. TF-IDF
BULLET_WEIGHT = 0.5    # Weight for bullet-based lines
SECTION_WEIGHT = 0.5   # Weight for section-level alignment
CLAMP_THRESHOLD = 0.7  # If doc-sim < 0.7 => clamp to 0.0
TOP_N_KEYWORDS = 10
BATCH_SIZE = 20        # Number of texts to batch for embeddings
CACHE_DIR = ".cache"   # Directory for caching embeddings

###############################################################################
# Caching Utilities
###############################################################################
def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]  # Use shorter hash

_cache = {}  # Memory cache

def get_cached_embedding(text: str, cache_dir: str = CACHE_DIR) -> Optional[List[float]]:
    key = get_cache_key(text)
    
    # Check memory cache first
    if key in _cache:
        return _cache[key]
    
    # Check disk cache
    cache_file = Path(cache_dir) / f"{key}.json"
    if cache_file.exists():
        try:
            emb = json.loads(cache_file.read_text())
            _cache[key] = emb  # Store in memory
            return emb
        except:
            return None
    return None

def save_embedding_cache(text: str, embedding: List[float], cache_dir: str = CACHE_DIR) -> None:
    key = get_cache_key(text)
    _cache[key] = embedding  # Store in memory
    
    # Store on disk
    try:
        Path(cache_dir).mkdir(exist_ok=True)
        cache_file = Path(cache_dir) / f"{key}.json"
        if not cache_file.exists():  # Only write if doesn't exist
            cache_file.write_text(json.dumps(embedding))
    except:
        pass  # Ignore cache write errors

###############################################################################
# Embedding Utilities
###############################################################################
def embed_text(client: OpenAI, text: str) -> Optional[List[float]]:
    """Single text embedding with error handling"""
    if not text.strip():
        return None
    try:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        return resp.data[0].embedding
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error: {str(e)}")
        return None

def batch_embed_texts(client: OpenAI, texts: List[str], batch_size: int = BATCH_SIZE) -> List[Optional[List[float]]]:
    """Batch process embeddings with minimal overhead"""
    embeddings = []
    valid_texts = [(i, text.strip()) for i, text in enumerate(texts) if text.strip()]
    if not valid_texts:
        return []
    
    indices, texts_to_process = zip(*valid_texts)
    
    # Process in batches without progress bar for small batches
    for i in range(0, len(texts_to_process), batch_size):
        batch = texts_to_process[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [data.embedding for data in resp.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.warning(f"Error in batch {i}: {e}")
            embeddings.extend([None] * len(batch))
    
    result = [None] * len(texts)
    for idx, emb in zip(indices, embeddings):
        result[idx] = emb
    return result

def batch_embed_all_texts(client: OpenAI, texts: List[str], batch_size: int = BATCH_SIZE) -> Dict[str, List[float]]:
    """Batch process all texts at once to minimize API calls"""
    # Create cache key for each text
    text_to_key = {text: get_cache_key(text) for text in texts if text.strip()}
    key_to_text = {k: t for t, k in text_to_key.items()}
    
    # Check cache first
    results = {}
    texts_to_embed = []
    keys_to_embed = []
    
    for text, key in text_to_key.items():
        emb = get_cached_embedding(text)
        if emb is not None:
            results[text] = emb
        else:
            texts_to_embed.append(text)
            keys_to_embed.append(key)
    
    # Batch embed remaining texts
    if texts_to_embed:
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                resp = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                for text, emb_data in zip(batch, resp.data):
                    emb = emb_data.embedding
                    results[text] = emb
                    save_embedding_cache(text, emb)
            except Exception as e:
                logger.warning(f"Error in batch {i}: {e}")
                for text in batch:
                    results[text] = None
    
    return results

###############################################################################
# PDF Processing
###############################################################################
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF exactly as in original"""
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
# Section Parsing
###############################################################################
def parse_resume_sections(resume_text: str) -> Dict[str, str]:
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

def parse_jd_sections(jd_text: str) -> Dict[str, str]:
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

    # Convert lists to strings
    for sec in sections:
        sections[sec] = "\n".join(sections[sec]).strip()

    return sections

###############################################################################
# Similarity Calculations
###############################################################################
def batch_cosine_similarity(embeddings_a: List[List[float]], embeddings_b: List[List[float]]) -> np.ndarray:
    """Vectorized similarity calculation for multiple embeddings"""
    if not embeddings_a or not embeddings_b:
        return np.array([[0.0]])
        
    # Convert to 2D numpy arrays
    A = np.array(embeddings_a)
    B = np.array(embeddings_b)
    
    # Ensure 2D
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    if len(B.shape) == 1:
        B = B.reshape(1, -1)
    
    # Compute dot product and norms
    dot_product = np.dot(A, B.T)
    norms_a = np.linalg.norm(A, axis=1)
    norms_b = np.linalg.norm(B, axis=1)
    
    # Handle zero norms
    mask = (norms_a[:, np.newaxis] * norms_b) != 0
    similarities = np.zeros_like(dot_product)
    similarities[mask] = dot_product[mask] / (norms_a[:, np.newaxis] * norms_b)[mask]
    
    return similarities

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Single vector cosine similarity"""
    if not a or not b:
        return 0.0
    return float(batch_cosine_similarity([a], [b])[0][0])

###############################################################################
# TF-IDF Processing
###############################################################################
def precompute_tfidf(texts: List[str]) -> tuple:
    """Precompute TF-IDF for multiple texts at once"""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,1))
    matrix = vec.fit_transform(texts)
    features = vec.get_feature_names_out()
    return vec, matrix, features

def extract_top_keywords_from_text(text: str, top_n: int = TOP_N_KEYWORDS) -> List[str]:
    if not text.strip():
        return []
    vec, matrix, features = precompute_tfidf([text])
    scores = matrix.toarray()[0]
    pairs = list(zip(features, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _s in pairs[:top_n]]

###############################################################################
# Requirements Processing
###############################################################################
HARD_REQ_MARKERS = {
    "required", "must", "mandatory", "doctor", "residency", "board certification",
    "10+ years",
}

def check_hard_requirements(requirements_text: str, resume_text: str) -> bool:
    """Exact match with original hard requirements check"""
    req_lower = requirements_text.lower()
    res_lower = resume_text.lower()
    
    for marker in HARD_REQ_MARKERS:
        if marker in req_lower and marker not in res_lower:
            return False
    return True

###############################################################################
# Bullet Processing
###############################################################################
def parse_bullet_sections_for_jd(jd_text: str) -> tuple:
    sections = parse_jd_sections(jd_text)
    resp_text = sections.get("Responsibilities", "")
    req_text = sections.get("Requirements", "")

    resp_lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
    req_lines = [ln.strip() for ln in req_text.splitlines() if ln.strip()]

    return resp_lines, req_lines

def average_bullet_similarity(client: OpenAI, lines: List[str], resume_emb: List[float]) -> float:
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
# Section Matching
###############################################################################
def match_jd_sections_with_resume(
    client: OpenAI,
    jd_sections: Dict[str, str],
    resume_sections: Dict[str, str]
) -> float:
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
# Score Combination
###############################################################################
def final_score(doc_level: float, bullet_avg: float, section_avg: float) -> float:
    doc_bullet = (doc_level ** (1-BULLET_WEIGHT)) * (bullet_avg ** BULLET_WEIGHT)
    final_val = (doc_bullet ** (1-SECTION_WEIGHT)) * (section_avg ** SECTION_WEIGHT)
    return final_val

###############################################################################
# Main Function
###############################################################################
def is_valid_text(text: str, min_length: int = 100) -> bool:
    """Check if extracted text is valid and meaningful"""
    if not text or len(text) < min_length:
        return False
    
    # Check if text contains actual words (not just garbage)
    words = text.split()
    if len(words) < 10:  # At least 10 words
        return False
    
    # Check if text contains common resume/job words
    common_words = {'experience', 'skills', 'education', 'work', 'job', 'the', 'and', 'for'}
    text_lower = text.lower()
    if not any(word in text_lower for word in common_words):
        return False
    
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python match_resume_jd.py <resume.pdf> <jd.pdf>")
        sys.exit(1)

    resume_pdf = sys.argv[1]
    jd_pdf = sys.argv[2]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1) Extract text
    resume_text = extract_text_from_pdf(resume_pdf)
    jd_text = extract_text_from_pdf(jd_pdf)

    # 2) Parse JD sections
    jd_sections = parse_jd_sections(jd_text)
    req_text = jd_sections.get("Requirements", "")
    
    # Must-have check
    if not check_hard_requirements(req_text, resume_text):
        print("\nResume fails must-have/hard requirements!")
        print("Final Combined Score: 0.0000")
        return

    # 3) Get all texts that need embeddings
    texts_to_embed = [resume_text, jd_text]  # Doc-level texts
    
    # Add bullet points
    resp_lines, req_lines = parse_bullet_sections_for_jd(jd_text)
    texts_to_embed.extend(ln for ln in resp_lines if ln.strip())
    texts_to_embed.extend(ln for ln in req_lines if ln.strip())
    
    # Add sections
    resume_sections = parse_resume_sections(resume_text)
    texts_to_embed.extend(text for text in resume_sections.values() if text.strip())
    texts_to_embed.extend(text for text in jd_sections.values() if text.strip())
    
    # Batch embed all texts at once
    try:
        embeddings = batch_embed_all_texts(client, texts_to_embed)
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error: {str(e)}")
        sys.exit(1)
    
    # Get doc-level embeddings
    r_emb = embeddings[resume_text]
    j_emb = embeddings[jd_text]
    
    # Calculate doc similarity
    doc_sim = cosine_similarity(r_emb, j_emb)
    if CLAMP_THRESHOLD and doc_sim < CLAMP_THRESHOLD:
        doc_sim = 0.0

    # 4) TF-IDF
    jd_keywords = extract_top_keywords_from_text(jd_text, TOP_N_KEYWORDS)
    tfidf_sc = keyword_match_score(resume_text, jd_keywords)

    doc_level = ALPHA*doc_sim + (1-ALPHA)*tfidf_sc

    # 5) Calculate bullet similarities using cached embeddings
    bullet_sims_resp = []
    for line in resp_lines:
        if line.strip() and line in embeddings and embeddings[line] is not None:
            sim = cosine_similarity(embeddings[line], r_emb)
            bullet_sims_resp.append(sim)
    
    bullet_sims_req = []
    for line in req_lines:
        if line.strip() and line in embeddings and embeddings[line] is not None:
            sim = cosine_similarity(embeddings[line], r_emb)
            bullet_sims_req.append(sim)
    
    bullet_sc_resp = sum(bullet_sims_resp)/len(bullet_sims_resp) if bullet_sims_resp else 0.0
    bullet_sc_req = sum(bullet_sims_req)/len(bullet_sims_req) if bullet_sims_req else 0.0
    bullet_avg = (bullet_sc_resp + bullet_sc_req)/2.0

    # 6) Calculate section similarities using cached embeddings
    section_sims = []
    for jd_sec_name, jd_sec_text in jd_sections.items():
        if not jd_sec_text.strip() or jd_sec_text not in embeddings:
            continue
        jd_emb = embeddings[jd_sec_text]
        if jd_emb is None:
            continue
            
        best_score = 0.0
        for r_sec_text in resume_sections.values():
            if not r_sec_text.strip() or r_sec_text not in embeddings:
                continue
            r_emb = embeddings[r_sec_text]
            if r_emb is None:
                continue
            score = cosine_similarity(jd_emb, r_emb)
            if score > best_score:
                best_score = score
        section_sims.append(best_score)
    
    section_avg = sum(section_sims)/len(section_sims) if section_sims else 1.0

    # 7) final
    final_val = final_score(doc_level, bullet_avg, section_avg)

    # Print results
    print("\n=== SECTION-LEVEL PARSING RESULTS ===")
    print("Resume Sections Found:")
    for k,v in resume_sections.items():
        print(f"  [{k}] => {len(v)} chars")

    print("\nJD Sections Found:")
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

def keyword_match_score(resume_text: str, jd_keywords: List[str]) -> float:
    """Calculate keyword match score between resume and job description keywords"""
    if not jd_keywords:
        return 0.0
    r_lower = resume_text.lower()
    matched = sum(kw.lower() in r_lower for kw in jd_keywords)
    return matched / len(jd_keywords)

if __name__ == "__main__":
    main() 