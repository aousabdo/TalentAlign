import sys
import re
import string
import numpy as np
import PyPDF2
import concurrent.futures
import os
import hashlib
import json
#from tqdm import tqdm  # Unused
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import argparse
import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from openai import OpenAI, APIConnectionError, APIStatusError
from document_classifier import DocumentClassifier

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

###############################################################################
# GLOBAL PARAMETERS
###############################################################################
ALPHA = 0.3
BULLET_WEIGHT = 0.5
SECTION_WEIGHT = 0.5
CLAMP_THRESHOLD = 0.7
TOP_N_KEYWORDS = 10
BATCH_SIZE = 20
CACHE_DIR = ".cache"

###############################################################################
# (1) Simple Domain Detection (FAST: just string checks)
###############################################################################
def determine_domain(text: str, preview_length: int = 1000) -> str:
    """Optimized domain detection using shorter preview."""
    text_preview = text[:preview_length].lower()
    domain_keywords = {
        "medical": {
            "md", "doctor", "surgery", "patient", "clinical", "medical", "healthcare"
        },
        "data_science": {
            "machine learning", "data scientist", "analytics", "python", "sql", 
            "data analysis", "statistical", "deep learning"
        },
        "management": {
            "product manager", "stakeholders", "business case", "product management",
            "roadmap", "product owner", "market research", "product strategy",
            "customer requirements", "product development"
        },
        "engineering": {
            "software engineer", "full stack", "frontend", "backend", "java",
            "javascript", "react", "node", "api", "web development"
        }
    }
    
    best_domain = "unknown"
    best_count = 0
    for dom, keywords in domain_keywords.items():
        # Weight longer phrases more heavily
        count = sum(2 if len(kw.split()) > 1 and kw in text_preview else 
                   1 if kw in text_preview else 0 
                   for kw in keywords)
        if count > best_count:
            best_domain = dom
            best_count = count
    return best_domain if best_count > 1 else "unknown"  # Require at least 2 matches

###############################################################################
# Caching
###############################################################################
def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]

_cache = {}

def get_cached_embedding(text: str, cache_dir: str = CACHE_DIR) -> Optional[List[float]]:
    key = get_cache_key(text)
    if key in _cache:
        return _cache[key]
    cache_file = Path(cache_dir) / f"{key}.json"
    if cache_file.exists():
        try:
            emb = json.loads(cache_file.read_text())
            _cache[key] = emb
            return emb
        except:
            return None
    return None

def save_embedding_cache(text: str, embedding: List[float], cache_dir: str = CACHE_DIR) -> None:
    key = get_cache_key(text)
    _cache[key] = embedding
    try:
        Path(cache_dir).mkdir(exist_ok=True)
        cache_file = Path(cache_dir) / f"{key}.json"
        if not cache_file.exists():
            cache_file.write_text(json.dumps(embedding))
    except:
        pass

###############################################################################
# Embedding Utilities
###############################################################################
def embed_text(client: OpenAI, text: str) -> Optional[List[float]]:
    if not text.strip():
        return None
    try:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        return resp.data[0].embedding
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error: {str(e)}")
        return None

def batch_embed_all_texts(client: OpenAI, texts: List[str], batch_size: int = BATCH_SIZE) -> Dict[str, List[float]]:
    text_to_key = {txt: get_cache_key(txt) for txt in texts if txt.strip()}
    results = {}
    texts_to_embed = []
    for txt, key in text_to_key.items():
        emb = get_cached_embedding(txt)
        if emb is not None:
            results[txt] = emb
        else:
            texts_to_embed.append(txt)
    if texts_to_embed:
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                resp = client.embeddings.create(model="text-embedding-ada-002", input=batch)
                for t, emb_data in zip(batch, resp.data):
                    emb = emb_data.embedding
                    results[t] = emb
                    save_embedding_cache(t, emb)
            except Exception as e:
                logger.warning(f"Error in batch {i}: {e}")
                for t in batch:
                    results[t] = None
    return results

###############################################################################
# PDF
###############################################################################
def extract_text_from_pdf(pdf_path: str) -> str:
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
        low = line.lower()
        matched_heading = None
        for hd in headings:
            if hd in low and low.index(hd) < 5:
                matched_heading = hd.title()
                break
        if matched_heading:
            current_section = matched_heading
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    for sec in sections:
        sections[sec] = "\n".join(sections[sec]).strip()
    return sections

def parse_jd_sections(jd_text: str) -> Dict[str, str]:
    known_headings = {
        "responsibilities", "requirements", "preferred", "benefits",
        "how to apply", "job description", "about us", "education", "notes"
    }
    lines = jd_text.splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    sections = {}
    current_section = "Misc"
    sections[current_section] = []

    for line in lines:
        low = line.lower()
        matched = None
        for hd in known_headings:
            if low.startswith(hd):
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
# Similarity
###############################################################################
def batch_cosine_similarity(embeddings_a: List[List[float]], embeddings_b: List[List[float]]) -> np.ndarray:
    if not embeddings_a or not embeddings_b:
        return np.array([[0.0]])
    A = np.array(embeddings_a)
    B = np.array(embeddings_b)
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    if len(B.shape) == 1:
        B = B.reshape(1, -1)
    dot_product = np.dot(A, B.T)
    norms_a = np.linalg.norm(A, axis=1)
    norms_b = np.linalg.norm(B, axis=1)
    mask = (norms_a[:, np.newaxis] * norms_b) != 0
    sims = np.zeros_like(dot_product)
    sims[mask] = dot_product[mask] / (norms_a[:, np.newaxis] * norms_b)[mask]
    return sims

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return float(batch_cosine_similarity([a], [b])[0][0])

###############################################################################
# TF-IDF
###############################################################################
def precompute_tfidf(texts: List[str]) -> tuple:
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
# Requirements
###############################################################################
HARD_REQ_MARKERS = {
    "required", "must", "mandatory", "doctor", "residency", "board certification",
    "10+ years",
}

def check_hard_requirements(requirements_text: str, resume_text: str) -> bool:
    req_lower = requirements_text.lower()
    res_lower = resume_text.lower()
    for marker in HARD_REQ_MARKERS:
        if marker in req_lower and marker not in res_lower:
            return False
    return True

###############################################################################
# Bullet
###############################################################################
def parse_bullet_sections_for_jd(jd_text: str) -> tuple:
    """Parse sections with resume-aware handling."""
    sections = parse_jd_sections(jd_text)
    resp_text = sections.get("Responsibilities", sections.get("Professional Experience", ""))
    req_text = sections.get("Requirements", sections.get("Skills", ""))

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
def match_jd_sections_with_resume(client: OpenAI,
    jd_sections: Dict[str, str],
    resume_sections: Dict[str, str]
) -> float:
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
        for jd_sec_name, jd_sec_text in jd_sections.items():
            future = executor.submit(embed_text, client, jd_sec_text)
            future_jd[future] = jd_sec_name

        jd_embs = {}
        for fut in concurrent.futures.as_completed(future_jd):
            j_name = future_jd[fut]
            jd_embs[j_name] = fut.result()

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
    return 1.0

###############################################################################
# Score Combination
###############################################################################
def final_score(doc_level: float, bullet_avg: float, section_avg: float) -> float:
    # If doc_level is perfect (1.0), it's likely same document
    if doc_level >= 0.9999:
        return 1.0
    
    doc_bullet = (doc_level ** (1-BULLET_WEIGHT)) * (bullet_avg ** BULLET_WEIGHT)
    final_val = (doc_bullet ** (1-SECTION_WEIGHT)) * (section_avg ** SECTION_WEIGHT)
    return final_val

def keyword_match_score(resume_text: str, jd_keywords: List[str]) -> float:
    if not jd_keywords:
        return 0.0
    r_lower = resume_text.lower()
    matched = sum(kw.lower() in r_lower for kw in jd_keywords)
    return matched / len(jd_keywords)

###############################################################################
# (2) Domain Check (Minimal Overhead)
###############################################################################
class ResumeJDMatcher:
    def __init__(self, model: str = "llama3.2"):
        self.classifier = DocumentClassifier(model=model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_cache = {}
        self.title_cache = {}
        self.domain_cache = {}  # Cache domain results

    def get_cached_domain(self, text: str) -> Optional[str]:
        """Get cached domain."""
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
        return self.domain_cache.get(text_hash)

    def set_cached_domain(self, text: str, domain: str):
        """Cache domain classification."""
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
        self.domain_cache[text_hash] = domain

    def extract_text(self, pdf_path: str, max_chars: Optional[int] = None) -> str:
        """Extract text with optional length limit and caching."""
        cache_key = f"{pdf_path}:{max_chars}"
        if cache_key in self.text_cache:
            return self.text_cache[cache_key]

        try:
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                    if max_chars and len(text) >= max_chars:
                        text = text[:max_chars]
                        break
                self.text_cache[cache_key] = text
                return text
        except Exception as e:
            raise Exception(f"Error extracting text: {e}")

    def check_compatibility(self, resume_path: str, jd_path: str) -> tuple[bool, str, float]:
        """Quick compatibility check using domain, title, and field."""
        # If comparing same file, return perfect match
        if resume_path == jd_path:
            return True, "Same document", 1.0
            
        # Get classifications (using cache)
        resume_class = self.classifier.process_file(resume_path)
        jd_class = self.classifier.process_file(jd_path)
        
        # Add debug logging
        logger.debug(f"Resume classification: {resume_class}")
        logger.debug(f"JD classification: {jd_class}")
        
        # Calculate similarity score based on title and field
        title_match = resume_class['title'] == jd_class['title']
        field_match = resume_class['field'] == jd_class['field']
        
        compatibility_score = 1.0 if title_match else 0.6 if field_match else 0.0
        
        reason = (f"Resume: {resume_class['title']} ({resume_class['field']}), "
                 f"JD: {jd_class['title']} ({jd_class['field']})")
        
        return (title_match or field_match), reason, compatibility_score

    def get_cached_title(self, text: str) -> Optional[str]:
        """Get cached title or None."""
        text_hash = hashlib.md5(text[:2500].encode()).hexdigest()
        return self.title_cache.get(text_hash)
        
    def set_cached_title(self, text: str, title: str):
        """Cache a title classification."""
        text_hash = hashlib.md5(text[:2500].encode()).hexdigest()
        self.title_cache[text_hash] = title

    def quick_title_check(self, resume_text: str, jd_text: str) -> tuple[bool, str]:
        """Quick check of job titles with caching."""
        # Check cache first
        resume_title = self.get_cached_title(resume_text)
        jd_title = self.get_cached_title(jd_text)
        
        # Classify if not cached
        if not resume_title:
            resume_title = self.classifier.classify_document(resume_text[:2500])
            self.set_cached_title(resume_text, resume_title)
            
        if not jd_title:
            jd_title = self.classifier.classify_document(jd_text[:2500])
            self.set_cached_title(jd_text, jd_title)

        # Define related job families
        job_families = {
            "tech": {"Software Engineer", "Full Stack Developer", "Frontend Developer", 
                    "Backend Developer", "DevOps Engineer", "Java Developer"},
            "data": {"Data Scientist", "Data Analyst", "Machine Learning Engineer", 
                    "Data Engineer", "Analytics Engineer"},
            "product": {"Product Manager", "Product Owner", "Program Manager", 
                       "Project Manager", "Business Analyst"}
        }
        
        # Find job families for both titles
        resume_family = next((family for family, titles in job_families.items() 
                            if any(t.lower() in resume_title.lower() for t in titles)), None)
        jd_family = next((family for family, titles in job_families.items() 
                         if any(t.lower() in jd_title.lower() for t in titles)), None)
        
        # Consider match if in same job family or exact match
        title_match = (resume_title == jd_title) or (resume_family and resume_family == jd_family)
        reason = f"Resume Title: {resume_title}, JD Title: {jd_title}"
        
        return title_match, reason

def domain_check(resume_text: str, jd_text: str, matcher: ResumeJDMatcher) -> tuple[bool, str]:
    """Enhanced domain check using both simple keywords and LLM classification."""
    # Get basic domain check
    basic_match = determine_domain(resume_text) == determine_domain(jd_text)
    
    # Get LLM-based job title match
    resume_title, jd_title, title_match = matcher.quick_title_check(resume_text, jd_text)
    
    reason = f"Resume Title: {resume_title}, JD Title: {jd_title}"
    
    # If either method suggests a match, allow it
    return (basic_match or title_match), reason

###############################################################################
# Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Resume-JD Matcher")
    parser.add_argument("resume_pdf", help="Path to resume PDF file")
    parser.add_argument("jd_pdf", help="Path to job description PDF file")
    parser.add_argument(
        "--model", 
        default="gpt-4o",
        help="Model for job classification"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.WARNING)

    matcher = ResumeJDMatcher(model=args.model)

    try:
        # Quick compatibility check with score
        is_compatible, reason, compat_score = matcher.check_compatibility(args.resume_pdf, args.jd_pdf)
        if not is_compatible:
            print(f"\n{reason}")
            print("Final Combined Score: 0.0000")
            return

        # Only extract full text if compatible
        resume_text = matcher.extract_text(args.resume_pdf)
        jd_text = matcher.extract_text(args.jd_pdf)

        # Continue with detailed matching...

    except Exception as e:
        print(f"Error in preliminary checks: {e}")
        return

    client = matcher.client

    jd_sections = parse_jd_sections(jd_text)
    req_text = jd_sections.get("Requirements", "")
    if not check_hard_requirements(req_text, resume_text):
        print("\nResume fails must-have/hard requirements!")
        print("Final Combined Score: 0.0000")
        return

    # Gather texts to embed
    texts_to_embed = [resume_text, jd_text]
    resp_lines, req_lines = parse_bullet_sections_for_jd(jd_text)
    texts_to_embed.extend(ln for ln in resp_lines if ln.strip())
    texts_to_embed.extend(ln for ln in req_lines if ln.strip())

    resume_sections = parse_resume_sections(resume_text)
    jd_vals = parse_jd_sections(jd_text)
    for val in resume_sections.values():
        if val.strip():
            texts_to_embed.append(val)
    for val in jd_vals.values():
        if val.strip():
            texts_to_embed.append(val)

    # Batch
    try:
        embeddings = batch_embed_all_texts(client, texts_to_embed)
    except (APIConnectionError, APIStatusError) as e:
        print(f"OpenAI API error: {str(e)}")
        sys.exit(1)

    # Doc-level
    r_emb = embeddings[resume_text]
    j_emb = embeddings[jd_text]
    doc_sim = cosine_similarity(r_emb, j_emb)
    if CLAMP_THRESHOLD and doc_sim < CLAMP_THRESHOLD:
        doc_sim = 0.0

    # TF-IDF
    jd_keywords = extract_top_keywords_from_text(jd_text, TOP_N_KEYWORDS)
    tfidf_sc = keyword_match_score(resume_text, jd_keywords)
    doc_level = ALPHA * doc_sim + (1 - ALPHA) * tfidf_sc

    # Bullet
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

    # Section-level
    section_sims = []
    for jd_sec_text in jd_vals.values():
        if not jd_sec_text.strip() or jd_sec_text not in embeddings:
            continue
        jd_emb = embeddings[jd_sec_text]
        if jd_emb is None:
            continue
        best_score = 0.0
        for r_sec_text in resume_sections.values():
            if not r_sec_text.strip() or r_sec_text not in embeddings:
                continue
            r_semb = embeddings[r_sec_text]
            if r_semb is None:
                continue
            score = cosine_similarity(jd_emb, r_semb)
            if score > best_score:
                best_score = score
        section_sims.append(best_score)
    section_avg = sum(section_sims)/len(section_sims) if section_sims else 1.0

    # Final
    final_val = final_score(doc_level, bullet_avg, section_avg)

    # Apply compatibility score to final score
    final_val = final_val * compat_score  # Reduce score based on title/field match

    print("\n=== SECTION-LEVEL PARSING RESULTS ===")
    print("Resume Sections Found:")
    for k,v in resume_sections.items():
        print(f"  [{k}] => {len(v)} chars")

    print("\nJD Sections Found:")
    for k,v in jd_vals.items():
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
