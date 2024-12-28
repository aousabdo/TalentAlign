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
    "required", "must", "mandatory", "doctor", "residency", "board certification"
}

def extract_experience_years(text: str) -> int:
    """Extract years of experience from text, handling various formats."""
    import re
    
    # Common patterns
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:of\s+)?experience',  # "20+ years experience"
        r'experience\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)',  # "experience of 20 years"
        r'(?:over|more than)\s+(\d+)\s*(?:years?|yrs?)',  # "over 20 years"
        r'(\d+)(?:-|\s*to\s*)(\d+)\s*(?:years?|yrs?)',  # "15-20 years"
    ]
    
    max_years = 0
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            if len(match.groups()) == 2:  # Range pattern
                start, end = map(int, match.groups())
                max_years = max(max_years, end)
            else:
                years = int(match.group(1))
                max_years = max(max_years, years)
    
    # Also look for work history duration
    try:
        # Find dates like "2017 – Present" or "1999 - 2002"
        date_pattern = r'(\d{4})\s*[-–]\s*(Present|\d{4})'
        dates = re.findall(date_pattern, text)
        if dates:
            current_year = 2024  # Or use datetime.now().year
            total_years = 0
            for start, end in dates:
                end_year = current_year if end == 'Present' else int(end)
                total_years += end_year - int(start)
            max_years = max(max_years, total_years)
    except:
        pass
        
    return max_years

def check_experience_requirement(requirement: str, resume_text: str) -> bool:
    """Check if resume meets years of experience requirement."""
    import re
    
    # Extract required years
    req_years = extract_experience_years(requirement)
    if not req_years:
        return True  # No explicit requirement found
        
    # Extract experience from resume
    resume_years = extract_experience_years(resume_text)
    if not resume_years:
        return True  # Give benefit of doubt if no explicit mention
        
    logger.debug(f"Required years: {req_years}, Resume years: {resume_years}")
    return resume_years >= req_years

def check_hard_requirements(requirements_text: str, resume_text: str) -> bool:
    """Check both basic requirements and experience requirements."""
    req_lower = requirements_text.lower()
    res_lower = resume_text.lower()
    
    # Check basic hard requirements
    for marker in HARD_REQ_MARKERS:
        if marker in req_lower and marker not in res_lower:
            return False
            
    # Check experience requirements
    if not check_experience_requirement(req_lower, res_lower):
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
class ResumeJDMatcher:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.resume_classifier = DocumentClassifier(model=model, doc_type="resume")
        self.jd_classifier = DocumentClassifier(model=model, doc_type="jd")
        self.client = OpenAI()
        self.title_cache = {}
        self.text_cache = {}
        self.resume_data = None
        self.jd_data = None

    def keyword_match_score(self, resume_text: str, jd_keywords: List[str]) -> float:
        """Calculate keyword match score."""
        if not jd_keywords:
            return 0.0
        r_lower = resume_text.lower()
        matched = sum(kw.lower() in r_lower for kw in jd_keywords)
        return matched / len(jd_keywords)

    def get_llm_match_score(self, resume_data: Dict, jd_data: Dict) -> tuple[float, str]:
        """Use LLM to evaluate match and provide reasoning."""
        prompt = f"""
        You are an expert technical recruiter evaluating a job match. Consider transferable skills and experience.
        
        Job Requirements:
        - Title: {jd_data.get('job_title')}
        - Required Skills: {', '.join(jd_data.get('required_skills', []))}
        - Responsibilities: {', '.join(jd_data.get('responsibilities', []))}
        - Education: {jd_data.get('required_education_level')}
        - Years Required: {jd_data.get('years_of_experience_required')}

        Candidate Profile:
        - Current Title: {resume_data.get('current_or_most_recent_job_title')}
        - Skills: {', '.join(resume_data.get('skills', []))}
        - Years Experience: {resume_data.get('years_of_experience')}
        - Summary: {resume_data.get('summary')}

        Consider:
        1. Direct skill matches
        2. Transferable skills and experience
        3. Career level and leadership
        4. Technical depth and breadth
        5. Problem-solving abilities

        Important Notes:
        - Data Scientists often have strong programming/engineering skills
        - ML/AI skills transfer well to software development
        - Senior technical roles share leadership competencies
        - Database/Data modeling skills transfer across domains
        - Cloud/Big Data experience is valuable for all technical roles
        - System architecture skills are universal
        - API development is common across roles
        - Problem-solving and analytical skills are transferable

        Evaluate potential for success in this role, considering ALL transferable skills and experience.
        Be open-minded about career transitions between technical roles.

        Respond with JSON only:
        {{
            "match_score": <float 0-1>,
            "reasoning": "<brief explanation>",
            "transferable_skills": ["skill1", "skill2"],
            "skill_gaps": ["skill1", "skill2"],
            "strengths": ["strength1", "strength2"],
            "potential_success": <float 0-1>
        }}
        """

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical recruiter. Consider transferable skills and career transitions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"LLM Match Analysis: {result}")
            
            # Store full analysis for reporting
            self.last_analysis = result
            
            # Use average of match_score and potential_success
            final_score = (result["match_score"] + result.get("potential_success", result["match_score"])) / 2
            return final_score, result["reasoning"]
            
        except Exception as e:
            logger.error(f"Error getting LLM match score: {e}")
            return 0.7, "Error in LLM analysis, defaulting to neutral score"

    def final_score(self, doc_level: float, bullet_avg: float, section_avg: float) -> tuple[float, float]:
        """Calculate final score incorporating LLM analysis."""
        if doc_level >= 0.9999:  # Same document
            return 1.0, 1.0
            
        try:
            # Get LLM score
            llm_score, reasoning = self.get_llm_match_score(self.resume_data, self.jd_data)
            logger.debug(f"LLM Score: {llm_score}, Reason: {reasoning}")
            
            # Calculate traditional score with adjusted weights
            # More weight to bullet matching (most specific)
            doc_bullet = (doc_level * 0.2) + (bullet_avg * 0.8)
            
            # More weight to bullets/sections than doc-level
            traditional_score = (doc_bullet * 0.7) + (section_avg * 0.3)
            
            # Penalize low LLM scores more heavily
            if llm_score < 0.7:
                traditional_score *= (llm_score / 0.7)  # Scale down traditional score
            
            # Weight LLM score more heavily
            final = (traditional_score * 0.3) + (llm_score * 0.7)
            
            # Apply role-specific adjustments
            is_senior = any(word in str(self.resume_data.get('current_or_most_recent_job_title', '')).lower() 
                           for word in ['senior', 'principal', 'lead', 'chief', 'head'])
            
            # Bonus only for very strong matches
            if final > 0.8 and llm_score > 0.8:
                bonuses = 0.0
                
                # Title match bonus
                if self.resume_data.get('current_or_most_recent_job_title', '').lower() == self.jd_data.get('job_title', '').lower():
                    bonuses += 0.1
                    
                # Experience bonus
                try:
                    required_years = int(self.jd_data.get('years_of_experience_required', '0'))
                    actual_years = int(self.resume_data.get('years_of_experience', '0'))
                    if actual_years >= required_years * 1.5:
                        bonuses += 0.05
                except ValueError:
                    pass
                    
                # Skill match bonus
                if doc_level > 0.85 and bullet_avg > 0.8:
                    bonuses += 0.05
                    
                final = min(1.0, final + bonuses)
                
            return final, llm_score
            
        except Exception as e:
            logger.error(f"Error in LLM scoring: {e}")
            return traditional_score * 0.5, 0.0  # Penalize errors

    def check_compatibility(self, resume_path: str, jd_path: str) -> tuple[bool, str, float]:
        """Quick compatibility check using LLM."""
        # Get structured data
        self.resume_data = self.resume_classifier.process_file(resume_path)
        self.jd_data = self.jd_classifier.process_file(jd_path)
        
        logger.debug(f"Resume data: {self.resume_data}")
        logger.debug(f"JD data: {self.jd_data}")
        
        # Get LLM score first
        llm_score, reasoning = self.get_llm_match_score(self.resume_data, self.jd_data)
        logger.debug(f"Initial LLM Score: {llm_score}, Reason: {reasoning}")
        
        # Include details in reason
        reason = (f"Resume: {self.resume_data.get('current_or_most_recent_job_title', '')} "
                 f"({self.resume_data.get('years_of_experience', 'N/A')} years), "
                 f"JD: {self.jd_data.get('job_title', '')} "
                 f"({self.jd_data.get('years_of_experience_required', '')}+ years required)")
        
        # Use LLM score as compatibility indicator
        return llm_score >= 0.5, reason, llm_score

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with caching."""
        if pdf_path in self.text_cache:
            return self.text_cache[pdf_path]
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            self.text_cache[pdf_path] = text
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def check_requirements(self, resume_path: str, jd_path: str) -> bool:
        """Check all requirements using structured data."""
        if not hasattr(self, 'resume_data') or not hasattr(self, 'jd_data'):
            self.resume_data = self.resume_classifier.process_file(resume_path)
            self.jd_data = self.jd_classifier.process_file(jd_path)
            
        # Check education
        required_edu = self.jd_data.get('required_education_level', '').lower()
        resume_edu = ' '.join([
            self.resume_data.get('summary', ''),
            *[str(e) for e in self.resume_data.get('education', [])],
            self.resume_data.get('current_or_most_recent_job_title', '')
        ]).lower()
        
        # Check experience
        required_years = self.jd_data.get('years_of_experience_required', '')
        resume_years = self.resume_data.get('years_of_experience', '0')
        try:
            if required_years and int(resume_years) < int(required_years):
                logger.debug(f"Experience requirement not met: need {required_years}, has {resume_years}")
                return False
        except ValueError:
            pass  # Skip if can't parse years
            
        # Check skills with fuzzy matching
        required_skills = self.jd_data.get('required_skills', [])
        resume_skills = {s.lower() for s in self.resume_data.get('skills', [])}
        resume_text = ' '.join([
            self.resume_data.get('summary', ''),
            *[str(s) for s in self.resume_data.get('skills', [])],
            self.resume_data.get('current_or_most_recent_job_title', '')
        ]).lower()
        
        # Define skill equivalents
        skill_equivalents = {
            'python': {'py', 'python3', 'python programming'},
            'javascript': {'js', 'ecmascript', 'node.js'},
            'machine learning': {'ml', 'deep learning', 'ai', 'neural networks'},
            'statistics': {'statistical', 'regression', 'analytics'},
            'visualization': {'tableau', 'power bi', 'd3', 'charts', 'spotfire'},
            'sql': {'database', 'postgresql', 'mysql', 'vertica'},
            'communication': {'collaborative', 'interpersonal', 'leadership'},
            'problem-solving': {'analytical', 'analysis'},
            'tensorflow': {'deep learning', 'neural networks', 'ml frameworks'},
            'scikit-learn': {'sklearn', 'machine learning libraries'}
        }
        
        # Check each required skill
        missing_skills = []
        for skill in required_skills:
            skill_found = False
            skill_lower = str(skill).lower()
            
            # Direct match
            if skill_lower in resume_skills or skill_lower in resume_text:
                continue
                
            # Check equivalents
            for main_skill, alternatives in skill_equivalents.items():
                if (skill_lower in alternatives or 
                    main_skill in skill_lower or 
                    any(alt in skill_lower for alt in alternatives)):
                    if (any(alt in resume_skills for alt in alternatives) or
                        main_skill in resume_skills or
                        any(alt in resume_text for alt in alternatives)):
                        skill_found = True
                        break
                        
            if not skill_found:
                missing_skills.append(skill)
                
        # Allow missing up to 40% of required skills
        if len(missing_skills) > len(required_skills) * 0.4:
            logger.debug(f"Missing skills: {missing_skills}")
            return False
            
        return True

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
    parser.add_argument("--model", default="gpt-4o", help="Model for job classification")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.WARNING)

    matcher = ResumeJDMatcher(model=args.model)

    try:
        # Get structured data and text
        matcher.resume_data = matcher.resume_classifier.process_file(args.resume_pdf)
        matcher.jd_data = matcher.jd_classifier.process_file(args.jd_pdf)
        resume_text = matcher.extract_text(args.resume_pdf)
        jd_text = matcher.extract_text(args.jd_pdf)

        # Calculate all scores first
        client = matcher.client
        
        # Doc-level similarity
        texts_to_embed = [resume_text, jd_text]
        embeddings = batch_embed_all_texts(client, texts_to_embed)
        doc_sim = cosine_similarity(embeddings[resume_text], embeddings[jd_text])

        # TF-IDF matching
        jd_keywords = extract_top_keywords_from_text(jd_text, TOP_N_KEYWORDS)
        tfidf_sc = matcher.keyword_match_score(resume_text, jd_keywords)
        doc_level = ALPHA * doc_sim + (1 - ALPHA) * tfidf_sc

        # Bullet matching
        resp_lines, req_lines = parse_bullet_sections_for_jd(jd_text)
        bullet_sc_resp = calculate_bullet_score(resp_lines, resume_text, client)
        bullet_sc_req = calculate_bullet_score(req_lines, resume_text, client)
        bullet_avg = (bullet_sc_resp + bullet_sc_req) / 2.0

        # Section matching
        resume_sections = parse_resume_sections(resume_text)
        jd_sections = parse_jd_sections(jd_text)
        section_avg = match_jd_sections_with_resume(client, jd_sections, resume_sections)

        # Get LLM analysis
        llm_score, reasoning = matcher.get_llm_match_score(matcher.resume_data, matcher.jd_data)
        
        # Show all scores
        print("\n=== MATCH SCORES ===")
        print(f"Document Similarity:     {doc_sim:.4f}")
        print(f"TF-IDF Score:           {tfidf_sc:.4f}")
        print(f"Bullet (Resp) Score:    {bullet_sc_resp:.4f}")
        print(f"Bullet (Req) Score:     {bullet_sc_req:.4f}")
        print(f"Bullet Average:         {bullet_avg:.4f}")
        print(f"Section Alignment:      {section_avg:.4f}")
        print(f"LLM Match Score:        {llm_score:.4f}")

        # Show LLM analysis
        print("\n=== LLM ANALYSIS ===")
        print(f"Match Score: {llm_score:.4f}")
        print(f"Reasoning: {reasoning}")
        
        if hasattr(matcher, 'last_analysis'):
            print("\nTransferable Skills:")
            for skill in matcher.last_analysis.get('transferable_skills', []):
                print(f"- {skill}")
            print("\nSkill Gaps:")
            for gap in matcher.last_analysis.get('skill_gaps', []):
                print(f"- {gap}")
            print("\nStrengths:")
            for strength in matcher.last_analysis.get('strengths', []):
                print(f"- {strength}")

        # Calculate final score
        final_val, _ = matcher.final_score(doc_level, bullet_avg, section_avg)
        
        # Scale final score based on LLM score
        if llm_score < 0.5:
            print("\nLLM indicates low match potential")
            # Scale down the score based on LLM assessment
            final_val *= (llm_score / 0.5)  # Linear scaling
            
        print(f"\nFinal Combined Score: {final_val:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        return

def calculate_bullet_score(bullet_lines: List[str], resume_text: str, client: OpenAI) -> float:
    """Calculate similarity score between bullet points and resume."""
    if not bullet_lines:
        return 0.0
        
    # Embed resume text once
    resume_emb = embed_text(client, resume_text)
    if resume_emb is None:
        return 0.0
        
    # Embed and compare each bullet
    bullet_sims = []
    for line in bullet_lines:
        if not line.strip():
            continue
        bullet_emb = embed_text(client, line)
        if bullet_emb is None:
            continue
        sim = cosine_similarity(bullet_emb, resume_emb)
        bullet_sims.append(sim)
        
    return sum(bullet_sims)/len(bullet_sims) if bullet_sims else 0.0

if __name__ == "__main__":
    main()
