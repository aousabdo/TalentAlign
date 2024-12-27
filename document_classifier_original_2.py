import fitz  
import json
import os
import argparse
import ollama
import openai
import google.generativeai as genai
from difflib import get_close_matches
from typing import List, Dict, Optional
import concurrent.futures
import logging
import warnings
from absl import logging as absl_logging
from pathlib import Path
import hashlib
import tempfile

# Configure logging with environment variable control
log_level = os.getenv('TALENTALIGN_LOG_LEVEL', 'WARNING')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress other loggers by default
logging.getLogger('httpx').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)
absl_logging.set_verbosity(absl_logging.ERROR)
logging.getLogger('googleapiclient').setLevel(logging.ERROR)

class DocumentClassifier:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

        # Load or define job titles
        self.job_titles = self.load_job_categories()

        # Define or load fields
        self.fields = self.load_fields()

        # A small cache to store text from PDFs so we don't re-parse
        self.text_cache: Dict[str, str] = {}

        # Setup cache directory in current working directory
        self.cache_dir = Path(".cache/classifications")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")

        # Initialize Gemini if needed
        if self.model.startswith('gemini-'):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )

    def load_job_categories(self) -> List[str]:
        """Load job categories from JSON file with fallback."""
        try:
            json_path = os.path.join(os.path.dirname(__file__), 'job_categories.json')
            with open(json_path, 'r') as file:
                categories = json.load(file)
            # Flatten the nested structure
            return [
                title 
                for category in categories.values() 
                for subcategory in category.values() 
                for title in subcategory
            ]
        except Exception as e:
            print(f"Warning: Using fallback job titles. Error: {e}")
            return [
                "Data Scientist", "Software Engineer", "Doctor", "Nurse",
                "Teacher", "Lawyer", "Accountant", "Product Manager"
            ]

    def load_fields(self) -> List[str]:
        """Load or define fields (e.g., IT, Hospitality, Accounting, etc.)."""
        # If you have a separate file, load from it. Otherwise, define them here:
        return [
            "IT", "Hospitality", "Accounting", "HR", "Healthcare", 
            "Education", "Finance", "Marketing", "Engineering", "Operations",
            "Sales", "Business", "Consulting", "Design", "Legal", 
            "Public Relations", "Project Management", "Construction", 
            "Customer Service"
        ]

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 3) -> str:
        """Extract text from first few pages of a PDF (with caching)."""
        if pdf_path in self.text_cache:
            return self.text_cache[pdf_path]
        
        try:
            with fitz.open(pdf_path) as doc:
                # Only extract first few pages to speed things up
                text = "\n".join(
                    page.get_text() 
                    for page in doc.pages(0, min(max_pages, len(doc)))
                )
                self.text_cache[pdf_path] = text
                return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def validate_label(self, label: str, valid_list: List[str]) -> str:
        """
        Validate and correct a label (job title or field) using fuzzy matching.
        Returns either the exact match or the closest match in valid_list.
        """
        if label in valid_list:
            return label
            
        matches = get_close_matches(
            label.lower(),
            [x.lower() for x in valid_list],
            n=1,
            cutoff=0.6
        )
        if matches:
            # Return the correctly capitalized version from the valid_list
            for x in valid_list:
                if x.lower() == matches[0]:
                    return x
        return label  # If no close match, return as is (or handle differently)

    def classify_document(self, text: str) -> Dict[str, str]:
        """Classify the document to find both job title and field."""
        max_length = 2500
        if len(text) > max_length:
            text = f"{text[:max_length//2]}\n...\n{text[-max_length//2:]}"
        
        prompt = f"""Classify this document into exactly one job title and one field.

Available Job Titles:
{', '.join(self.job_titles)}

Available Fields:
{', '.join(self.fields)}

Document:
{text}

RESPOND WITH VALID JSON ONLY, using this exact format:
{{"title": "EXACT_JOB_TITLE", "field": "EXACT_FIELD"}}

Choose only from the provided lists. No explanations or additional text."""

        try:
            if self.model.startswith(('gpt-', 'text-')):
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise job classifier. Respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,
                    response_format={ "type": "json_object" }  # Force JSON response
                )
                content = response.choices[0].message.content.strip()

            elif self.model.startswith('gemini-'):
                chat = self.gemini_model.start_chat(
                    history=[{
                        "role": "user",
                        "parts": ["You are a precise job classifier. Respond with valid JSON only."]
                    }]
                )
                response = chat.send_message(prompt)
                content = response.text.strip()

            else:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise job classifier. Respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={"temperature": 0}
                )
                content = response['message']['content'].strip()

            # Clean the response if needed
            content = content.strip('`').strip()  # Remove any markdown code blocks
            if content.startswith('json'):  # Remove any language identifier
                content = content[4:].strip()

            try:
                parsed = json.loads(content)
                raw_title = str(parsed.get("title", "")).strip()
                raw_field = str(parsed.get("field", "")).strip()

                if not raw_title or not raw_field:
                    return {
                        "title": "Error: Missing classification",
                        "field": "Error: Missing classification"
                    }

                # Validate/fuzzy match each
                valid_title = self.validate_label(raw_title, self.job_titles)
                valid_field = self.validate_label(raw_field, self.fields)

                return {
                    "title": valid_title,
                    "field": valid_field
                }
            except json.JSONDecodeError as je:
                print(f"JSON Parse Error. Content was: {content}")
                return {
                    "title": f"Error: Invalid JSON response",
                    "field": f"Error: Invalid JSON response"
                }

        except Exception as e:
            print(f"Classification Error: {str(e)}")
            return {
                "title": f"Error in classification: {str(e)}",
                "field": f"Error in classification: {str(e)}"
            }

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a stable hash for a file."""
        try:
            with open(file_path, 'rb') as f:
                # Only hash first 8KB for speed
                return hashlib.md5(f.read(8192)).hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing file {file_path}: {e}")
            return None

    def _get_cached_classification(self, file_path: str) -> Optional[Dict]:
        """Retrieve cached classification if it exists."""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return None
            
        cache_file = self.cache_dir / f"{file_hash}_{self.model}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
                return None
        return None

    def _save_classification(self, file_path: str, result: Dict) -> None:
        """Save classification result to cache."""
        try:
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                logger.debug("No file hash generated")
                return
            
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            cache_file = self.cache_dir / f"{file_hash}_{self.model}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            logger.debug(f"Successfully cached {os.path.basename(file_path)} -> {cache_file}")
            
            # Verify cache file was created
            if not cache_file.exists():
                logger.debug(f"Cache file not created: {cache_file}")
            else:
                logger.debug(f"Verified cache file exists: {cache_file}")
            
        except Exception as e:
            logger.debug(f"Cache error for {file_path}: {str(e)}")
            logger.debug(f"Cache directory: {self.cache_dir}")
            logger.debug(f"Current working directory: {os.getcwd()}")

    def process_file(self, file_path: str) -> Dict[str, str]:
        """Process a single file with caching."""
        # Check cache first
        cached = self._get_cached_classification(file_path)
        if cached:
            return cached

        # Existing classification logic
        result = self._process_file_internal(file_path)
        
        # Cache the result
        self._save_classification(file_path, result)
        return result

    def _process_file_internal(self, file_path: str) -> Dict[str, str]:
        """Internal method containing existing classification logic."""
        if not os.path.exists(file_path):
            return {"title": "Error: File not found", "field": "Error: File not found"}
        
        try:
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
                return self.classify_document(text)
            else:
                return {"title": "Error: Only PDF files are supported", 
                        "field": "Error: Only PDF files are supported"}
        except Exception as e:
            return {"title": f"Error processing file: {e}",
                    "field": f"Error processing file: {e}"}

def process_batch(classifier: DocumentClassifier, file_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Process multiple files in parallel. Returns a dict:
    {
       file_path: {"title": ..., "field": ...},
       ...
    }
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(classifier.process_file, file_path): file_path 
            for file_path in file_paths
        }
        
        results = {}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                results[file_path] = future.result()
            except Exception as e:
                results[file_path] = {"title": f"Error: {e}", "field": f"Error: {e}"}
        return results

def main():
    parser = argparse.ArgumentParser(description="Fast document classifier (Job Title + Field)")
    parser.add_argument(
        "file_paths",
        nargs='+',
        help="Path(s) to PDF file(s)"
    )
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

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY") if args.model.startswith(('gpt-', 'text-')) else os.getenv("GEMINI_API_KEY")
    
    # Validate API keys if needed
    if args.model.startswith(('gpt-', 'text-')):
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI models")
        openai.api_key = api_key
    elif args.model.startswith('gemini-'):
        if not api_key:
            raise ValueError("Gemini API key required for Google models")

    classifier = DocumentClassifier(model=args.model)
    
    if len(args.file_paths) == 1:
        result = classifier.process_file(args.file_paths[0])
        print(f"Classification Result: {result}")
    else:
        results = process_batch(classifier, args.file_paths)
        for file_path, classification in results.items():
            print(f"{os.path.basename(file_path)} -> Title: {classification['title']}, Field: {classification['field']}")

if __name__ == "__main__":
    main()
