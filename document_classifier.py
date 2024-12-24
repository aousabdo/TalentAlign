import fitz  # PyMuPDF for PDF handling
import json
import os
import argparse
import ollama
from difflib import get_close_matches
from typing import List, Dict, Optional
import concurrent.futures

class DocumentClassifier:
    def __init__(self, model: str = "phi3"):
        self.model = model
        self.job_titles = self.load_job_categories()
        self.text_cache: Dict[str, str] = {}
        
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
            print(f"Warning: Using fallback categories. Error: {e}")
            return [
                "Data Scientist", "Software Engineer", "Doctor", "Nurse",
                "Teacher", "Lawyer", "Accountant", "Product Manager"
            ]

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 3) -> str:
        """Extract text from first few pages of PDF with caching."""
        if pdf_path in self.text_cache:
            return self.text_cache[pdf_path]
        
        try:
            with fitz.open(pdf_path) as doc:
                # Only extract first few pages for faster processing
                text = "\n".join(
                    page.get_text() 
                    for page in doc.pages(0, min(max_pages, len(doc)))
                )
                
                # Cache the result
                self.text_cache[pdf_path] = text
                return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def validate_classification(self, classification: str) -> str:
        """Validate and correct classification using fuzzy matching."""
        if classification in self.job_titles:
            return classification
            
        matches = get_close_matches(
            classification.lower(),
            [title.lower() for title in self.job_titles],
            n=1,
            cutoff=0.6
        )
        
        if matches:
            return next(
                title for title in self.job_titles 
                if title.lower() == matches[0]
            )
            
        return classification

    def classify_document(self, text: str) -> str:
        """Classify document using LLM."""
        # Take first and last portions of text for efficiency
        max_length = 2500
        if len(text) > max_length:
            text = f"{text[:max_length//2]}\n...\n{text[-max_length//2:]}"

        prompt = f"""Classify this document into ONE job category from this list:
{', '.join(self.job_titles)}

Document:
{text}

Return ONLY the job title. No other text."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a job classifier. Respond with exactly one job title from the provided list."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={"temperature": 0}  # Reduce randomness for speed and consistency
            )
            
            classification = response['message']['content'].strip()
            # Basic cleaning
            classification = classification.split('\n')[0].split(',')[0].strip()
            
            # Validate and return
            return self.validate_classification(classification)

        except Exception as e:
            return f"Error in classification: {e}"

    def process_file(self, file_path: str) -> str:
        """Process a single file and return classification."""
        if not os.path.exists(file_path):
            return "Error: File not found"
        
        try:
            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
                return self.classify_document(text)
            else:
                return "Error: Only PDF files are supported"
        except Exception as e:
            return f"Error processing file: {e}"

def process_batch(classifier: DocumentClassifier, file_paths: List[str]) -> Dict[str, str]:
    """Process multiple files in parallel."""
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
                results[file_path] = f"Error: {str(e)}"
                
        return results

def main():
    parser = argparse.ArgumentParser(description="Fast document job classifier")
    parser.add_argument(
        "file_paths",
        nargs='+',
        help="Path(s) to PDF file(s)"
    )
    parser.add_argument(
        "--model",
        default="phi3",
        help="Model to use for classification (default: phi3)"
    )
    args = parser.parse_args()

    classifier = DocumentClassifier(model=args.model)
    
    if len(args.file_paths) == 1:
        # Single file processing
        result = classifier.process_file(args.file_paths[0])
        print(f"Job Classification: {result}")
    else:
        # Batch processing
        results = process_batch(classifier, args.file_paths)
        for file_path, classification in results.items():
            print(f"{os.path.basename(file_path)}: {classification}")

if __name__ == "__main__":
    main()