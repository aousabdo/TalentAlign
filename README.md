# TalentAlign: Advanced Resume-Job Description Matching System

TalentAlign is a sophisticated document matching system that uses advanced NLP and multiple AI models to intelligently match resumes with job descriptions. It employs a multi-layered analysis approach combining semantic understanding, domain expertise, and skill alignment.

## 🚀 Key Features

- **Multi-Model AI Support**
  - OpenAI Models (gpt-4o, etc.)
  - Google Gemini 2.0 Flash Exp
  - Local Models (via Ollama):
    - Llama 3.2
    - Phi-3
    - Mistral
  - Flexible architecture for easy model addition

- **Intelligent Classification**
  - Job Title Recognition
  - Field/Industry Detection
  - Domain Expertise Mapping
  - Cross-domain Skill Transfer Analysis

- **Advanced Matching Algorithm**
  - Document-level Semantic Similarity
  - Section-wise Alignment
  - Bullet Point Analysis
  - TF-IDF Keyword Matching
  - Field-aware Scoring
  - Title Family Recognition

- **Performance Optimizations**
  - Smart Caching System
  - Batch Processing
  - Concurrent Execution
  - Memory-efficient Text Processing

## 🛠 Installation & 🛠 Environment Setup

```bash
git clone https://github.com/aousabdo/TalentAlign.git
cd TalentAlign

```

### Using venv (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
## On Windows
venv\Scripts\activate
## On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Using conda
```bash
# Create conda environment
conda create -n talentalign python=3.9
conda activate talentalign
pip install -r requirements.txt
```


## 🔑 API Keys Setup

```bash
# Add these to your .env file or export in your shell
export OPENAI_API_KEY="your-key-here"  # For OpenAI models
export GEMINI_API_KEY="your-key-here"  # For Google models

# For local models, install Ollama
# Visit: https://ollama.ai/download
```

## 💻 Usage

# Ensure your virtual environment is activated
source venv/bin/activate  # or conda activate talentalign

### Basic Matching

```bash
python match_resume_jd_optimized.py path/to/resume.pdf path/to/job_description.pdf
```

### Model Selection
```bash
# Using OpenAI GPT-4
python match_resume_jd_optimized.py resume.pdf jd.pdf --model gpt-4o

# Using Google Gemini
python match_resume_jd_optimized.py resume.pdf jd.pdf --model gemini-2.0-flash-exp

# Using Local Models
python match_resume_jd_optimized.py resume.pdf jd.pdf --model llama3.2
```

### Document Classification
```bash
python document_classifier.py path/to/document.pdf --model gpt-4o
```

## 🎯 Matching Algorithm

The system uses a sophisticated multi-layer matching approach:

1. **Quick Compatibility Check**
   - Job Title Analysis
   - Field/Industry Alignment
   - Domain Expertise Verification

2. **Deep Content Analysis**
   - Document Embedding Similarity
   - Section-level Alignment
   - Requirement-Experience Matching
   - Keyword Density Analysis

3. **Smart Scoring System**
   - Title Match: 100% weight
   - Field Match: 60% weight
   - Cross-domain Skills: Partial scoring
   - Section Alignment: Weighted contribution

## 📊 Output Example

```
=== SECTION-LEVEL PARSING RESULTS ===
Resume Sections Found:
  [Skills] => 1303 chars
  [Education] => 1458 chars
  [Professional Experience] => 18142 chars
  [Publications] => 659 chars
  [Honors] => 540 chars

JD Sections Found:
  [Misc] => 1250 chars
  [Responsibilities] => 745 chars
  [Requirements] => 586 chars

=== MATCH RESULTS ===
Document Embedding Similarity: 0.7931
TF-IDF Keyword Score:         0.9000
Bullet (Resp) Score:          0.7519
Bullet (Req) Score:          0.7634
Bullet Average:              0.7576
Section-level alignment:     0.8680
Final Combined Score:        0.8390
```

## 🔄 Recent Updates

- Added field-aware matching for cross-domain compatibility
- Implemented job title family recognition (tech, data, product roles)
- Integrated Google Gemini support
- Enhanced caching system for embeddings and classifications
- Improved domain detection with weighted phrase matching
- Added support for multiple local models via Ollama

## 📝 Requirements

- Python 3.8+
- PDF documents only
- Required packages (see requirements.txt):
  - OpenAI API client (>=1.0.0)
  - Google Generative AI
  - PyMuPDF (fitz)
  - scikit-learn
  - numpy
  - Other utilities (see requirements.txt)
- OpenAI API key for GPT models
- Google API key for Gemini models
- Ollama installed for local models
  - Download from: https://ollama.ai/download

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see LICENSE file for details

## 📚 Project Structure

```
TalentAlign/
├── document_classifier.py      # Job title & field classification
├── match_resume_jd_optimized.py # Main matching algorithm
├── benchmark_models.py         # Model performance testing
├── job_categories.json        # Job title taxonomy
├── requirements.txt           # Package dependencies
└── process_documents.sh       # Batch processing script
```
