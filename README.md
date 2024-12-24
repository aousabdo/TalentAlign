# TalentAlign

An intelligent resume matching system that helps HR professionals find the best candidates for their job openings. TalentAlign uses advanced NLP and machine learning techniques to analyze resumes and job descriptions, providing accurate matching scores based on multiple factors.

## Features

- Advanced document similarity analysis using OpenAI embeddings
- Section-level resume and job description parsing
- Bullet-point based matching
- TF-IDF keyword analysis
- Hard requirements verification
- Weighted scoring system

## Technical Stack

- Python 3.x
- OpenAI API
- scikit-learn
- PyPDF2
- NumPy

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install openai scikit-learn PyPDF2 numpy
```
3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

```bash
python match_resume_jd.py <resume.pdf> <job_description.pdf>
```

## Coming Soon

- Web interface for HR professionals
- Batch processing of multiple resumes
- Candidate ranking system
- Resume and JD management dashboard
- Export and reporting features

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 