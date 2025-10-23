ClassMind — AI Study Assistant

Automated summarization, mind mapping, and flashcard generation from lecture slides.

Overview

ClassMind converts PDF or PowerPoint lecture slides into structured study materials.
It extracts text, summarizes content, builds mind maps, and creates Anki flashcards — all in one step.
Supports both English and German, with optional OCR for image-based slides.

Live Demo

https://classmind-app-yourname.streamlit.app

Key Features

Extractive deck summarization (accurate, multilingual)

Mind map export (.mmd for Mermaid)

Semantic search using multilingual embeddings

Automatic flashcard generation (.csv for Anki)

Lightweight: runs fully on CPU

Tech Stack

Streamlit • PyMuPDF • python-pptx • Sentence-Transformers • scikit-learn • Pillow • pytesseract (optional)

Run Locally
git clone https://github.com/<md-touhidur-rahman/classmind-app.git
cd classmind-app
pip install -r requirements.txt
streamlit run app.py

Output Files
File	Purpose
deck_summary.txt	Overall summary
mindmap.mmd	Topic hierarchy
flashcards.csv	Study flashcards
slide_summaries.json	Slide text data

Author
Md Touhidur Rahman
