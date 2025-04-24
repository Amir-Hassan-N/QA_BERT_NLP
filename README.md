# 🤖 Question Answering System Using BERT in Python

## Context-Aware QA Pipeline with BERT and Word2Vec

### MSc Final Project – Data Science & Business Informatics – University of Pisa

## 📌 Overview
This project demonstrates the development of a Contextual Question Answering System (QAS) using BERT (Bidirectional Encoder Representations from Transformers) and Word2Vec in Python. The goal is to build a system that reads a given article and accurately answers user questions based on context—enhancing traditional information retrieval with deep NLP techniques.

By leveraging pretrained transformer models and semantic embeddings, this QAS goes beyond keyword matching and retrieves contextually correct answers, showing the power of modern language models in real-world applications.

## 🎯 Objectives
 ●  Develop a QAS that understands natural language context and returns precise answers

 ●  Fine-tune BERT for SQuAD-style question answering

 ●  Implement a Word2Vec-based semantic similarity baseline

 ●  Compare both models for performance and accuracy

 ●  Handle long-text limitations of transformer models through smart chunking and preprocessing

## 🧠 Model Architecture
### 1. Word2Vec-Based QA (Baseline)
✦ Trained on dataset with cleaned question-answer pairs

✦ Questions and text passages embedded using Word2Vec

✦ Answers selected based on cosine similarity between question and sentence embeddings

### 2. BERT-Based QA
✦ Used HuggingFace BertForQuestionAnswering model

✦ Fine-tuned on Q&A data following SQuAD format

✦ Employed chunking (512 tokens + stride) to handle large article inputs

✦ Selected the most probable answer span based on model confidence

## 🛠️ Tools & Technologies
➤ Languages: Python

➤ Frameworks: PyTorch, Transformers (HuggingFace), Gensim

➤ Libraries: transformers, torch, sklearn, pandas, matplotlib

➤ IDE: Jupyter Notebook / Anaconda

## 🧪 Dataset
Source: Public Question-Answer dataset curated by [Rachael Tatman]

➤ Features:

➤ Article Title

➤ Question Answer Difficulty (from questioner and answerer) Associated Wikipedia article text Preprocessing involved merging yearly subsets, cleaning noisy tokens, and handling missing data

## ⚙️ Pipeline Architecture

graph TD;
    A[Load Dataset] --> B[Preprocessing & Cleaning];
    B --> C[Word2Vec Embedding];
    B --> D[BERT Tokenization];
    C --> E[Cosine Similarity Matching];
    D --> F[BERT Span Prediction];
    E --> G[Answer Selection];
    F --> G;
## 📈 Results

➤ Model	Task	Performance Highlights
➤ Word2Vec	Sentence Similarity QA	Reasonable accuracy using cosine similarity
➤ BERT	Span-Based QA (SQuAD format)	High precision answers in long-context data
➤ BERT Chunking	Handles long texts > 512 tokens	Achieved accurate span predictions
### 📌 Output: BERT consistently returned more context-aware answers, while Word2Vec showed strong performance in basic semantic retrieval.

## 🔍 Key Features
✦ Handles long contexts using BERT stride-based chunking

✦ Interactive answer prediction for any given article and question

✦ Visualizations of model performance and predictions

✦ Scalable architecture for extension into chatbots or virtual assistants

## 👨‍🎓 Author
### Amir Hassan
### 📧 amirhassanunipi29@gmail.com
### 🎓 Master’s in Data Science & Business Informatics – Università di Pisa, Italy
