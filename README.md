# 🧠 Product Review Summarizer

### A Natural Language Processing (NLP) Project  
Summarizing multiple customer reviews of digital music products using **Extractive** and **Abstractive** text summarization techniques.

---

## 📖 Project Overview

This project implements an intelligent system that generates concise summaries of product reviews.  
Given a product ID (ASIN), product title, or a block of user-provided text, the system fetches relevant reviews and produces a readable summary highlighting the overall sentiment and key feedback points.

The goal is to help users quickly understand collective opinions about a product without reading hundreds of individual reviews.

---

## 🎯 Objectives

- Develop a **working NLP pipeline** that can summarize product reviews accurately.  
- Compare **Extractive** and **Abstractive** summarization techniques.  
- Build an **interactive Flask web application** for easy use.  
- Evaluate summarization performance using standard metrics (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore).

---

## 🧩 Features

✅ **Dual Summarization Options**
- **Extractive Summary:** Selects the most important sentences directly from reviews using TF-IDF and semantic embeddings.  
- **Abstractive Summary:** Generates human-like summaries using a pretrained **BART** transformer model.

✅ **Smart Review Search**
- Enter a **Product ID (parent_asin)**, **Product Title**, or paste **custom review text** directly.  
- The app automatically fetches and displays all reviews, titles, ratings, and helpfulness scores.

✅ **User Interface**
- Clean, responsive Flask web app.  
- Option to select summary type (Extractive / Abstractive).  
- Displays reviews and generated summaries in an easy-to-read layout.  
- Handles GPU memory limits gracefully with clear user messages.

✅ **Automatic Error Handling**
- If abstractive summarization exceeds GPU memory limits, the app automatically falls back to extractive summarization and notifies the user.

---

## 🧠 NLP Techniques Used

| Technique | Purpose | Libraries |
|------------|----------|-----------|
| **TF-IDF Vectorization** | Represent review sentences numerically for extractive scoring | scikit-learn |
| **Cosine Similarity + TextRank** | Rank sentences for extractive summarization | numpy, networkx |
| **Sentence-BERT Embeddings** | Measure semantic similarity between sentences | sentence-transformers |
| **Seq2Seq Transformer (BART)** | Generate fluent abstractive summaries | transformers, torch |
| **Evaluation Metrics** | Compare summary quality | evaluate, bert-score |

---

## 🧮 Evaluation Metrics

Summaries were evaluated using standard text summarization metrics:

| Metric | Description |
|---------|--------------|
| **ROUGE-1** | Overlap of individual words between system and reference summaries |
| **ROUGE-2** | Overlap of word pairs (bigrams) |
| **ROUGE-L** | Longest common subsequence (sentence structure similarity) |
| **BERTScore** | Semantic similarity using contextual embeddings |

### ✴️ Key Observations
- **ROUGE-1 ≈ 0.62, ROUGE-2 ≈ 0.56, ROUGE-L ≈ 0.48** → good content coverage.  
- **BERT F1 ≈ 0.54** → abstractive summaries capture core meaning while rephrasing naturally.  
- Abstractive summaries are ~10-15% shorter and more fluent than extractive ones.  

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|-------|-------------------|
| **Backend** | Flask, Python 3.x |
| **NLP** | Transformers (BART), Sentence-BERT, NLTK, scikit-learn |
| **Evaluation** | Hugging Face evaluate, bert-score |
| **Frontend** | HTML5, CSS3, Vanilla JS |
| **Hardware Used** | Ryzen 7 CPU, 16 GB RAM, RTX 3060 GPU |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/product-review-summarizer.git
cd product-review-summarizer
```

### 2️⃣ Create a virtual environment
```
python -m venv env
```
Activate it:
```
env\Scripts\activate
```


### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run the Flask app
```
python app.py
```

Then open your browser and visit: http://127.0.0.1:5000


---

## 🧱 Project Structure
```
## 🧱 Project Structure

Product Review Summarizer/
│
├── app.py                   
│
├── Data/                           
│   ├── grouped_reviews.csv         
│   └── merged_reviews.csv       
│
├── notebooks/                
│   ├── Preprocessing.ipynb          
│   ├── Extractive_summary.ipynb   
│   └── Summarization_module.ipynb   
│
├── templates/                   
│   ├── index.html                 
│   └── result.html            
│
├── static/                        
│   └── style.css                 
│
├── requirements.txt             
└── README.md                       

```


## 🧩 Dataset Information

The system uses preprocessed **Digital Music** product data containing:
- parent_asin — unique product ID  
- product_title — name of the product  
- review_text — user-written review  
- title — review headline  
- rating — user rating (1–5)  
- helpful_vote — number of helpful votes  
- verified_purchase — boolean indicating verified buyers  
- average_rating — product’s overall average rating  

Each product can have multiple reviews grouped by parent_asin.

---

## 💡 How It Works (Simplified Flow)

1. **User Input** → Product ID / Title / Custom Text  
2. **Review Retrieval** → Filter dataset by input  
3. **Summarization**
   - *Extractive:* Rank sentences using embeddings and TextRank.
   - *Abstractive:* Generate human-like summary using BART model.  
4. **Output Display** → Summary + all relevant reviews in an organized table.  
5. **Error Handling** → On GPU memory issue, switch to extractive with a friendly message.

---

## 🧭 Results & Insights

- Extractive summarization offers factual, sentence-level summaries.
- Abstractive summarization produces more natural, paraphrased outputs.
- Hybrid system balances **accuracy** and **readability** effectively.
- Demonstrates how transformer-based models outperform classical extractive methods for textual coherence.

---

## 🚀 Future Improvements

- Fine-tune the abstractive model on a larger, domain-specific dataset.  
- Add **sentiment-based pros/cons extraction** alongside summaries.  
- Support multilingual product reviews.  
- Add caching to avoid recomputation for the same product IDs.

---


