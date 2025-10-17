import os
import re
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from tqdm.auto import tqdm

# NLP model imports
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# Ensure punkt is available for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# ---------- CONFIG ----------
CSV_PATH = "merged_reviews.csv"   # change if needed
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else -1
MAX_DISPLAY_REVIEWS = 200

# Model names
ABSTRACTIVE_MODEL_NAME = "facebook/bart-large-cnn"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- LOAD DATA ----------
print("Loading merged reviews CSV...")
df = pd.read_csv(CSV_PATH)
df['review_text'] = df['review_text'].fillna("")
df['title'] = df['title'].fillna("")
df['product_title'] = df['product_title'].fillna("")
print(f"Loaded {len(df)} reviews from {CSV_PATH}")

# ---------- LOAD MODELS ----------
print("Loading models (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    ABSTRACTIVE_MODEL_NAME,
    torch_dtype=torch.float16 if USE_GPU else torch.float32
)
device_str = "cuda" if USE_GPU else "cpu"
model.to(device_str)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if USE_GPU else -1
)

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_str)
print("✅ Models loaded.")

# ---------- UTIL FUNCTIONS ----------
def split_sentences(text):
    text = (text or "").strip()
    if not text:
        return []
    sents = sent_tokenize(text)
    sents = [s.strip() for s in sents if len(s.strip()) > 3]
    return sents

def extract_key_by_embedding(text, top_k=6, diversity=0.7):
    sents = split_sentences(text)
    if not sents:
        return ""
    embs = embed_model.encode(sents, convert_to_numpy=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    scores = embs.dot(centroid)
    selected = []
    selected_idx = []
    for _ in range(min(top_k, len(sents))):
        if not selected:
            idx = int(np.argmax(scores))
        else:
            sim_to_selected = np.max(np.dot(embs, embs[selected_idx].T), axis=1)
            mmr = diversity * scores - (1 - diversity) * sim_to_selected
            mmr[selected_idx] = -np.inf
            idx = int(np.argmax(mmr))
        selected_idx.append(idx)
        selected.append(sents[idx])
    selected_idx_sorted = sorted(selected_idx)
    return " ".join([sents[i] for i in selected_idx_sorted])

def chunk_text_for_model(text, tokenizer, max_tokens=800, stride=50):
    toks = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = toks['input_ids'][0].tolist()
    chunks = []
    i = 0
    while i < len(input_ids):
        chunk_ids = input_ids[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        i += max_tokens - stride
    return chunks

def summarize_abstractive(text, max_input_tokens=800):
    if not text or not text.strip():
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    sents = split_sentences(text)
    if len(sents) <= 8:
        prompt = "Write a concise, fluent summary in your own words for the following product reviews:\n\n" + text
        out = summarizer(
            prompt,
            max_length=180,
            min_length=40,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2
        )
        return out[0]['summary_text']
    chunks = chunk_text_for_model(text, tokenizer, max_tokens=max_input_tokens)
    chunk_summaries = []
    for c in chunks:
        key = extract_key_by_embedding(c, top_k=6)
        if not key:
            key = c
        prompt = "Write a concise, fluent summary in your own words for the following product reviews:\n\n" + key
        out = summarizer(
            prompt,
            max_length=180,
            min_length=40,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2
        )
        chunk_summaries.append(out[0]['summary_text'])
    combined = " ".join(chunk_summaries)
    prompt = "Write a concise, fluent summary in your own words for the following product reviews:\n\n" + combined
    out = summarizer(
        prompt,
        max_length=200,
        min_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )
    return out[0]['summary_text']

def summarize_extractive(text, top_k_sentences=6):
    summary = extract_key_by_embedding(text, top_k=top_k_sentences, diversity=0.7)
    if not summary:
        s = split_sentences(text)
        return " ".join(s[:min(3, len(s))])
    return summary

# ---------- FLASK ROUTES ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

def get_reviews_by_input(user_input):
    ui = (user_input or "").strip()
    if not ui:
        return pd.DataFrame([], columns=df.columns)
    mask_asin = df['parent_asin'].str.lower() == ui.lower()
    if mask_asin.any():
        return df[mask_asin].copy()
    mask_title = df['product_title'].str.contains(ui, case=False, na=False)
    if mask_title.any():
        return df[mask_title].copy()
    row = {
        'parent_asin': "USER_INPUT",
        'rating': float("nan"),
        'title': "",
        'review_text': ui,
        'helpful_vote': 0,
        'verified_purchase': False,
        'product_title': "User provided text",
        'average_rating': float("nan")
    }
    return pd.DataFrame([row])

@app.route("/summarize", methods=["POST"])
def summarize_route():
    user_input = request.form.get("user_input", "").strip()
    summary_type = request.form.get("summary_type", "extractive")
    try:
        top_k = int(request.form.get("top_k", 6))
    except Exception:
        top_k = 6

    if not user_input:
        return redirect(url_for('index'))

    reviews_df = get_reviews_by_input(user_input)
    if reviews_df.empty:
        return render_template("result.html", error="No reviews found for the input.", reviews=[], summary="", input_text=user_input)

    reviews_to_show = reviews_df.sort_values(by="helpful_vote", ascending=False)
    reviews_to_display = reviews_to_show.head(MAX_DISPLAY_REVIEWS)
    combined_text = " ".join([str(t) for t in reviews_df['review_text'].tolist() if str(t).strip()])

    error_msg = None
    if summary_type == "extractive":
        summary = summarize_extractive(combined_text, top_k_sentences=top_k)
    else:
        # Try abstractive; on any exception (e.g., CUDA OOM) fall back immediately to extractive
        try:
            summary = summarize_abstractive(combined_text)
        except Exception as e:
            # Fallback: extractive summary and inform the user
            summary = summarize_extractive(combined_text, top_k_sentences=top_k)
            error_msg = ("Abstractive summarization failed due to resource limits on the server. "
                         "Automatically switched to Extractive summary. Try a smaller input if you need abstractive output.")

    review_rows = []
    for _, r in reviews_to_display.iterrows():
        review_rows.append({
            "title": r.get("title", ""),
            "text": r.get("review_text", ""),
            "rating": r.get("rating", ""),
            "helpful_vote": int(r.get("helpful_vote", 0)),
            "verified_purchase": bool(r.get("verified_purchase", False)),
            "parent_asin": r.get("parent_asin", ""),
            "product_title": r.get("product_title", "")
        })

    return render_template(
        "result.html",
        input_text=user_input,
        product_title=reviews_to_display.iloc[0]['product_title'] if len(reviews_to_display)>0 else "",
        parent_asin=reviews_to_display.iloc[0]['parent_asin'] if len(reviews_to_display)>0 else "",
        summary_type="Extractive summary" if summary_type=="extractive" else "Abstractive summary",
        summary=summary,
        reviews=review_rows,
        error_msg=error_msg
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
