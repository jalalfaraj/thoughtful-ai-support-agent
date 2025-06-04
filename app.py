#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 19:19:55 2025

@author: jalalfaraj
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Set page config as the first Streamlit command
st.set_page_config(page_title="Thoughtful AI Support Agent", page_icon="🤖")

# Hardcoded FAQ dataset with keywords
FAQ_DATA = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections.",
        "keywords": ["eligibility verification", "eva"]
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements.",
        "keywords": ["claims processing", "cam"]
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden.",
        "keywords": ["payment posting", "phil"]
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others.",
        "keywords": ["agents", "thoughtful ai"]
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting.",
        "keywords": ["benefits", "advantages", "using"]
    }
]

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return embedder, classifier

embedder, classifier = load_models()
faq_questions = [entry["question"] for entry in FAQ_DATA]
faq_answers = [entry["answer"] for entry in FAQ_DATA]
faq_keywords = [entry["keywords"] for entry in FAQ_DATA]
question_embeddings = embedder.encode(faq_questions, convert_to_tensor=True)

def get_best_response(user_input: str) -> str:
    user_input_clean = user_input.lower()

    # Priority 1: Keyword filtering
    keyword_matched_indices = []
    for idx, entry in enumerate(FAQ_DATA):
        if any(kw in user_input_clean for kw in entry["keywords"]):
            keyword_matched_indices.append(idx)

    if keyword_matched_indices:
        filtered_questions = [faq_questions[i] for i in keyword_matched_indices]
        filtered_embeddings = torch.stack([question_embeddings[i] for i in keyword_matched_indices])
        user_embedding = embedder.encode(user_input, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(user_embedding, filtered_embeddings)[0]
        best_idx = keyword_matched_indices[torch.argmax(cosine_scores).item()]
        if cosine_scores[torch.argmax(cosine_scores)] >= 0.6:
            return FAQ_DATA[best_idx]["answer"]

    # Priority 2: Sentence similarity across full FAQ
    user_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(cosine_scores).item()
    if cosine_scores[best_match_idx] >= 0.65:
        return FAQ_DATA[best_match_idx]["answer"]

    # Priority 3: Zero-shot classifier fallback
    result = classifier(user_input, faq_questions)
    if result["scores"][0] >= 0.6:
        best_label = result["labels"][0]
        for entry in FAQ_DATA:
            if entry["question"] == best_label:
                return entry["answer"]

    # Fallback response
    return "I'm your Thoughtful AI support agent. You can ask me about our automation tools like EVA, CAM, or PHIL."

def main():
    st.title("Thoughtful AI Support Agent")
    st.write("Ask me anything about Thoughtful AI’s automation agents. I’ll try my best to help!")

    user_input = st.text_input("Your Question:")
    if user_input:
        try:
            response = get_best_response(user_input)
            st.markdown(f"**Agent:** {response}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

main()
