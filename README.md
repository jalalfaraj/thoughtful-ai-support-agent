# Thoughtful AI Support Agent 🧠🤖

This is a conversational AI agent I built for a technical screen. It answers user 
questions about Thoughtful AI's automation agents using a mix of semantic search and 
zero-shot classification.

The idea was to simulate a helpful support bot that gives accurate answers from a 
hardcoded FAQ, while also gracefully handling unexpected input. I added layers like 
keyword prioritization, abbreviation matching, and fallback to an LLM when needed.

Built with:
- Streamlit for the UI
- Hugging Face Transformers for zero-shot classification
- Sentence Transformers for semantic similarity

### Run locally
```bash
streamlit run app.py'''

### Streamlit Website:
https://thoughtful-ai-support-agent-ufghda8uj4kf4wi7u4iy9f.streamlit.app/
