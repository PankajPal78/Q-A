import streamlit as st
from transformers import pipeline

# Hide warnings (optional)
import warnings, os
from transformers import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load text generation pipeline
generator = pipeline("text-generation", model="gpt2", device=-1)

# Streamlit UI
st.title("Q&A")
st.write("Generate text using Hugging Face GPT-2 model")

# User input
prompt = st.text_area("Enter your prompt:")

# Generate button
if st.button("Generate"):
    result = generator(prompt, max_new_tokens=100, truncation=True)
    st.subheader("Generated Text:")
    st.write(result[0]["generated_text"])
