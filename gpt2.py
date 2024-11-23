import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

@st.cache_resource
def load_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer, model

def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt") # type: ignore

    outputs = model.generate( # type: ignore
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore

tokenizer, model = load_model()

st.title("Text Generator with GPT-2")

prompt = st.text_input("Enter a prompt:")
max_length = st.slider("Maximum length of generated text:", min_value=10, max_value=500, value=100)

if st.button("Generate Text"):
    if prompt:
        with st.spinner("Generating text...."):
            generated_text = generate_text(prompt, max_length)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt.")                 
