import streamlit as st
import time
import os

from src.dataset import load_comtrans_data
from src.smt_model import StatisticalMT
from src.evaluation import evaluate_translation

st.set_page_config(page_title="English-German MT", layout="wide")

@st.cache_resource
def load_models():
    # Load small subset (e.g. 2000 sentences for fast app startup)
    data = load_comtrans_data(num_sentences=2000)
    
    model_en2de = StatisticalMT(direction="en2de")
    model_de2en = StatisticalMT(direction="de2en")
    
    model_en2de.train(data, iterations=5)
    model_de2en.train(data, iterations=5)
    
    return model_en2de, model_de2en

st.title("Bidirectional Statistical Machine Translation")
st.markdown("English ↔ German Phrase-Based/Statistical MT")

# Show loading spinner while NLTK dataset is downloaded/trained
with st.spinner("Training IBM Model 1 on parallel corpus... (this will take 30-60 secs on first load)"):
    model_en2de, model_de2en = load_models()

# UI Layout
tab1, tab2 = st.tabs(["Translate", "Evaluation & Error Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Text")
        direction = st.radio("Translation Direction", ["English -> German", "German -> English"])
        
        default_val = "The government announced a new housing scheme." if "English" in direction.split("->")[0] else "Die regierung kündigte ein neues wohnungsbauprogramm an."
        input_text = st.text_area("Enter text to translate:", value=default_val)
        
        if st.button("Translate"):
            if not input_text:
                st.warning("Please enter text.")
            else:
                start_time = time.time()
                
                if direction == "English -> German":
                    translation = model_en2de.translate(input_text)
                else:
                    translation = model_de2en.translate(input_text)
                    
                time_taken = time.time() - start_time
                
                with col2:
                    st.subheader("Translation Output (SMT)")
                    st.info(translation)
                    st.caption(f"Translated in {time_taken:.3f} seconds.")

with tab2:
    st.header("Evaluation (BLEU Score)")
    st.markdown("Test the translation quality against a known reference.")
    
    ref_text = st.text_input("Reference (Ground Truth):", "Die regierung kündigte ein neues wohnungsbauprogramm an.")
    hyp_text = st.text_input("Hypothesis (Model Output):", "Die regierung kündigte ein neues.")
    
    if st.button("Calculate BLEU"):
        score = evaluate_translation(ref_text, hyp_text)
        st.metric(label="BLEU Score", value=f"{score:.4f}")
        
    st.markdown("---")
    st.subheader("Qualitative Analysis Notes")
    st.markdown("""
    * **Strengths**: Memorizes exact phrases from the training set flawlessly (Pseudo Phrase-Based). Handles simple word-to-word alignments well via IBM Model 1.
    * **Errors**: Struggles with long-distance reordering (e.g. German verbs moving to the end). Unseen words map directly without morphological fallback. Because the vocabulary size in realistic bounds is relatively small, unknown tokens appear frequently in Out of Domain inputs.
    """)
