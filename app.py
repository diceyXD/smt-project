import streamlit as st
import time
import pickle

from src.smt_model import StatisticalMT
from src.evaluation import evaluate_translation

st.set_page_config(page_title="English ↔ German MT", layout="wide")

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        cache = pickle.load(f)
        
    model_en2de = StatisticalMT(direction="en2de")
    model_en2de.best_trans = cache["en2de_trans"]
    model_en2de.phrase_memory = cache["en2de_phrase"]
    
    model_de2en = StatisticalMT(direction="de2en")
    model_de2en.best_trans = cache["de2en_trans"]
    model_de2en.phrase_memory = cache["de2en_phrase"]
    
    return model_en2de, model_de2en

st.title("Bidirectional Statistical Machine Translation")
st.markdown("English ↔ German Phrase-Based/Statistical MT")

with st.spinner("Loading pre-trained IBM Model 1..."):
    model_en2de, model_de2en = load_models()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input & Translate")
    direction = st.radio("Translation Direction", ["English -> German", "German -> English"])
    
    # Default placeholder text based on direction
    default_val = "The government announced a new housing scheme." if "English" in direction.split("->")[0] else "Die regierung kündigte ein neues wohnungsbauprogramm an."
    input_text = st.text_area("Enter text to translate:", value=default_val)
    
    translate_btn = st.button("Translate", type="primary")

with col2:
    st.subheader("2. Translation Output")
    
    # We use session state to persist the translation across widget interactions (like clicking Evaluate later)
    if translate_btn:
        if not input_text:
            st.warning("Please enter text.")
        else:
            start_time = time.time()
            if direction == "English -> German":
                st.session_state.translation = model_en2de.translate(input_text)
            else:
                st.session_state.translation = model_de2en.translate(input_text)
            st.session_state.time_taken = time.time() - start_time
            
    if "translation" in st.session_state:
        st.info(st.session_state.translation)
        st.caption(f"Translated in {st.session_state.time_taken:.3f} seconds.")

st.markdown("---")

if "translation" in st.session_state:
    st.subheader("3. Interactive Error Analysis & Evaluation")
    st.markdown("Provide the **Ground Truth Reference** to mathematically evaluate the model's output using the **BLEU** metric.")
    
    ref_col, btn_col = st.columns([4, 1])
    with ref_col:
        # Provide a matching default ground truth for the demo inputs
        default_ref = "Die regierung kündigte ein neues wohnungsbauprogramm an." if "English" in direction else "The government announced a new housing scheme."
        ref_text = st.text_input("Ground Truth Reference:", default_ref)
    
    with btn_col:
        st.write("")
        st.write("") # Alignment padding
        eval_btn = st.button("Evaluate Score")
        
    if eval_btn:
        score = evaluate_translation(ref_text, st.session_state.translation)
        
        # Display score nicely
        st.metric(label="BLEU Score", value=f"{score:.4f}")
