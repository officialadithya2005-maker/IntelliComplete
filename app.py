import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sentence Completion using LSTM",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load Model & Files
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model.h5")


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_max_len():
    with open("max_len.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
tokenizer = load_tokenizer()
max_len = load_max_len()

# -----------------------------
# Predictor Function
# -----------------------------
def predictor(model, tokenizer, text, max_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len,
        padding='pre'
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None


# -----------------------------
# Generate Sentence
# -----------------------------
def generate_text(model, tokenizer, seed_text, n_words, max_len):

    text = seed_text

    for _ in range(n_words):

        next_word = predictor(model, tokenizer, text, max_len)

        if next_word is None:
            break

        # Stop if model predicts the same word repeatedly
        words = text.split()

        if len(words) > 0 and next_word == words[-1]:
            break

        text += " " + next_word

    return text


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Sentence Completion using LSTM")

st.write(
    "Enter the beginning of a sentence and let the model complete it."
)

user_input = st.text_input(
    "Enter your sentence",
    placeholder="Example: Once upon a time"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=30,
    value=15
)

if st.button("Generate Sentence"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Generating sentence..."):

            generated_sentence = generate_text(
                model,
                tokenizer,
                user_input,
                num_words,
                max_len
            )

        st.success("Generated Sentence")

        st.markdown(
            f"""
            <div style="
                background:#f5f5f5;
                padding:20px;
                border-radius:10px;
                font-size:20px;
                color:black;">
                {generated_sentence}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Built using TensorFlow • LSTM • Streamlit")