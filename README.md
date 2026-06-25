# 🧠 IntelliComplete - AI Sentence Completion using LSTM

An AI-powered Sentence Completion application built using **TensorFlow, Keras, LSTM, and Streamlit**. The model predicts and generates the most probable continuation of a user-provided sentence by leveraging a Long Short-Term Memory (LSTM) neural network trained on sequential text data.

---

## 🚀 Features

- 🔮 Predicts the next words to complete a sentence
- 🧠 Deep Learning model built using LSTM
- 📚 Text preprocessing with Keras Tokenizer
- ⚡ Interactive Streamlit web application
- 🎯 User-friendly interface for real-time sentence generation
- 💾 Pre-trained model for instant predictions

---

## 📸 Demo

Enter the beginning of a sentence:

```
Once upon a time
```

Generated Output:

```
Once upon a time there was a young prince who dreamed of exploring the world.
```

*(Output depends on the trained model and dataset.)*

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- LSTM (Long Short-Term Memory)
- NumPy
- Streamlit
- Pickle

---

## 📂 Project Structure

```
IntelliComplete/
│
├── app.py                  # Streamlit Application
├── lstm_model.h5           # Trained LSTM Model
├── tokenizer.pkl           # Saved Tokenizer
├── max_len.pkl             # Maximum sequence length
├── requirements.txt
├── README.md
└── notebook.ipynb          # Model training notebook
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/IntelliComplete.git
```

### 2. Navigate to the project directory

```bash
cd IntelliComplete
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

```
Input Layer
      ↓
Embedding Layer
      ↓
LSTM (128 Units)
      ↓
Dense Layer
      ↓
Softmax Output Layer
```

---

## 📖 Workflow

1. User enters the beginning of a sentence.
2. The text is converted into numerical sequences using the saved tokenizer.
3. Sequences are padded to the required length.
4. The trained LSTM model predicts the most probable next word.
5. The predicted word is appended to the input.
6. The process repeats until the desired number of words is generated.

---

## 📊 Libraries Used

- TensorFlow
- Keras
- Streamlit
- NumPy
- Pickle

---

## 🎯 Future Improvements

- Top-k Sampling
- Temperature Sampling
- Beam Search Decoding
- Transformer-based Language Model
- Hugging Face Integration
- Better UI/UX
- Multi-language Support

---

## 👨‍💻 Author

**Adithya Anand B A**

GitHub: https://github.com/officialadithya2005-maker

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
