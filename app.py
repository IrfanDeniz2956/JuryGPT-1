import streamlit as st
import joblib
from gemini_helper import analyze_speech_with_gemini

# Modelleri yükleme
argument_model = joblib.load(r"C:\Users\Acer\Desktop\JuryGPT\JuryGPT-main\models\argumentmodel.pkl")
language_model = joblib.load(r"C:\Users\Acer\Desktop\JuryGPT\JuryGPT-main\models\language_model.pkl")
debate_model = joblib.load(r"C:\Users\Acer\Desktop\JuryGPT\JuryGPT-main\models\debate_model.pkl")

st.title("Debate AI Evaluator (Gemini Enhanced)")

# Kullanıcıdan giriş al
text_input = st.text_area("Enter Debate Speech:")

if st.button("Evaluate"):
    argument_score = argument_model.predict([text_input])[0]
    language_score = language_model.predict([text_input])[0]
    debate_score = debate_model.predict([text_input])[0]

    # Gemini API'yi çağır
    gemini_analysis = analyze_speech_with_gemini(text_input)

    st.subheader("AI Model Evaluation:")
    st.write(f"**Argument Strength:** {'Strong' if argument_score == 1 else 'Weak'}")
    st.write(f"**Language Fluency:** {'Fluent' if language_score == 1 else 'Needs Improvement'}")
    st.write(f"**Debate Counter-Arguments:** {'Effective' if debate_score == 1 else 'Ineffective'}")

    st.subheader("Gemini AI Analysis:")
    st.write(gemini_analysis)
