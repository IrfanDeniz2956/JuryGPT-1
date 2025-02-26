from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import data
import joblib

# 📌 Train-Test Split
X_train, X_test, y_train_arg, y_test_arg = train_test_split(data["text"], data["argument_strength"], test_size=0.2, random_state=42)
X_train, X_test, y_train_lang, y_test_lang = train_test_split(data["text"], data["language_skills"], test_size=0.2, random_state=42)
X_train, X_test, y_train_debate, y_test_debate = train_test_split(data["text"], data["debate_skills"], test_size=0.2, random_state=42)

# 📌 3 Model Oluştur
argument_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
language_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
debate_model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 📌 Modelleri Eğit
argument_model.fit(X_train, y_train_arg)
language_model.fit(X_train, y_train_lang)
debate_model.fit(X_train, y_train_debate)


joblib.dump(argument_model, "models/argument_model.pkl")
joblib.dump(language_model, "models/language_model.pkl")
joblib.dump(debate_model, "models/debate_model.pkl")

def load_model(filename):
    import os
    model_path = os.path.join(os.getcwd(), "models", filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

# Modelleri yükle
argument_model = load_model("argument_model.pkl")
speech_classification_model = load_model("speech_classification_model.pkl")
debate_scoring_model = load_model("debate_scoring_model.pkl")

