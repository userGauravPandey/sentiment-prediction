import streamlit as st
import pickle

# 🎨 Page Config
st.set_page_config(page_title="Emotion Prediction App", page_icon="💬", layout="centered")

# 🧠 Load Model and Vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"error model:{e}")
# 🏷️ Title and Description
st.title("💬 Emotion Prediction App")
st.markdown("""
Welcome to the *Emotion Prediction Web App*!  
Enter any text below and find out whether it's expressing *Joy, **Sadness, **Anger, **Love*, or more.  
Built using *Logistic Regression* and *TF-IDF Vectorization*.
""")

# ✍️ User Input
user_input = st.text_area("Enter your text here:", height=150, placeholder="Type something like 'I am feeling happy today!'")

# 🔍 Prediction Button
if st.button("🔮 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Transform input
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        # 🧠 Emotion Display with Emojis
        emoji_map = {
            "joy": "😄",
            "sadness": "😢",
            "anger": "😡",
            "love": "❤️",
            "fear": "😨",
            "surprise": "😲"
        }
        emoji = emoji_map.get(prediction.lower(), "🙂")

        st.success(f"*Predicted Emotion:* {prediction.capitalize()} {emoji}")

# ✨ Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")