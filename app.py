import streamlit as st
import pickle
import string
import spacy

nlp = spacy.load("en_core_web_sm")

def transform_text(text):
    text = text.lower()
    doc = nlp(text)

    transformed_text = [token.text for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(transformed_text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
