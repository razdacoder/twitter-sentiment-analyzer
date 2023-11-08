import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
import string
nltk.download('stopwords')


stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

# Load your pre-trained NLTK model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

# Function to analyze sentiment
def sentiment_analyzer(text: str):
    text_val = [clean_text(text=text)]
    test_vect = vectorizer.transform(text_val)
    pred = model.predict(test_vect)
    print("Text:", text)
    if pred<0.5:
        return "Not Abusive Sentiment"
    else:
        return "Hate and Abusive Sentiment"

# Streamlit app
st.title("Text Sentiment Analyzer")

# Input for the user to enter text
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        st.write("Analyzing sentiment...")
        sentiment = sentiment_analyzer(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text for analysis.")
