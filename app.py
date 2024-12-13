import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask import Flask, request, jsonify, render_template

# Define the custom AttentionLayer class
class AttentionLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        query, value = inputs
        attention_scores = tf.matmul(query, value, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.matmul(attention_scores, value)
        return context

# Load the saved model with custom layer
with custom_object_scope({'AttentionLayer': AttentionLayer}):
    model = load_model("enhanced_sentiment_analysis_model.h5")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open("label_encoder.pickle", "rb") as handle:
    label_encoder = pickle.load(handle)

# Define maximum length and other preprocessing constants
max_len = 350

def predict_sentiment(review):
    """Predict the sentiment for a single review.

    Args:
        review (str): A string representing the review text.

    Returns:
        str: The predicted sentiment label.
    """
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)
    sentiment = label_encoder.inverse_transform([np.argmax(prediction)])
    return sentiment[0]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page with the input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle the form submission and return the prediction."""
    review = request.form['review']
    if not review:
        return render_template('index.html', error="Please enter a review.")

    sentiment = predict_sentiment(review)
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)