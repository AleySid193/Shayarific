import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load max_sequence_len
with open('max_sequence_len.pkl', 'rb') as f:
    max_sequence_len = pickle.load(f)

# Load trained model
model = tf.keras.models.load_model('poetry_gru.keras')

# Function to generate text
def generate_text(seed_text, next_words):
    for _ in range(next_words): 
        token_list = tokenizer.texts_to_sequences([seed_text])[0] 
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') 
        
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1) 
        output_word = "" 

        for word, index in tokenizer.word_index.items(): 
            if index == predicted: 
                output_word = word
                break
        
        seed_text += " " + output_word 

    return seed_text.replace('newline', '\n')


# Streamlit UI
st.title("Shayarific 🎶")
st.write("The magic of Shayari, now generated for you! ✨")
st.write("Enter a starting phrase and select the number of words to generate a poem!")

# User input
seed_text = st.text_input("Enter the starting words:", "mohabbat barsa dena tu")
next_words = st.slider("Select the number of words to generate:", min_value=5, max_value=100, value=50)

# Generate poem button
if st.button("Generate Poetry"):
    generated_poem = generate_text(seed_text, next_words)
    st.subheader("Generated Poetry:")
    st.text_area("", generated_poem, height=250)

# Footer
st.markdown("---")
st.markdown("🚀 *Powered by TensorFlow & Streamlit*")
