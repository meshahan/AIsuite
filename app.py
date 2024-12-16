import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv('GROQ_API_KEY')

if api_key is None:
    raise ValueError("API key is not found in the .env file")

# Set the API key in the environment
os.environ['GROQ_API_KEY'] = api_key

import aisuite as ai

# Initialize the AI client for accessing the language model
client = ai.Client()

# Streamlit UI Enhancements
st.markdown("""
    <style>
    .title {
        color: #0044cc;
        font-size: 25px;
        font-weight: bold;
        text-align: center;
        font-family: 'Arial', sans-serif;
        white-space: nowrap; /* Ensure the title stays in one line */
        background-color: #FFEB3B; /* Yellow background */
        padding:10px;
        border-radius: 5px;
    }
    .subheader {
        color: #0066ff;
        font-size: 24px;
        font-weight: bold;
    }
    .button {
        background-color: #22eff2; /* Light green background */
        color: white;
        padding: 10px 20px; /* Medium button size */
        border: none;
        font-size: 18px;  /* Medium font size */
        cursor: pointer;
        border-radius: 5px;
    }
    .button:hover {
        background-color: #22eff2;
    }
    .input-box {
        background-color: #FFEB3B; /* Yellow background */
        border: 2px solid #ccc;
        padding: 10px;
        border-radius: 5px;
        font-size: 18px;
        width: 100%;  /* Full width */
    }
    .slider {
        background-color: #ADD8E6; /* Light Blue background */
        border-radius: 5px;
        width: 80%;
    }
    .stTextInput, .stSlider {
        width: 100%;  /* Full width */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit Title and Styling
st.markdown('<div class="title">MEDICAL RESEARCH QUERY ASSISTANT</div>', unsafe_allow_html=True)

# Adding dart symbol along with "HARD TARGET"
st.markdown('<div class="title"> DEVELOPED BY :  HARD TARGET 🎯</div>', unsafe_allow_html=True)

# Input prompt from user with increased width
query = st.text_input("Enter Your Query:", "Write Your Text here...", key="query", help="Type your medical query here.", placeholder="E.g. What is diabetes?", label_visibility="collapsed", max_chars=1000)

# Control parameters with Streamlit sliders and input fields, increase the width
col1, col2, col3 = st.columns([2, 2, 2])  # Creating columns for layout

with col1:
    temperature = st.slider("Temperature (controls randomness)", 0.0, 2.0, 1.0, 0.1, key="temperature", help="Higher values produce more random responses.", format="%0.1f")

with col2:
    max_tokens = st.slider("Max Tokens (controls the response length)", 50, 500, 150, 10, key="max_tokens", help="Controls the length of the response.", format="%d")

with col3:
    top_p = st.slider("Top-p (controls diversity of responses)", 0.0, 1.0, 1.0, 0.1, key="top_p", help="Higher values provide more diversity.", format="%0.1f")

# Graphical Representation of Control Parameter Changes
def plot_parameter_changes(temperature, max_tokens, top_p):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plotting a simple visualization of how the parameters may change
    x = np.linspace(0, 2, 100)
    y_temp = np.sin(x) * temperature  # Impact of temperature
    y_tokens = np.cos(x) * (max_tokens / 150)  # Impact of max tokens
    y_top_p = np.sin(x) * top_p  # Impact of top_p

    ax.plot(x, y_temp, label=f"Temperature Impact", color='blue', linewidth=2)
    ax.plot(x, y_tokens, label=f"Max Tokens Impact", color='green', linewidth=2)
    ax.plot(x, y_top_p, label=f"Top-p Impact", color='red', linewidth=2)

    ax.set_title('Impact of Control Parameters on Response Characteristics')
    ax.set_xlabel('Control Parameter (Range: 0 - 2)')
    ax.set_ylabel('Response Intensity')
    ax.legend(loc="upper right")
    st.pyplot(fig)

# Show graphical representation
plot_parameter_changes(temperature, max_tokens, top_p)

# Define a function to get the response from the model with control parameters
def get_medical_answer(query, temperature, max_tokens, top_p):
    # Construct the messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in the medical field."},
        {"role": "user", "content": f"response '{query}'"}
    ]
    
    # Request a response from the model with the control parameters
    response = client.chat.completions.create(
        model="groq:llama-3.2-3b-preview", 
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    # Return the model's response
    return response.choices[0].message.content

# Button to generate the response
if st.button("Get Answer", key="generate_answer", help="Click to get a response to your query.", use_container_width=True):
    answer = get_medical_answer(query, temperature, max_tokens, top_p)
    st.markdown('<div class="subheader">Answer:</div>', unsafe_allow_html=True)
    st.write(answer)
