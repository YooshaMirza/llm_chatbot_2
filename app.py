import pandas as pd
import requests
import json
import streamlit as st
from gradio_client import Client

# Define your Google Gemini API key and endpoint
GEMINI_API_KEY = "AIzaSyA9pYRt95gwUm3UvoZTy30PQ0P65F8niYA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Load your CSV file into a pandas DataFrame
df = pd.read_csv("Medicine_Details.csv")

# Initialize the Gradio client
client = Client("ruslanmv/Medical-Llama3-v2")

# Function to search for medicine details in the CSV dataset
def search_in_csv(medicine_name):
    result = df[df['Medicine Name'].str.lower().str.contains(medicine_name.strip().lower())]
    if not result.empty:
        medicine_info = result.iloc[0]
        response = f"Here are the details I found for {medicine_info['Medicine Name']}:\n\n" \
                   f"Composition: {medicine_info['Composition']}\n" \
                   f"Uses: {medicine_info['Uses']}\n" \
                   f"Side Effects: {medicine_info['Side_effects']}\n" \
                   f"Manufacturer: {medicine_info['Manufacturer']}\n" \
                   f"Reviews:\n" \
                   f"  - Excellent: {medicine_info['Excellent Review %']}%\n" \
                   f"  - Average: {medicine_info['Average Review %']}%\n" \
                   f"  - Poor: {medicine_info['Poor Review %']}%\n" \
                   f"[Image of {medicine_info['Medicine Name']}]({medicine_info['Image URL']})"
        return response
    return None

# Function to generate a response using Gradio's Medical-Llama3 API
def fetch_from_llama3(message):
    try:
        result = client.predict(
            message=message,
            system_message="You are a Medical AI Assistant. Please be thorough and provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.",
            max_tokens=512,
            temperature=0.8,
            top_p=0.9,
            api_name="/chat"
        )
        return result
    except Exception as e:
        return f"Error generating response from model: {e}"

# Function to fetch information from Google Gemini
def fetch_from_gemini(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        try:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "I'm sorry, but I couldn't retrieve the information you requested."
        except json.JSONDecodeError:
            return "I'm sorry, there was an error processing your request."
    else:
        return f"Error fetching information from Google Gemini: {response.status_code} - {response.text}"

# Function to handle the submission
def submit_data():
    if 'input_text' in st.session_state and st.session_state.input_text:
        user_input = st.session_state.input_text

        if user_input.strip().lower() == 'exit':
            st.write("Thank you for using the assistant. Stay healthy!")
            st.session_state.input_text = ""
            return

        # Add the user's query to the conversation
        st.session_state.conversation.append(f"<div class='user-message'><strong>You:</strong> {user_input}</div>")

        # First check the CSV dataset
        csv_response = search_in_csv(user_input)
        if csv_response:
            st.session_state.conversation.append(f"<div class='assistant-message'><strong>Assistant:</strong> {csv_response}</div>")
        else:
            # If not found in CSV, fetch from Medical-Llama3 via Gradio API
            llama3_response = fetch_from_llama3(user_input)
            if llama3_response:
                st.session_state.conversation.append(f"<div class='assistant-message'><strong>Assistant:</strong> {llama3_response}</div>")
            else:
                # If not found in Llama3, fetch from Google Gemini
                st.session_state.conversation.append("<div class='assistant-message'><strong>Assistant:</strong> I couldn't find details for this medicine in the dataset or model, so I'm fetching information from Google Gemini...</div>")
                response = fetch_from_gemini(user_input)
                st.session_state.conversation.append(f"<div class='assistant-message'><strong>Assistant:</strong> {response}</div>")

        # Clear input field
        st.session_state.input_text = ""

# Streamlit app
def main():
    st.set_page_config(page_title="Medicine Information Chatbot", page_icon="💊", layout="wide")

    # Sidebar for app title and instructions
    st.sidebar.title("Medicine Information Chatbot 💬")
    st.sidebar.write("Enter your query below and get information about medicines. The conversation history will be preserved until you close the bot.")

    # Initialize or retrieve the context from session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Call the submission handler before rendering the input box
    submit_data()

    # Apply custom CSS based on theme
    st.markdown(
        """
        <style>
        .user-message {
            color: blue;
        }
        .assistant-message {
            color: var(--text-color);
        }
        body {
            background-color: var(--background-color);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Chat container with scrollable history
    chat_container = st.container()

    # Display the conversation history
    with chat_container:
        for chat in st.session_state.conversation:
            st.markdown(chat, unsafe_allow_html=True)

    # Input box for the user query
    st.text_input("Type your message here:", value="", key="input_text", on_change=submit_data)

if __name__ == '__main__':
    main()
