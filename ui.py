import streamlit as st
import requests
import os

# Path to save the user questions and feedback
questions_file = "user_questions.txt"

# Check if file exists, if not create it
if not os.path.exists(questions_file):
    with open(questions_file, "w") as f:
        f.write("User Questions and Feedback Log\n\n")

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Streamlit app title
st.title("ROBIN (experimental)")

st.write("Queries regarding NTU CS dept. Please remember, your questions are being recorded to make ROBIN better but rest assured we have no way to track who asked the question!")

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Text input from user
user_input = st.text_input("Your question:", key="input_box", value=st.session_state['user_input'])

# Handle submission when the button is clicked
if st.button("Submit"):
    # Check if there's user input
    if user_input:
        with st.spinner("Processing..."):
            # API endpoint (Update with your ngrok URL)
            url = f"https://virtually-talented-monster.ngrok-free.app/llm?input={user_input}"

            # Send request to the FastAPI app
            response = requests.get(url)

            # If the API request is successful
            if response.status_code == 200:
                result = response.json()

                # Display result
                st.write(f"**Input:** {result['input']}")
                st.write(f"**Result:** {result['result']}")

                # Save the user question locally
                with open(questions_file, "a") as f:
                    f.write(f"Question: {user_input}\n")

                # Clear the input box after submission
                st.session_state['user_input'] = ''  # Reset input state after submission

                st.success("Question saved anonymously!")