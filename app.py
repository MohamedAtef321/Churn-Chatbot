# --------------------------- API ---------------------------

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import requests
import re
import os
from huggingface_hub import InferenceClient
import json
import streamlit as st
import threading

app = Flask(__name__)
with open('churn_normalizer.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load your churn model
model = load_model('churn_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_data = pd.DataFrame([input_data])
    try :
      data = input_data.drop(["customerID"], axis=1)
    except :
      data = input_data
      data["customerID"] = "unknown"
    print(f"data :\n\n{data}")
    data = pd.get_dummies(data).astype(float)
    data = data.reindex(columns=list(scaler.feature_names_in_))
    data = data.fillna(0).astype(float)
    data = scaler.transform(data)
    predictions = model.predict(data)
    predictions = pd.DataFrame(predictions, columns=['Churn'])
    predictions['Churn'] = predictions['Churn'].apply(lambda x: "Yes" if x > 0.5 else "No")
    predictions = pd.concat([predictions, input_data['customerID']], axis=1)
    predictions = predictions[["customerID", "Churn"]]
    response = predictions.to_dict(orient='records')
    # response = [{"customerID": p["customerID"], **{k: v for k, v in p.items() if k != "customerID"}} for p in predictions.to_dict(orient='records')]
    print(f"prediction response : {response}")
    return jsonify({'predictions': response})

# --------------------------------------------------------------------------

hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Load the open-source data extraction model
model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Choose "flan-t5-xxl" for a larger model

# Prediction API endpoint
PREDICTION_API_URL = "http://127.0.0.1:8000/predict"  # Replace with your API UR

client = InferenceClient(api_key=hf_token)

# Function to extract data
def extract_customer_data(input_text):
    prompt = f"""
        You are an intelligent assistant tasked with extracting structured data from a text.
        Use options between [] seperated by ',' with same cases.
        reply only according to the input data. reply only with the josn code. answer with only the json object and nothing else.
        choose only one option or choose null for the field that you do not know. write only json format.
        don't start it with null.
        choose only one option for each field, don't create a list for any fields.

        The information to extract includes the following fields:
        - customerID [ID of customer]
        - gender [Female, Male]
        - 'Senior_Citizen' [0, 1]
        - Is_Married [Yes, No]
        - Dependents [Yes, No]
        - tenure
        - Phone_Service [Yes, No]
        - Dual [No phone service, No, Yes]
        - Internet_Service [DSL, Fiber optic, No]
        - Online_Security [No, Yes, No internet service]
        - Online_Backup [No, Yes, No internet service]
        - Device_Protection [No, Yes, No internet service]
        - Tech_Support [No, Yes, No internet service]
        - Streaming_TV [No, Yes, No internet service]
        - Streaming_Movies [No, Yes, No internet service]
        - Contract [Month-to-month, One year, Two year]
        - Paperless_Billing [Yes, No]
        - Payment_Method [Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)]
        - Monthly_Charges
        - Total_Charges

        Given the input text, extract the data and return it as a valid JSON object. Ensure the response includes all the fields even if the data is missing (use `null` for missing fields).

        Input:
        {input_text}

        Output:
        """

    pattern = r"\{(.*?)\}"
    messages = [
      {
        "role": "user",
        "content": prompt
      }
    ]
    
    completion = client.chat.completions.create(model="meta-llama/Llama-3.2-1B-Instruct", messages=messages, max_tokens=2048)
    answer = completion.choices[0].message.content
    match = re.search(pattern, answer)
    if match:
        response = match.group(1)
        response = "{"+response+"}"
    else:
        response = answer
    response = response.replace("```json", "").replace("```", "")
    response = response.replace("'", '"')
    print(f"response:\n{response}")
    # if "[" in response:
    #     response =  "{"+response+"}"
    response = json.loads(response)
    return response



@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    extracted_data = extract_customer_data(message)
    predictions = requests.post(PREDICTION_API_URL, json=extracted_data).json()

    prompt = f"""
    User asked if the customer has churned or not. we have made a predictor that can predict this churn.
    Respond to the user directly and shortly according to his message and the predictions.
    Respond to user about each customer status. respond in english.
    Talk to the user directly.
    You are only AI assistant, write a simple and short answer.

    User Message :
    {message}

    Predictions :
    {predictions}

    """

    messages = [
      {
        "role": "user",
        "content": prompt
      }
    ]
    
    completion = client.chat.completions.create(model="meta-llama/Llama-3.2-1B-Instruct", messages=messages, max_tokens=2048)
    response = {"response": completion.choices[0].message.content}
    # response = requests.post(PREDICTION_API_URL, json={"input_text": extracted_data})
    return response








if __name__ == '__main__':
    
    # ----------------------------------  ChatBot ----------------------------------
    
    # Function to run Flask in a thread
    def run_flask():
        app.run(host='0.0.0.0', port=8000)

    # Start Flask in a background thread
    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()

    # API endpoint
    CHAT_API_URL = "http://127.0.0.1:8000/chat"  # Replace with your API URL

    # Streamlit App Title
    st.title("🙋‍♂️ Churn LLM ChatBot Interface")

    # Sidebar Info
    # st.sidebar.title("Instructions")
    # st.sidebar.info(
    #     "This is a simple chatbot GUI. Type your message below and the bot will respond!"
    # )

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input box
    user_input = st.text_input("You:", key="input", placeholder="Type your message here...")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            # Add user's message to session state
            st.session_state.messages.append({"user": "You", "message": user_input})

            # Call the chatbot API
            try:
                response = requests.post(
                    CHAT_API_URL, json={"message": user_input}
                )
                response.raise_for_status()
                bot_reply = response.json().get("response", "No response received.")
            except requests.exceptions.RequestException as e:
                bot_reply = f"Error: {e}"

            # Add bot's response to session state
            st.session_state.messages.append({"user": "Bot", "message": bot_reply})

    # Display Chat History
    for msg in st.session_state.messages:
        if msg["user"] == "You":
            st.markdown(f"**>> You:** {msg['message']}")
        else:
            st.markdown(f"**>> Bot:** {msg['message']}")