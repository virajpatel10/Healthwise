import gradio as gr
import random
import time
import json
import numpy as np
from keras.models import load_model
import preprocessing
import re
import joblib
from model import label_encoder
import os
import pytesseract
import random
import pandas as pd
try:
    from PIL import Image
except ImportError:
    import Image
from pdf2image import convert_from_path


file_path = 'Drug.csv'
drug = pd.read_csv(file_path)

vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load data from JSON
with open('messages.json', 'r') as f:
    data = json.load(f)

disease_advice = data["disease_advice"]
class_names = list(disease_advice.keys())
greetings = data["greetings"]
responses = data["responses"]
goodbyes = data["goodbyes"]
goodbye_replies = data["goodbye_replies"]

model = load_model('disease_model.h5')

# Load JSON data (e.g., symptoms and class mappings)
with open('disease_data.json', 'r') as f:
    disease_data = json.load(f)

class_names = disease_data['class_names']
advices = disease_data['advice']


# Define a function to predict disease from user-provided symptoms
def predict_disease(user_text):
    # Step 1: Clean the user-provided text
    cleaned_text = preprocessing.clean_text(user_text)
    
    # Step 2: Vectorize the cleaned text using the trained vectorizer
    user_vector = vectorizer.transform([cleaned_text]).toarray()
    user_vector = user_vector.reshape(user_vector.shape[0], 1, user_vector.shape[1])  # Reshape for RNN compatibility
    
    # Step 3: Predict the label
    prediction = model.predict(user_vector)
    predicted_label = np.argmax(prediction, axis=1)  # Get the index of the max probability
    print("Predicted Label Index:", predicted_label)
    # Step 4: Decode the label to the original category
    disease = label_encoder.inverse_transform(predicted_label)
    print("Decoded Disease:", disease)
    return disease[0]


# Chatbot response function
def respond(message, chat_history):
    message_lower = message.lower()
    try:
        if message_lower in greetings:
            bot_message = random.choice(responses)
        elif message_lower in goodbyes:
            bot_message = random.choice(goodbye_replies)
        else:
            disease = predict_disease(message)
            d = drug.loc[drug['Disease']==disease]['Drug']
            drug_names = d.tolist()
            for name in drug_names:
                drug_name = name
            advice = advices.get(disease)
            print(advice)
            bot_message = f"The predicted condition is: {disease}. Recommended drug: {drug_name}. Here is some advice: {advice}. Please consult a doctor for further guidance."
    except Exception as e:
        bot_message = "I'm sorry, I don't have advice for that. Could you rephrase or provide more details?"
    
    chat_history.append((message, bot_message))
    time.sleep(5)
    return "", chat_history

# Define the folder to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist



def extract_section(text, section_name):
    pattern = rf"{section_name}\s*[:\-]\s*(.+?)(?=\s*[\n\r]+[A-Z])"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "Not Found"


def extraction_text_from_image():
    extracted_text = ""
    for filename in os.listdir('uploads'):
        extractedInformation = pytesseract.image_to_string(Image.open('uploads/'+filename))
        extracted_text += extractedInformation + "\n"
    
    comments = extract_section(extracted_text, "Comments")
    advice = extract_section(extracted_text, "Advise")
    final_diagnosis = extract_section(extracted_text, "Final Diagnosis")

    summary = "Comments : "+comments+"\n"+"Advise : "+advice+"\n"+"Final Diagnosis : "+final_diagnosis

    return summary

# Updated file upload handling function
def handle_file_upload(file, chat_history):
    if not file:  # Check if no file is uploaded
        bot_message = "No file uploaded. Please upload a valid file."
    else:
        try:
            print(file.name)
            # Save the uploaded file to the specified folder
            images = convert_from_path(file.name)
            
            # Save each page as an image
            for i, image in enumerate(images):
                image_filename = 'uploads/'
                image_filename += f"page_{i+1}.jpg"  # Save with page number
                image.save(image_filename, "JPEG")
                print(f"Saved: {image_filename}")

            report_information = extraction_text_from_image()

            bot_message = report_information
        except Exception as e:
            bot_message = f"An error occurred while processing the file: {e}"
    
    chat_history.append(("Uploaded file", bot_message))
    return "", chat_history

# Update the chatbot UI for file upload
with gr.Blocks(css="""
    #col_container { margin-left: auto; margin-right: auto; }
    #chatbot { height: 520px; overflow: auto; }
""") as demo:
    gr.HTML("<h1>Medical Chatbot: Your Virtual Health Guide 🌟🏥🤖</h1>")
    with gr.Accordion("Follow these steps to use the chatbot", open=True):
        gr.HTML("""<p>
        1. Type your message or symptoms in the textbox below.<br>
        2. Upload a file containing symptoms for analysis.<br>
        3. The chatbot will respond with possible advice or recommendations.<br>
        4. To clear the chat, click the clear button.
        </p>""")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your message here...")
    upload = gr.File(label="Upload a file with symptoms or text")
    clear = gr.ClearButton([msg, chatbot])
    
    # Link input components to their respective functions
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    upload.change(handle_file_upload, [upload, chatbot], [msg, chatbot])

# Launch the chatbot
demo.launch()