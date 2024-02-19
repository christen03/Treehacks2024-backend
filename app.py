from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import chromadb
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from werkzeug.utils import secure_filename
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import tempfile
import requests
import random
import requests

import GenerateQuestions
import VectorRetreiver
import MultiModalRAG
import convexdb
import TF_IDF


app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False  
CORS(app)
load_dotenv()

chroma_api_key = os.getenv("CHROMA_API_KEY")
chroma_server_host = os.getenv("CHROMA_SERVER_HOST")
chroma_server_port = os.getenv("CHROMA_SERVER_PORT")
openai_api_key = os.getenv("OPENAI_API_KEY")
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")

client = chromadb.HttpClient(
    host = chroma_server_host,
    port = chroma_server_port,
    headers = {"X-Chroma-Token": chroma_api_key}
)
twilio = Client(twilio_account_sid, twilio_auth_token)

docstore = InMemoryStore()

def send_query_to_api(inbound_message):
    url = "https://ta.ai.ngrok-free.app/query"
    collection_id = "js789ywt8bt2eq1aqsqvjjr5rn6kprx2"  # Replace with your static collection ID
    payload = {
        "collection_id": collection_id,
        "query": inbound_message,
    }
    headers = {
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        data = response.json()
        print(data)
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()  # Raise an exception for error status codes

    with open('temp_download.pdf', 'wb') as f:
        f.write(response.content)

    return 'temp_download.pdf'

def save_file_locally(file, filename):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    # Construct the file path
    file_path = os.path.join(temp_dir, secure_filename(filename))
    # Save the file to the constructed path
    file.save(file_path)
    # Return the file path for further processing
    return file_path

@app.route('/embed', methods=['POST'])
def upload_slide():
    # Use request.get_json() to access the JSON data sent in the request body
    data = request.get_json()

    # Access 'collection_id' and 'file' from the JSON data
    if 'collection_id' not in data:
        return jsonify({'error': 'No collection ID provided'}), 400
    collection_id = data['collection_id']

    if 'file' not in data:
        return jsonify({'error': 'No PDF URL provided'}), 400
    pdf_url = data['file']

    try:
        local_file_path = download_pdf(pdf_url)
        vectorstore = Chroma(
            client=client,
            collection_name=collection_id,
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
        question_vectorstore = Chroma(
            client=client,
            collection_name=collection_id+"questions",
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
        data = MultiModalRAG.upload_slide(vectorstore, docstore, local_file_path)
        qa_pairs = GenerateQuestions.generate_qa_pairs(data[0], data[1], data[2], question_vectorstore)
        feature_names = TF_IDF.extract_feature_names(qa_pairs)
        return jsonify({"qa_pairs": qa_pairs, "feature_names": feature_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/pdf', methods=['POST'])
def upload_pdf():
    if 'collection_id' not in request.form:
        return jsonify({'error': 'No collection ID provided'}), 400
    collection_id = request.form['collection_id']

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        local_file_path = save_file_locally(file, filename)
        vectorstore = Chroma(
            client=client,
            collection_name=collection_id,
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
        question_vectorstore = Chroma(
            client=client,
            collection_name=collection_id+"questions",
            embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
    try:
        data = MultiModalRAG.upload_slide(vectorstore, docstore, local_file_path)
        qa_pairs = GenerateQuestions.generate_qa_pairs(data[0], data[1], data[2], question_vectorstore)
        feature_names = TF_IDF.extract_feature_names(qa_pairs).tolist()
        print(type(qa_pairs))
        print(type(feature_names))
        print(feature_names)
        return jsonify({"qa_pairs": qa_pairs, "feature_names": feature_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/query', methods=['POST'])
def query_vector_db():
    data = request.get_json()

    if 'collection_id' not in data:
        return jsonify({'error': 'No collection ID provided'}), 400
    collection_id = data['collection_id']

    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    query_text = data['query']
    vectorstore = Chroma(
        client=client,
        collection_name=collection_id,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    retriever = VectorRetreiver.create_multi_vector_retriever(vectorstore, docstore)
    rag_chain = VectorRetreiver.multi_modal_rag_chain(retriever)
    final_response = rag_chain.invoke(query_text)
    return jsonify(final_response)


@app.route("/receive-sms", methods=['GET', 'POST'])
def sms_reply():   
    response = MessagingResponse()
    inbound_message = request.form.get("Body")
    sender_phone_number = request.form.get("From").replace('+', '').replace('-', '')
    user = convexdb.get_student_from_phone_number(sender_phone_number)
    user_id = user['_id']
    collection_id = user['collection_id']
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_id,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    question_vectorstore = Chroma(
        client=client,
        collection_name=collection_id+"questions",
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    summary_score = vectorstore.similarity_search_with_score(inbound_message)[0][1]
    question_score = question_vectorstore.similarity_search_with_score(inbound_message)[0][1]
    total_score = (0.75)*question_score + (0.25)*summary_score

    response = MessagingResponse()
    inbound_message = request.form.get("Body")
    # Call the function with the inbound message
    api_response = send_query_to_api(inbound_message)
    twilio.messages.create(
                body=api_response,
                from_='+18449434713',
                to=sender_phone_number
            )
    # convexdb.increment_texts_responded_and_score(user_id, total_score)
    # if(convexdb.get_number_of_texts_responded(user_id) == 3):
        


    return Response(str(response), mimetype="application/xml"), 200


@app.route("/send-sms", methods=['POST']) 
def send_sms():
    data = request.get_json() 

    if 'studentPhoneNumbers' not in data:
        return jsonify({'error': 'No studentPhoneNumbers provided'}), 400

    studentPhoneNumbers = data['studentPhoneNumbers']
    print(studentPhoneNumbers)
    print(convexdb.get_student_from_phone_number(studentPhoneNumbers[0]))

    if not isinstance(studentPhoneNumbers, list):
        return jsonify({'error': 'studentPhoneNumbers must be an array'}), 400

    if "qa_pairs" not in data:
        return jsonify({'error': 'No qa_pairs provided'}), 400
    qa_pairs = data["qa_pairs"]

    for number in studentPhoneNumbers:
        student_obj = convexdb.get_student_from_phone_number(number)
        if not student_obj:
            continue
        else:
            indexes_seen = student_obj["indexesSeen"]
            all_indexes = list(range(len(qa_pairs)))

            unseen_indexes = [index for index in all_indexes if index not in indexes_seen]
            twilio.messages.create(
                body="Hey there! Thanks for coming to class today. Get excited for some review! ",
                from_='+18449434713',
                to=number  # Use the number from the array
            )
            # if unseen_indexes:
            random_index = random.choice(unseen_indexes)
        try:
            twilio.messages.create(
                body=qa_pairs[random_index]["Question"],
                from_='+18449434713',
                to=number  # Use the number from the array
            )
        except Exception as e:
            print(e)
            continue  # Continue sending to the next number even if one fails

    return jsonify({"message": "SMS sent to all student phone numbers"}), 200


@app.route("/test-sms", methods=['POST'])
def test_sms():
    student_obj = convexdb.get_student_from_phone_number("5712097011")
    return jsonify({"message": "SMS sent to all student phone numbers"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050) 