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

import GenerateQuestions
import VectorRetreiver
import MultiModalRAG


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
        return jsonify(qa_pairs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/query', methods=['POST'])
def query_vector_db():
    if 'collection_id' not in request.form:
        return jsonify({'error': 'No collection ID provided'}), 400
    collection_id = request.form['collection_id']
    if 'query' not in request.form:
        return jsonify({'error': 'No query provided'}), 400
    vectorstore = Chroma(
        client=client,
        collection_name=collection_id,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    retriever = VectorRetreiver.create_multi_vector_retriever(vectorstore, docstore)
    query = request.form['query']
    rag_chain = VectorRetreiver.multi_modal_rag_chain(retriever)
    final_response = rag_chain.invoke(query)
    return jsonify(final_response)


@app.route("/receive-sms", methods=['GET', 'POST'])
def sms_reply():    # Start our TwiML response
    response = MessagingResponse()
    inbound_message = request.form.get("Body")
    if inbound_message == "Hello":
        response.message("Hello back to you!")
    else:
        response.message("Hi! Not quite sure what you meant, but okay.")
    return Response(str(response), mimetype="application/xml"), 200

@app.route("/send-sms", methods=['GET', 'POST'])
def send_sms():
    try:
        twilio.messages.create(
        from_='+18449434713',
        body='Ahoy world, from Python', 
        to='+15103345535' #swap for students numbers later
    )
        return "success"
    except Exception as e:
        print(e);
        return "error"

if __name__ == '__main__':
    app.run()