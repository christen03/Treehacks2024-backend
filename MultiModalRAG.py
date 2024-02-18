from langchain.text_splitter import TokenTextSplitter
from unstructured.partition.pdf import partition_pdf
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import datetime
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from chromadb.config import Settings
import chromadb
import base64
from langchain_core.messages import HumanMessage
import uuid


# slides_collection = client.create_collection("slides")


input_path = os.getcwd()

openai_api_key = os.getenv("OPENAI_API_KEY")


def upload_slide(vectorstore, docstore, local_file_path):
#     """
#     Upload slide to Chroma
#     """

# # File path
    unique_output_path = os.path.join(os.getcwd(), "figures")
# Get elements
    raw_pdf_elements = extract_pdf_elements(local_file_path, unique_output_path)

# Get text, tables
    texts, tables = categorize_elements(raw_pdf_elements)

    ques_gen = ''
    for text in texts:
        ques_gen = ques_gen + text

    text_splitter = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size=1250, 
        chunk_overlap=200
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)


    text_summaries, table_summaries = generate_text_summaries(
    texts_4k_token, tables, summarize_texts=True
)
    print("TEXT SUMMARIES",text_summaries)
    
    img_base64_list, image_summaries = generate_img_summaries(unique_output_path)
    print("IMAGE SUMMARIES",image_summaries)

    add_documents_to_stores(
        vectorstore=vectorstore,
        docstore=docstore,
        text_summaries=text_summaries,
        texts=texts_4k_token,
        table_summaries=table_summaries,
        tables=tables,
        image_summaries=image_summaries,
        images=img_base64_list,
    )
    return [texts, table_summaries, image_summaries]
def extract_pdf_elements(local_file_path, unique_output_path):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    print("Extracting...")
    return partition_pdf(
        filename=os.path.join(os.getcwd(), local_file_path),
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        output_dir_path=unique_output_path,
    )
    

def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=True):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a DEATAILED summary of the table or text that is well optimized for retrieval. Focus on including \
    additional details that would help a student who has never seen this content before learn and understand this content. This \
    will later be used to help reinforce a student's understanding of the content. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4-vision-preview", openai_api_key=openai_api_key, max_tokens=1024)
    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
    except Exception as e:
        return None
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


# Image summaries


def add_documents_to_stores(vectorstore, docstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Add documents to the vectorstore and docstore
    """
    # Add texts
    if text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        vectorstore.add_documents(summary_docs)
        docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    if table_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in tables]
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i, s in enumerate(table_summaries)
        ]
        vectorstore.add_documents(summary_docs)
        docstore.mset(list(zip(doc_ids, tables)))

    # Add images
    if image_summaries:
    # Filter out None values from image_summaries and keep track of their indices
        filtered_summaries_with_indices = [(i, s) for i, s in enumerate(image_summaries) if s is not None]

        # Instead of pre-generating doc_ids based on filtered summaries, generate them dynamically in the loop
        summary_docs = []
        filtered_images = []

        for i, s in filtered_summaries_with_indices:
            doc_id = str(uuid.uuid4())  # Generate a new doc_id for each valid summary
            summary_docs.append(Document(page_content=s, metadata={"doc_id": doc_id}))
            filtered_images.append(images[i])  # Append the corresponding image based on the original index

        # Add the valid documents to the vectorstore
        vectorstore.add_documents(summary_docs)

        # Add the corresponding images to the docstore, using the newly generated doc_ids
        # docstore.mset(list(zip([doc.metadata["doc_id"] for doc in summary_docs], filtered_images)))
        # Assuming this is the line where you add items to the docstore
        docstore.mset(list(zip([doc.metadata["doc_id"] for doc in summary_docs], filtered_images)))

        # To verify, retrieve the items using the same document IDs
        doc_ids = [doc.metadata["doc_id"] for doc in summary_docs]
        retrieved_images = docstore.mget(doc_ids)

        # Print out the retrieved items to verify
        # for doc_id, image in zip(doc_ids, retrieved_images):
        #     print(f"Doc ID: {doc_id}, Image: {image[:30]}...")  # Print a portion of the base64 image string for brevity


# Create retriever
# retriever_multi_vector_img = create_multi_vector_retriever(
#     vectorstore,
#     text_summaries,
#     texts,
#     table_summaries,
#     tables,
#     image_summaries,
#     img_base64_list,
# )

