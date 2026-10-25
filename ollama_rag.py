import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import gradio as gr

# Load documents
#loader = pyPDFLoader("GPU Droplet GA FAQ.pdf")
#documents = loader.load()
from langchain_community.document_loaders import PyPDFLoader

# Resolve relative to this script so it works from any working directory
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rd module notes.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

# Create embeddings
embeddings = GPT4AllEmbeddings()

# Create a vector store
vector_store = Chroma.from_documents(documents, embeddings)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Given the context: {context}, answer the question: {question}"
)

# Initialize the LLM with gemma mode
llm = Ollama(model="gemma")

# Function to perform RAG
def rag_system(question):
    # Retrieve relevant documents
    context = vector_store.similarity_search(question, k=3)
    
    # Format the context for the prompt
    context_text = " ".join([doc.page_content for doc in context])
    
    # Generate a response using the LLM
    response = llm.generate([prompt_template.format(context=context_text, question=question)])
    
    # Extract the text from the LLMResult object
    if response.generations:
        answer = response.generations[0][0].text  # Access the first generation's text
    else:
        answer = "No answer found."
    
    return answer

# Gradio interface
def gradio_interface(question):
    return rag_system(question)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="RAG System",
    description="Ask a question and get an answer based on the document context."
)

# Launch the Gradio app
iface.launch()
