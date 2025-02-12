from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr

# Load my thesis
loader = PDFPlumberLoader("/home/tonny/Projects/hello_langchain/data/atheys-EndToEndLearningToPredictArcheologicalPotential-1-3.pdf")
docs = loader.load()

# Split into chunks
text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store for efficient similarity search
vector = FAISS.from_documents(documents, embedder)

# Convert to a retriever that fetches relevant documents
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = Ollama(model="mistral")

# Define prompt template
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

# Init prompt template
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

# Create the chain
llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Combine retrieved documents into a single context for the LLM
combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

# Create a retrieval-based QA system using the LLM and document retriever
qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )

# Define the function that takes user input and returns the chatbot's response
def respond(question,history):
    return qa(question)["result"]

# Launch gradio app
gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to my thesis", container=False, scale=7),
    title="Thesis Chatbot",
    examples=["Whats it about?"],
    cache_examples=True,
    retry_btn=None,

).launch(share = True)