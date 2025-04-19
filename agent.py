from langchain.agents import initialize_agent, AgentType
from openai import OpenAI
from langchain.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



def create_agent_with_tools(llm, pdf_path=None):
    """
    Create an agent with tools for PDF retrieval and conversational memory.
    Args:
        llm (ChatOpenAI): The language model to use.
        pdf_path (str): Path to the PDF file for retrieval.
    Returns:
        ConversationalRetrievalChain: The agent with tools.
    """
    try:
        # Check if the LLM is provided
        if llm is None:
            raise ValueError("LLM must be provided")
        

        # LLM without pdf retrieval
        if  pdf_path is None:
            memory = ConversationBufferMemory(memory_key="chat_history", retriever=None, return_messages=True, input_key="question", output_key="answer"  )
            conversational_chain  = ConversationalRetrievalChain.from_llm(
            llm,
            memory=memory,
            return_source_documents=False,
           verbose=False)
        
            return conversational_chain 
        
        # LLM with pdf retrieval

        ## Load the PDF and split it into chunks
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        ## Create the embeddings and vector store
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda:0"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectordb = Chroma.from_documents(chunks, embeddings, collection_name="pdf_chat")

        ### Create the retriever
        retriever = vectordb.as_retriever()

        ## Create memory and conversational chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"  )
        conversational_chain  = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
           verbose=False
        )
        
        return conversational_chain 
    except Exception as e:
        print(f"Error creating agent: {e.__str__()}")
        return None
