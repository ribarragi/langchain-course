import os
from dotenv import load_dotenv
# import txt loader
from langchain_community.document_loaders import TextLoader
# import text splitter
from langchain_text_splitters import CharacterTextSplitter
# import embeddings object
from langchain_openai import OpenAIEmbeddings
# import pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()
# Ingestion: take the data and load it into langchain doc object, split doc into 
# chunks with langchain splitter, embbed chunks and turn into vectors, and store vectors in pineone vector store


if __name__ == "__main__":
    print("Ingesting...")
    # print(os.environ['PINECONE_API_KEY'])
    # We are going to ingest the mediumblog1.txt file
    loader = TextLoader("/Users/ribarragi/Documents/agenticAI/UDEMY/projects/langchain-course/section_9/mediumblog1.txt")
    # loads the file into a langchain document
    document = loader.load()






