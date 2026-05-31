import os

from dotenv import load_dotenv
# import txt loader
from langchain_community.document_loaders import TextLoader
# import embeddings object
from langchain_openai import OpenAIEmbeddings
# import pinecone
from langchain_pinecone import PineconeVectorStore
# import text splitter
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
# Ingestion: take the data and load it into langchain doc object, split doc into
# chunks with langchain splitter, embbed chunks and turn into vectors, and store vectors in pineone vector store


if __name__ == "__main__":
    print("Ingesting...")
    # print(os.environ['PINECONE_API_KEY'])
    # We are going to ingest the mediumblog1.txt file
    loader = TextLoader(
        "/Users/ribarragi/Documents/agenticAI/UDEMY/projects/langchain-course/section_9/mediumblog1.txt"
    )
    # loads the file into a langchain document
    document = loader.load()

    # document will be a list of langchain documents. Some key attributes: page_content (content), metadata (source),...

    # Next we go and split text
    print("splitting...")
    # character texr splitter object. It has a lot of customizations, but here very basic, just
    # the chunk size (1000 characters, rule of thumb is to keep it small enough to fir in the
    # context window and it is readable, so 1000 chars sounds good) and the overlap.
    # We want overlaping data when we dont want to add context bewtween chuns.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # we invoke the function, the method split_documents recevies a list of langchain documents
    texts = text_splitter.split_documents(document)
    print(f"created: {len(texts)} chunks")

    # Now its time to ingest everything: initialize an open ai embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # which embeddings will we be using? In this case default.
    print((f"ingesting ..."))
    # The langchain vector store, in this case Pinecone, has the from_documents, it will receive 'texts', which
    # is a list of documents, also the embeddings object which has the information of the embeddings model to use,
    # and the index name that we have available in our env var.

    # Langchain will iteratate over all the documents, all the chunks, it will embedd each of them and
    # store them in the vector store.
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")

    # After this code runs, we have sent the chunks embedded into pinecone
