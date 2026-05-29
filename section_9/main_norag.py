import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("Initializing componentes...")

# initalizo openai embeddings and the llm
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

# initialize the vector store object, giving it the index name, and give the embeddings model, very similar to ingestion.py
vectorstore = PineconeVectorStore(
  index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Now we want to use the vectorstore searching capabilities
# we take the vectorstore, use its as_retriever method, this will return us an object (vector store retriever) that has 
# searching capabilities
# k=3: every time we want to search in the vector store for the relevant chunks, i want ti limit to only top 3 docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Now create the prompt template. Initialize with simple but powerful prompt:
# Answer the question based only on the following context:
# {context} is the augmentation part while {question} is the users original question
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

# Now a simple auxiliary function: format_docs will receive docs (langchain documents), and
# the funciton will take the docs and format them nivcely into a string iteratively
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# we have the query there
if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    # we invoke the llm and send the raw query with no additional context (no RAG)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)


