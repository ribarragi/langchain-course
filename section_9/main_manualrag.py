import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
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


# Retrieval without langchain expression language, recevies a string and returns response of LLM
# without langchain expression language, manual step by step: manually retrieve docs, format them and generate answer
# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# Manual implementation of RAG pipeline
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents, take the user query: whats pinecone in ML and get the most relevant docs
    # perform similarity search in the pinecone vector store. So retriever gives us the most relevant documents
    # from our original query. It is initialized with k= 3 so it will be a list of 3 langchain docs
    # this invoke is to get the similar docs from the pinecone vector store that we populated
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string, the context will be a string
    context = format_docs(docs)

    # so far we have the question nd the context for the question, now we need to take the prompt and plug
    # the value of the context, the value of the question and send to LLM
    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# we have the query there
if __name__ == "__main__":
    # print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # # ========================================================================
    # # Option 0: Raw invocation without RAG
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # # we invoke the llm and send the raw query with no additional context (no RAG)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)
