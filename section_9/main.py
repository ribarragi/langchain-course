import os
# import a puython utility funtion that creates a callble object to fish items from an object using indexing
# we could use lambda functions, but itemgetter is more convenient
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
# need to add this for the langchain expression implementation
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# this is a runnable that lets the inputs pass through when invoked, it wont change them,
# except that it can be configuredso that we can add additional keys to the poutput if the input
# is a dictionary
from langchain_core.runnables import RunnablePassthrough
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


# # Retrieval without langchain expression language, recevies a string and returns response of LLM
# # without langchain expression language, manual step by step: manually retrieve docs, format them and generate answer
# # ============================================================================
# # IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# # Manual implementation of RAG pipeline
# # ============================================================================
# def retrieval_chain_without_lcel(query: str):
#     """
#     Simple retrieval chain without LCEL.
#     Manually retrieves documents, formats them, and generates a response.

#     Limitations:
#     - Manual step-by-step execution
#     - No built-in streaming support
#     - No async support without additional code
#     - Harder to compose with other chains
#     - More verbose and error-prone
#     """
#     # Step 1: Retrieve relevant documents, take the user query: whats pinecone in ML and get the most relevant docs
#     # perform similarity search in the pinecone vector store. So retriever gives us the most relevant documents
#     # from our original query. It is initialized with k= 3 so it will be a list of 3 langchain docs
#     # this invoke is to get the similar docs from the pinecone vector store that we populated
#     docs = retriever.invoke(query)

#     # Step 2: Format documents into context string, the context will be a string
#     context = format_docs(docs)

#     # so far we have the question nd the context for the question, now we need to take the prompt and plug
#     # the value of the context, the value of the question and send to LLM
#     # Step 3: Format the prompt with context and question
#     messages = prompt_template.format_messages(context=context, question=query)

#     # Step 4: Invoke LLM with the formatted messages
#     response = llm.invoke(messages)

#     # Step 5: Return the content
#     return response.content


# This is the langchain expression language implementation of the RAG retrieval
# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
# This function has no arguments, it will return us a langchain chain, so the chain will be a runable so
# we can use the invoke method on what the function will return. the input to the chain will be the input
# to the invoke function
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    # this will be the rturn value for this function,
    retrieval_chain = (
        # these are piped: the prompt templated piped to the llm and the llm to the str output parser
        # the prompt template will be the prompt template of the direct prompt that we wrote with the users question
        # and the retrieve context, we pipe that into the llm, meaning we will invoke the llm with that input, and once
        # we get a reposnde from the llm, we will use the str output parser to access the .content key of the response.
        # Now how do we populate the prompt template with the context and question fields? we want to invoke the retriever,
        # then pipe the result of the retriever with the retrived docs to the format documents functionand that to pipe to the prompt
        # template, we do that inside the tunnablepassthrough function up there.
        # The reason being:
        # format_docs if a function, not a langchain runnable, doesnt have invoke method. When we use regular python functions in a langchain
        # expression language chain, LC automatically converts those reulgar python functions into runnable lambdas. So when we write
        # stuff like retrieval | format_docs | prompt_template, under the hood LC converts: retriever | RunnableLambda(format_docs) | prompt_template
        # so we can invoke format_docs.
        # Second issue: prompt template must get 2 args: question of user and context. So we wrap retriever | format_docs under a runnablepassthrough
        # and that goes into the prompt_template, use the assign method. RunnablePassthrough.assign creates a new dictionary that combines the original
        # input with the new computed field that we will explicitly mention.Here the input field is the question and what is pinecone and the
        # input_dict = {'question':'what is pinecone'} we wnat to add a new key with context and the value of the chain: itemgetter("question") | retriever | format_docs
        # this part: itemgetter("question") is equivalent to using a lmbda function like this: lambda x: x["question"], it only pulls
        # out the question string
        # in other words: the input to the RunnablePassthrough.assign(...) is input_dict = {'question':'what is pinecone'} but once it opasses though it we
        # add another key: item pair: input_dict = {'question':'what is pinecone', 'context':'doc1\ndoc2\ndoc3'}
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


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

    # # ========================================================================
    # # Option 1: Use implementation WITHOUT LCEL
    # # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    # we call the create_retrieval_chain_with_lcel, this will return us a LC chain, a runnable, so it has the invoke method
    # we can invoke it the with th einput of the dictionary with the key of question and users query
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
