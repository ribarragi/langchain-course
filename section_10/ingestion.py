# ingestion.py will hold the implementation of ingesting the LC docuentation, embedding into vectors and
# store them into Pinecone.
# Imports for the ingestion phase:

import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

# the chroma vector store in case you want to index everything locally, however he uses pinecone a cloud based vector store
from langchain_chroma import Chroma

# LC helper class to help split the documents
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# represents a text document with associated metadata
from langchain_core.documents import Document

# use open ai embeddingx
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# tavily for crwaling and data scraping
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# import from the logger.py file the logging functions to prettify them
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# configure ssl context with a valide certificate, this is to make all the API calls and dont encounter an ssl error
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# initialize openai embedding class to embed the text into a vector, chunk size to 50 to limit how many
# text objects LC documetns we will embed at every time, retry_min_secs : in case a batch fails, give it at leat 10 secs
# before re trying, this helps if the problem has to do with RATE LIMITS.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=5,
)
# This is if we were using Chroma (local store)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# created the db langchain-docs-2025 in Pinecone
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2025", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

# This function is to tak the chunks and embed them asynchronously. Takes a list of LC docs, an int of batch size, take them
# and batch index them into the vector store
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )
    
    # THIS IS A CO ROUTINE that gets a batch (list of LC docs) and the batch number to keep track in case of failing
    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


# Crawl the LC documentationusing tavily
async def main():
    """Main sync function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGETION PIPELINE")

    log_info(
        "TavilyCrawl: starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE,
    )

    # Crawl the documentation site: we are using a LC tool because its LC Tavily, we want to invoke that, we give the url, max depth,
    # advanced extraction retrieves more data
    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
            # can give instructions in natural language, for example this to restrict the scrape to ai agent content only:
            # "instructions":"content on ai gents"
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            # we are creating a LC document that has the page content (result but raw content) and in metadata a
            # dictionary with the key source and the url, we can use the metadata to know exactly where the information was retrieved from.
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )
    # So all docs has LC documents with metadata source: URL and page content: content

    # After we scrape the data, next step is to split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    # RecursiveTextSplitter: semantically chunk the documents, first chunk it by paragraphs, then by new line.
    # we limit the chunk size to 4000 chars and overlap to 200 chars.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # we have the splitter ready, then we pass the documents through
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Now we take the chiunks and index them and store them in the vector store
    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
