# This part is to load the API key
# To access the environment variable
import os
# Chat model and wrapper over the OpenAI API
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# add tavily for the internet connection
from tavily import TavilyClient

# initalize tavily, calls the API_KEY from .env
tavily = TavilyClient()


# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

@tool
def search(query: str) -> str:
    # Add dosctrings:
    """Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # print("Tokyo weather is sunny") #test

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model = llm, tools = tools)

def main():
    print("Hello")
    result = agent.invoke({"messages":HumanMessage(content = 'Whats the weather in Tokyo')})
    print(result)

if __name__ == "__main__":
    main()
