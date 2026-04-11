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
# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))



# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:
#     """Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query = query)

# tavily has a builtin search tool
from langchain_tavily import TavilySearch




llm = ChatOpenAI()
# its already a tool tavilysearch, so we initialize it here
# if we go to smith.langchain we can see the way more ellaborate tavily search tool
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools)

def main():
    print("Hello")
    result = agent.invoke({"messages":HumanMessage(content = 'Whats the weather in Tokyo')})
    print(result)
    print("actual readable results")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
