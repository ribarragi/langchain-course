# This part is to load the API key
# To access the environment variable
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
# add tavily for the internet connection
from tavily import TavilyClient

# Chat model and wrapper over the OpenAI API
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate


# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
# initalize tavily, calls the API_KEY from .env
tavily = TavilyClient()


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
    # instead of just the weather in toky is sunny lets make internet search using tavily
    # our query is the input to our search function. The llm will now that the query is weather in Tokyo from the prompt
    return tavily.search(query=query)


llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello")
    result = agent.invoke(
        {"messages": HumanMessage(content="Whats the weather in Tokyo")}
    )
    print(result)
    print(result["messages"][-1].content)


# to pull the acutal content of the reponse:
# reponse[-1].content
# [-1] just means the last message in the list, which is always the LLM's final answer. .content
# gets the actual text out of it.
# {
#   "messages": [
#     HumanMessage("What's the weather in Tokyo?"),  # [0] your original question
#     AIMessage("I need to search..."),               # [1] LLM deciding to use tool
#     ToolMessage("Tokyo weather is sunny"),          # [2] tool result
#     AIMessage("The weather in Tokyo is sunny!")     # [-1] final answer ← this is what you want
#   ]
# }
if __name__ == "__main__":
    main()
