# This part is to load the API key
# To access the environment variable
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Chat model and wrapper over the OpenAI API
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate


# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


# Lets start by defininf a search tool:
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
    print("Tokyo weather is sunny")  # test


llm = ChatOpenAI()
# a list of toools we are giving the agent in this case, the tool search
# that consists of printing searching for query and then tokyo weather is sunny
tools = [search]
# creates an agent by combining llm and the tools
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello")
    # human message to format a message as coming from a human
    result = agent.invoke(
        {"messages": HumanMessage(content="Whats the weather in Tokyo")}
    )
    print(result)


if __name__ == "__main__":
    main()

# This is the flow of what this is doing:
# main() starts
#   → print("Hello")                        prints: Hello
# agent.invoke() called with "What's the weather in Tokyo?"
#   ↓
# ChatOpenAI reasons: "I need to use the search tool"
#   ↓
# search("Tokyo weather") is called
#   → print(f"Searching for {query}")       prints: Searching for Tokyo weather
#   → print("Tokyo weather is sunny")       prints: Tokyo weather is sunny
#   ↓
# LLM receives tool result and formulates final answer
#   ↓
# result is returned to main()
#   → print(result)                         prints: the full agent response object

# important:
# Docstring → why to use the tool (the LLM reads this)  All the stuff in """..."""
# Type hints → how to call the tool (LangChain uses this)  (query: str) -> str:
