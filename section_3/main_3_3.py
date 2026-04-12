# This part is to load the API key
# To access the environment variable
import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Pydantic is a library for defining structured data schemas. Instead of the agent returning a messy
# string, you can force it to return a proper object with specific fields.
# Field lets you add a description to each field, which the LLM reads to understand what
# to put there.
from pydantic import BaseModel, Field

# add tavily for the internet connection
from tavily import TavilyClient

# Chat model and wrapper over the OpenAI API
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate


# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
from langchain_tavily import TavilySearch


# need to describe the sources that the agent uses to get the answers
# Source inherits from oydantic BaseModel
# we want our agent not to response a string but an agent response object that
# we can use downstream into an application
# Defines what a "source" looks like — just a URL. The LLM will populate
# this with the actual web pages it used to find the answer.
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


# This is the shape of the full response you want back. Instead of a raw string you get a structured object with:
# answer — the actual reply
# sources — a list of URLs the agent used
# default_factory=list just means "if no sources, default to an empty list."
class AgentResponse(BaseModel):
    """Schema for the agent response"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate answer"
    )


llm = ChatOpenAI()
tools = [TavilySearch()]
# we add to the create agent function the format
# This tells the agent: "don't give me a raw string, give me an AgentResponse object."
# The LLM will structure its reply to match that schema automatically.
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for three jobs for data analyst in the healthcare sector in companies headquartered in Pittsburgh, Pennsylvania that require use of SQL or Python or R"
            )
        }
    )
    print(result)
    print("\n----------------------------------\n")
    print(result["structured_response"].answer)  # the actual answer text
    print("\n----------------------------------\n")
    print(result["structured_response"].sources[0].url)  # first source URL
    print("\n----------------------------------\n")
    print(result["structured_response"].sources[1].url)  # second source URL
    # print("actual readable results")
    # print(result["messages"][-1].content)


# result.answer          # the actual answer text
# result.sources[0].url  # first source URL
# result.sources[1].url  # second source URL

if __name__ == "__main__":
    main()
