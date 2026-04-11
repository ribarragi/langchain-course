# from _typeshed import OpenBinaryMode
from dotenv import load_dotenv
import os
# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


# from langchain.chat_models import init_chat_model. This is a more flexible way to initialize any 
# chat model — OpenAI, Ollama, Anthropic, etc. — without importing 
# each one separately. You just pass the model name as a string and LangChain figures out the rest. 
# That's why MODEL = "qwen3:1.7b" works here instead of needing ChatOpenAI() specifically.
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ToolNessage: a message containing the tool result
# SystemMessage: a wrapepr to implicate that we are using a systemmessage to the LLM
# HumanMessage: when we use user input

# SystemMessage — sets the behavior and personality of the LLM before the conversation starts. Things like "you are a helpful shopping assistant"
# ToolMessage — wraps the result of a tool call so the LLM can receive it as part of the conversation history

# To limit agent execution runs
MAX_ITERATIONS = 10 
MODEL = "qwen3:1.7b"

# --- Tools (Langchain @tool decorator) ---
@tool
def get_product_price(product: str) -> float:
  """Look up for price of a product in the catalog."""
  print(f"Executing get_product_price(product='{product}')")
  # lets define a dictionary of prices for this example:
  prices = {'telephone': 599.00, "laptop": 1299.99, "headphones": 149.95, "keyboard": 59.95}
  # access the dictionary called prices and fetch the price of the product, if not listed, give me 0
  return prices.get(product, 0)

# Another tool to apply discount:
@tool
def apply_discount(price:float, discount_tier:str) -> float:
  """Apply a discount tier to a price and return the final price.
  Available tiers are : gold, silver, broze"""
  print(f"Executing apply_discount(price='{price}, discount_tier={discount_tier}')")
  discount_percentages = {'bronze': 5, 'silver':10, 'gold': 15}
  discount = discount_percentages.get(discount_tier, 0)
  return round(price * (1 - (discount/100)), 2)

# --- Agent loop ---
# Now define the agents loop raw without langchain, so we need langsmith traces and implementation:
from langsmith import traceable

# We use the traceable function as a decorator, will help trace all inside the scope of this
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
  pass

if __name__ == "__main__":
  print("Hello")
  print('--')
  result = run_agent("whats the price of a laptop with gold discount?")





 