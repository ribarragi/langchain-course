# from _typeshed import OpenBinaryMode
import os

from dotenv import load_dotenv

# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# import ollama python sdk
import ollama
# must import this now:
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# Two options for OLLAMA (only):
# 1. based on olllama documentation, we must provide the tools explicitly here manually as a JSON
# 2. ollama can convert python funcitons into tools that can be used in ollama, but needs functions to have
# docstrings in the google docstrings format

# For Anthropic:
# Defiene tools as JSON with a bit different schema

# This means that changing models is costly


# --- Tools (no longer using the @tool decorator) ---
# want to trace them in langsmith
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up for price of a product in the catalog."""
    print(f"Executing get_product_price(product='{product}')")
    prices = {
        "telephone": 599.00,
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 59.95,
    }
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers are : gold, silver, broze"""
    print(f"Executing apply_discount(price='{price}, discount_tier={discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - (discount / 100)), 2)


# we need to manually define the JSON schema for each function (get_product_price, apply_function):
# this is what the tools decorator is doing, but fitting the requirements of each llm vendor
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# --- Helper tool for trace Ollama calls ---
# with lagchain we get out of the box tracing, here we need to manually trace llm calls for langsmith
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # tools = [get_product_price, apply_discount]
    # tools_dict = {t.name: t for t in tools}
    # we dont have the tools, we must create the dictionary by hand
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # llm = init_chat_model(f"openai:gpt-5", temperature=0) # CHANGE MODEL HERE TO OPEN AI GPT-5
    # we created the ollama_chat_traced, so we dont need this:
    # llm_with_tools = llm.bind_tools(tools=tools)
    print(f"Question: {question}")
    print("=" * 60)

    # we dont have HumanMessage, so we must format the messages differently giving it the
    # role of the user and the content of the question. In Ollama the user is called "user", in some
    # other vendors is called "human"
    messages = [
        # SystemMessage(
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assisstant."
                "You have access to a product catalog tool."
                "and a discount tool. \n\n"
                "Strict Rules: you must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "2. You Must call get_product_price when you need to get the price \n"
                "3. Only call apply_discount after you have received"
                "a price from get_product_price. Pass that exact price"
                "returned by get_product_price, Do not pass a made up number"
                "4. Never calculate the discounts using math, but always use the apply_discount tool."
            ),
            # ),
            # HumanMessage(content=question),
        },
        {"role": "user", "content": question},
    ]
    # Implement the agent loop:
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  -- iteration {iteration} --")

        # ai_message = llm_with_tools.invoke(messages)
        # tool_calls = ai_message.tool_calls

        # here we use the function we built instead:
        # response is an ollma reponse, not a langchain
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        print(tool_calls)
        if not tool_calls:
            answer = ai_message.content
            print(f"\nFinal answer: {answer}")
            return answer
        tool_call = tool_calls[0]
        # tool_name = tool_call.get("name")
        # tool_args = tool_call.get("args", {})
        # tool_call_id = tool_call.get("id")

        # Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # observation = tool_to_use.invoke(tool_args)
        observation = tool_to_use(**tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        # messages.append(
        #     ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        # )
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: max iterarions reached with no answer")
    return None


if __name__ == "__main__":
    print("Hello")
    print("--")
    result = run_agent("whats the price of a laptop with gold discount?")
