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
    prices = {
        "telephone": 599.00,
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 59.95,
    }
    # access the dictionary called prices and fetch the price of the product, if not listed, give me 0
    return prices.get(product, 0)


# Another tool to apply discount:
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers are : gold, silver, broze"""
    print(f"Executing apply_discount(price='{price}, discount_tier={discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - (discount / 100)), 2)


# --- Agent loop ---
# Now define the agents loop raw without langchain, so we need langsmith traces and implementation:
from langsmith import traceable


# We use the traceable function as a decorator, will help trace all inside the scope of this
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # this is a list of tools:
    tools = [get_product_price, apply_discount]
    # Now a dicitonary linking the tool anme and the tool itself.
    # each tool has .name, also has.description, args.schema and func
    # this is a dictionary of the tool anme: tool itself
    tools_dict = {t.name: t for t in tools}
    # initialize model, tchis is how you initialize any model flexibly.
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)
    # we now take the model and let it know what tools we have:
    llm_with_tools = llm.bind_tools(tools=tools)
    print(f"Question: {question}")
    print("=" * 60)

    # Now implement the brain of the agent: a bunch of prompts to send to llm, which will
    # decide whats the answer, twhat to execute:
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assisstant."
                "You have access to a product catalog tool."
                "and a discount tool. \n\n"
                # Some defensive prompting. It is not as necessary when using other first tier llms
                "Strict Rules: you must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "2. You Must call get_product_price when you need to get the price \n"
                "3. Only call apply_discount after you have received"
                "a price from get_product_price. Pass that exact price"
                "returned by get_product_price, Do not pass a made up number"
                "4. Never calculate the discounts using math, but always use the apply_discount tool."
            )
        ),
        HumanMessage(content=question),
    ]
    # Implement the agent loop:
    # This is the heart of the agent. It loops up to 10 times (MAX_ITERATIONS)
    for iteration in range(1, MAX_ITERATIONS + 1):
        # in every iteration we want to send this message to the LLM
        print(f"\n  -- iteration {iteration} --")
        # the aimessage will contain either a tool call decision from the LLM or content in case it has the answer:
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(tool_calls)
        # If there are no tools, then it means the llm has decided to provide output, so there must the content
        if not tool_calls:
            answer = ai_message.content
            print(f"\nFinal answer: {answer}")
            return answer
        # access the first element of the tool call list and extract the name, the argumnents and the id
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        # here we have like a search into the tolls we have to look for the tool caled.
        # Uses the tools_dict lookup table you built earlier to find the actual function.
        # So if tool_name = "get_product_price", this gives you back the actual get_product_price function.
        # we need to go and fetch the actual function to run it because the LLM only decided that it wants to run
        # it but it hasnt run it!
        tool_to_use = tools_dict.get(tool_name)

        # Just in case the llm hallucinates and gives you a tool non existent
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # This part runs the tool
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        # now we need to append the aimessage which will contain the tool call (the decision of the llm) and
        # append the tool result (Observation) and the tool call id for tracing, and we feed back this to the llm
        # So now the agent when it processes input it will see what it did in the past
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

        # we expect at some point to finish and have a final answer if not, print an error
    print("ERROR: max iterarions reached with no answer")
    return None


if __name__ == "__main__":
    print("Hello")
    print("--")
    result = run_agent("whats the price of a laptop with gold discount?")
