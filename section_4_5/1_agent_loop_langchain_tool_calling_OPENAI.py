# from _typeshed import OpenBinaryMode
from dotenv import load_dotenv
import os

# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (Langchain @tool decorator) ---
@tool
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


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers are : gold, silver, broze"""
    print(f"Executing apply_discount(price='{price}, discount_tier={discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - (discount / 100)), 2)


# --- Agent loop ---
from langsmith import traceable


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"openai:gpt-5", temperature=0) # CHANGE MODEL HERE TO OPEN AI GPT-5
    llm_with_tools = llm.bind_tools(tools=tools)
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]
    # Implement the agent loop:
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  -- iteration {iteration} --")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(tool_calls)
        if not tool_calls:
            answer = ai_message.content
            print(f"\nFinal answer: {answer}")
            return answer
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: max iterarions reached with no answer")
    return None


if __name__ == "__main__":
    print("Hello")
    print("--")
    result = run_agent("whats the price of a laptop with gold discount?")
