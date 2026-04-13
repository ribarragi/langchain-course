# from _typeshed import OpenBinaryMode
from dotenv import load_dotenv
import os
# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
from langsmith import traceable
import ollama

# add some imports, re to parse the raw response from the LLM, which will be text.
# regex because we wont use function calling, so we wont get JSON file, but receive the llm's
# raw response, and need to aprse form tet what function to be called.
import re
# inspect to get metadata on the functions that we will use as tools to use in llm and propagate
import inspect

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"



# --- Tools (no longer using the @tool decorator) ---
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


# small change to this function
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers are : gold, silver, broze"""
    print(f"Executing apply_discount(price='{price}, discount_tier={discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - (discount / 100)), 2)



# tools_for_llm = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_product_price",
#             "description": "Look up the price of a product in the catalog.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "product": {
#                         "type": "string",
#                         "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
#                     },
#                 },
#                 "required": ["product"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "apply_discount",
#             "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "price": {"type": "number", "description": "The original price"},
#                     "discount_tier": {
#                         "type": "string",
#                         "description": "The discount tier: 'bronze', 'silver', or 'gold'",
#                     },
#                 },
#                 "required": ["price", "discount_tier"],
#             },
#         },
#     },
# ]

# We will instead of tools_for_llm , we will create a dictionary of the tools:
tools = {
    "get_product_price":get_product_price,
    "apply_discount":apply_discount
}

# we need to inform the llm about the tools:
# receives the tool dicitonary as input, iterates over the tools and for each tool, gets its metadata, 
# its arguments, return value, docstring and format everything as a string so we can inject it into our ReAct 
# prompt and send to LLM
def get_tool_descriptions(tools_dict):
    descriptions = []
    #first element the key, second element the function
    for tool_name, tool_function in tools_dict.items():
        # for every function i want to get it metadata, 
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        # but since all funcitons are traced with langsmith, we need the original function before the
        # langsmith decorator: this basically gets us the function implementation
        # getattr(apply_discount, "__wrapped__", apply_discount) gives you: <function __main__.apply_discount(price: float, discount_tier: str) -> float>
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        # Now i need the metadata of the function: name, arguments and types, return value type
        # inspect.signature(getattr(apply_discount, "__wrapped__", apply_discount))  gives you: <Signature (price: float, discount_tier: str) -> float>
        signature = inspect.signature(original_function)
        # inspect.getdoc(apply_discount) or "" gives you the docstring (or empty string) 
        docstring = inspect.getdoc(tool_function) or ""
        # append allinto desrcriptions list
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    # join into one string to propagate into the LLM
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# Include the ReAct prompt (using first agent implementation original prompt):
# the promt is to make it function as a reasoning agent to choose the correct tool, its smart promt engineering
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# --- Helper tool for trace Ollama calls ---
# with lagchain we get out of the box tracing, here we need to manually trace llm calls for langsmith
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    # return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
    # we will use the raw intelligence of the model instead of the tools_for_llm, so it will
    # still call ollama and receive the model, messages and options: special configurations for the llm
    return ollama.chat(model=MODEL, messages=messages, options=options)
# The options are the stop arguments: its telling the LLM to, after it used the tool, dont go ahead and hallucinate
# and keep going, but rather stop, and then be fed the output of the tool, and then continue


# --- Agent loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # we have this tool dict already
    # tools_dict = {
    #     "get_product_price": get_product_price,
    #     "apply_discount": apply_discount,
    # }

    # were we inject the question to the ReAct prompt:
    # If we look at the promt, at build timethe tool_descriptions and tool_names will be fed
    # But at runtime, the question will be plugged dinamycally from the user.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    print(f"Question: {question}")
    print("=" * 60)
    # we dont need the system prompt anymore, now we need scratchpad:
    prompt = react_prompt.format(question=question)
    scratchpad = "" 
    # messages = [
    #     # SystemMessage(
    #     {
    #         "role": "system",
    #         "content": (
    #             "You are a helpful shopping assisstant."
    #             "You have access to a product catalog tool."
    #             "and a discount tool. \n\n"
    #             "Strict Rules: you must follow these exactly:\n"
    #             "1. Never guess or assume any product price."
    #             "2. You Must call get_product_price when you need to get the price \n"
    #             "3. Only call apply_discount after you have received"
    #             "a price from get_product_price. Pass that exact price"
    #             "returned by get_product_price, Do not pass a made up number"
    #             "4. Never calculate the discounts using math, but always use the apply_discount tool."
    #         ),
    #         # ),
    #         # HumanMessage(content=question),
    #     },
    #     {"role": "user", "content": question},
    # ]
    # Implement the agent loop:
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  -- iteration {iteration} --")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls


        print(tool_calls)
        if not tool_calls:
            answer = ai_message.content
            print(f"\nFinal answer: {answer}")
            return answer
        tool_call = tool_calls[0]
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
