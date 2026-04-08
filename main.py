# This part is to load the API key
# To access the environment variable
import os
# Chat model and wrapper over the OpenAI API
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# To bring in the API KEY from the .env file
def main():
    print("Hello from langchain-course")

    # This is just a raw string of text. Think of it as the input data you want to summarize. 
    # In a real app this might come from a database, a web scrape, a user upload, etc.
    information = """Jacques Tati (French: [tati]; born Jacques Tatischeff, pronounced [tatiʃɛf]; 9 October 1907 – 5 November 1982) was a French mime, filmmaker, actor and screenwriter. In an Entertainment Weekly poll of the Greatest Movie Directors he was voted 46th (a list of the top 50 was published), though he had directed only six feature-length films.
Tati is perhaps best known for his portrayal of the character Monsieur Hulot, featured in Les Vacances de Monsieur Hulot (1953), Mon Oncle (1958), Playtime (1967) and Trafic (1971). Playtime ranked 23rd in the 2022 Sight and Sound critics' poll of the greatest films ever made.
As David Bellos puts it, "Tati, from l'École des facteurs to Playtime, is the epitome of what an auteur is (in film theory) supposed to be: the controlling mind behind a vision of the world on film."""
    # This is your prompt, but with a placeholder {information} in curly braces. Instead of hardcoding the biography into the prompt, 
    # you leave a blank that gets filled in later. This makes the prompt reusable for any person's bio.
    summary_template = """Given the information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them"""

# we want reusable dynamic prompts, we use PromptTemplate
# This takes your template string and formally tells LangChain: "this template has one variable called information." 
# LangChain will later substitute {information} with the actual biography text when the chain runs.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template = summary_template
    )

# temperature control how random or creative / deterministic the model will be: deterministic 0-0.3 good for summarisation, test
# 0.8-1.0 very creative results, fiction, out of box ideas.
                    #  model="gpt-5"
    llm = ChatOpenAI(temperature=0, model = "gpt-4o-mini")
# we now create out first chain:
# The | pipe operator is LangChain's way of connecting steps together. This is the core idea of LangChain — you build pipelines.
# What this chain does in order:
# Takes your summary_prompt_template
# Fills in the {information} placeholder with real text
# Sends the completed prompt to llm (OpenAI)
# Returns the response
    chain = summary_prompt_template | llm

# This is what actually runs the chain. You pass in a dictionary with the variable name 
# information mapped to your Jacques Tati biography string. LangChain slots it into the prompt template and fires off the API call.
    response = chain.invoke(input = {"information" : information})


    # print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
