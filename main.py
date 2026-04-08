# This part is to load the API key
# To access the environment variable
import os
# Chat model and wrapper over the OpenAI API
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# print(os.environ.get("OPENAI_API_KEY"))  # should print your key


# To bring in the API KEY from the .env file
def main():
    print("Hello from langchain-course")

    # This is just a raw string of text. Think of it as the input data you want to summarize. 
    # In a real app this might come from a database, a web scrape, a user upload, etc.
    information = """Jacques Tati (French: [tati]; born Jacques Tatischeff, pronounced [tatiʃɛf]; 9 October 1907 – 5 November 1982) 
    was a French mime, filmmaker, actor and screenwriter. In an Entertainment Weekly poll of the Greatest Movie Directors he was 
    voted 46th (a list of the top 50 was published), though he had directed only six feature-length films.
    Tati is perhaps best known for his portrayal of the character Monsieur Hulot, featured in Les Vacances de Monsieur Hulot (1953), 
    Mon Oncle (1958), Playtime (1967) and Trafic (1971). Playtime ranked 23rd in the 2022 Sight and Sound critics' poll of the greatest films ever made.
    As David Bellos puts it, "Tati, from l'École des facteurs to Playtime, is the epitome of what an auteur is (in film theory) 
    supposed to be: the controlling mind behind a vision of the world on film."""
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
# The | serves to create a runnable chain connecting  output of left component to input of right component, so we format the input using
# the prompt template, then pass the resulting prompt string into the LLM to generate a response
# we invoke the runnable object
    chain = summary_prompt_template | llm

# This is what actually runs the chain. You pass in a dictionary with the variable name 
# information mapped to your Jacques Tati biography string. LangChain slots it into the prompt template and fires off the API call.
    response = chain.invoke(input = {"information" : information})


    # print(os.environ.get("OPENAI_API_KEY"))
    print(response.content)

if __name__ == "__main__":
    main()


# Response
### Short Summary
# Jacques Tati was a renowned French mime, filmmaker, actor, and screenwriter, celebrated for his unique comedic style and innovative approach to cinema. Born on October 9, 1907, Tati is best known for his iconic character Monsieur Hulot, who appears in several of his films, including "Les Vacances de Monsieur Hulot" (1953) and "Mon Oncle" (1958). Tati's work is characterized by its visual humor and meticulous attention to detail, earning him a place among the greatest directors in film history, as evidenced by his ranking in Entertainment Weekly's poll of the top movie directors. His film "Playtime" (1967) is particularly acclaimed, having been ranked 23rd in the 2022 Sight and Sound critics' poll of the greatest films ever made.

# ### Interesting Facts
# 1. **Innovative Filmmaking Techniques**: Tati was known for his pioneering use of sound and visual gags, often employing elaborate set designs and long takes to create a unique cinematic experience. His film "Playtime" featured a meticulously constructed Parisian set that was so detailed it took over three years to build.

# 2. **Cultural Impact**: Tati's character Monsieur Hulot became a cultural icon, representing the everyman in a rapidly modernizing world. His films often critique the impact of technology and urbanization on human interactions, making them not only entertaining but also thought-provoking reflections on contemporary society.

