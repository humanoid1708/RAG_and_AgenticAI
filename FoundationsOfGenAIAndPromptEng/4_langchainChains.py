"""The whole purpose of this code inside this repository is to learn about how sequential chaining works
and nowadays LCEL method is used for chaining different prompts"""

from langchain.chains import LLMChain, SequentialChain
from langchain.prompts  import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)
template = """Your job is to come up with a classic dish from the area which the user suggests.
{location}
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template = template, input_variables = ['location'])

#chain 1
location_chain = LLMChain(llm = llm, prompt = prompt_template, output_key = 'meal')

template = """Given a meal {meal}, give a short and simple recipe to make that dish at home
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template =  template, input_variables = ['meal'])

#chain 2
meal_chain = LLMChain(llm = llm, prompt = prompt_template, output_key = 'recipe')

template = """Given the recipe {recipe}, estimate how much time is required to cook it
YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template = template, input_variables = ['recipe'])

#chain 3
recipe_chain = LLMChain(llm = llm, prompt =  prompt_template, output_key = 'time')

#overall chain

overall_chain = SequentialChain(chains = [location_chain, meal_chain, recipe_chain], 
                                input_variables = ['location'], 
                                output_variables = ['meal', 'recipe', 'time'], 
                                verbose = True)

result = overall_chain.invoke({"location": "USA"})

print("=" * 50)
print("Meal:", result["meal"])
print("=" * 50)
print("Recipe:\n", result["recipe"])
print("=" * 50)
print("Estimated Time:", result["time"])