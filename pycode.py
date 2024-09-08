# from langchain.llms import OpenAI
# OPEN_AI_API_KEY = ''
# llm = OpenAI(
#     openai_api_key=OPEN_AI_API_KEY,
# )

# result = llm('write a poem')

import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='python')
parser.add_argument('--task', default='return the sum of two numbers')
args = parser.parse_args()

llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY')
)

# system = "You are a helpful assistant."
# human = "{text}"
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# chain = prompt | chat
# result = chain.invoke({"text": "Write a poem."})

# print(result)

code_prompt = PromptTemplate(
    template="Write a short {language} function that will {task}",
    input_variables=['language', 'task']
)

testing_prompt = PromptTemplate(
    template="Write test for {language} and code:\n{code}",
    input_variables=['language', 'code']
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code'
)

testing_chain = LLMChain(
    llm=llm,
    prompt=testing_prompt,
    output_key='test'
)

chain = SequentialChain(
    chains=[code_chain, testing_chain],
    input_variables=['language', 'task'],
    output_variables=['language', 'code', 'test']
)

result = chain({
    'language': args.language,
    'task': args.task
})

print('LANGUAGE: ', result['language'], '\n')
print('CODE: ', result['code'], '\n')
print('TEST: ', result['test'], '\n')