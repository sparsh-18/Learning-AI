from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory

load_dotenv()

llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY'),
    verbose=True
)

# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory('t-chat-messages.json'), # file to store the messages to load on each run
#     memory_key='messages',
#     return_messages=True # so that it does not only returns strings but also the messages
# )

# to summarize the conversation and avoid sending all the messages in the prompt
memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory('t-chat-messages.json'), # does not work well with summary memory, went into an infinite loop on passing empty user input
    memory_key='messages',
    return_messages=True, # so that it does not only returns strings but also the messages,
    llm=llm
)

prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        MessagesPlaceholder(variable_name='messages'), # same variable name that is used in the memory so that it can extract the messages
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input('>>')
    result = chain({'content': content})
    print(result['text'])

# output
# >>call me king
# {'content': 'call me king', 'messages': [HumanMessage(content='call me king'), AIMessage(content='Your Majesty! *bows* I shall address you as King from this moment forward. May I inquire as to the realm you rule over, or perhaps the decrees you wish to issue to your loyal subjects?')], 'text': 'Your Majesty! *bows* I shall address you as King from this moment forward. May I inquire as to the realm you rule over, or perhaps the decrees you wish to issue to your loyal subjects?'}      
# >>what did I asked you to call me?
# {'content': 'what did I asked you to call me?', 'messages': [HumanMessage(content='call me king'), AIMessage(content='Your Majesty! *bows* I shall address you as King from this moment forward. May I inquire as to the realm you rule over, or perhaps the decrees you wish to issue to your loyal subjects?'), HumanMessage(content='what did I asked you to call me?'), AIMessage(content='You asked me to call you "King".')], 'text': 'You asked me to call you "King".'}

