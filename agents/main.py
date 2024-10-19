from typing import Literal
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
# from sql_tool import run_query_tool, list_tables, describe_tables_tool, list_tables_tool
from sql_tool import list_tables, run_sqlite_query, describe_tables, write_html_report
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

chat_llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY'),
    verbose=True
)

# tables = list_tables()

prompt_template = ChatPromptTemplate(
    messages=[
        # SystemMessage(content=f"You are an AI having access to a sqlite database. Tables are: {tables}."), - This is not working so given a tool to list tables
        # SystemMessage(content=f"You are an AI having access to a sqlite database."), - This was causing the model to assume the tables
        SystemMessage(content=f"You are an AI having access to a sqlite database."),
        HumanMessagePromptTemplate.from_template("Do not assume any tables or their structure. Use provided tools."),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

tools = [list_tables, run_sqlite_query, describe_tables, write_html_report]
tool_node = ToolNode(tools)

c = chat_llm.bind_tools(tools)
# print('==========================================================================')
# pr = prompt_template.format_prompt(input="Give me a list of tables in my database?").to_messages()
# print(pr, '\n======================================\n')
# res = c.invoke(pr)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    # print("\n=============================================================\n")
    # print("call_model function arguments: ", state)
    messages = state['messages']
    response = c.invoke(messages)
    # We return a list, because this will get added to the existing list
    # print("call_model function response: ", response)
    # print("\n=============================================================\n")
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
def getPrompt(input):
    return prompt_template.format_prompt(input=input).to_messages()

# inputPrompt = "What is the name and address of the user having highest orders?"
# ans - Miss Karen Knight has the highest orders. Her addresses are 343 Thomas Mills, Port Steven, IN 85636, 38540 Anthony Valley, West Paul, 
# MI 42440, and 92961 Warren Plaza Apt. 211, East Colinchester, LA 90837.

# inputPrompt = "Which is the most trendy product and give me a list of users who have bought it?"
# ans - The most trendy product is Soap.

# inputPrompt = "Which is the most trendy product and provide a list of name of users who have ordered it?"
# It used the order_products table to find the most trendy product
# ans - The list of users who have ordered the most trendy product "Soap" is:
# 1. Lauren Perry
# 2. Ronald Lee
# 3. Joshua Hines
# 4. Gregory Lambert
# 5. Nicholas Dunn
# 6. Leslie Lewis
# 7. Robert Rivas
# 8. Jonathan Wilson
# 9. Andrew Stewart
# 10. Samantha Gomez
# 11. Amy Goodwin
# 12. William Little
# 13. Jennifer Palmer
# 14. Jessica Martinez
# 15. Nicole Mora
# 16. Paul Jackson
# 17. Paul Stephenson
# 18. Jesse Wells Jr.
# 19. Deborah Reynolds
# 20. Jeremy Hanson
# 21. Timothy Lucas Jr.
# 22. Margaret Powell
# 23. Amanda Spencer
# 24. William Mccoy
# 25. Leslie Tapia
# 26. Andrea Garcia
# 27. Mary Goodman
# 28. Brian Cooley
# 29. Amy Johnson
# 30. Dwayne Vega
# 31. Jeffrey Hess
# 32. Katherine Fitzgerald
# 33. Jonathan Lopez
# 34. Terry Edwards
# 35. Kyle Reynolds
# 36. James Frye
# 37. Kiara Weaver
# 38. Crystal Smith
# 39. Stacey Lewis
# 40. Jennifer Morton
# 41. Bethany Watson
# 42. James Owens
# 43. Christina Jennings
# 44. Fred Howe
# 45. Cathy Jones
# 46. Elizabeth Simpson
# 47. Thomas Jackson
# 48. Edward Jones
# 49. Cassandra Smith
# 50. Rachel Wallace
# 51. Sandra Mckinney
# 52. Peter Martin
# 53. Charles Guzman
# 54. James Olson
# 55. Tyrone Long
# 56. Deborah Rivera
# 57. Tracy Allison
# 58. Erin Hall
# 59. Jamie Morgan DVM
# 60. Cassandra Scott
# 61. Anthony Colon
# 62. Catherine Massey
# 63. Robert Henderson
# 64. Stephanie Garcia
# 65. Gregory Collins
# 66. Michelle Smith
# 67. Jonathan Cannon
# 68. Julian Goodman
# 69. Rebecca Jones
# 70. Christopher Rogers
# 71. Francis Lopez
# 72. Christopher Reed
# 73. Samantha Fritz
# 74. Brooke Lee PhD
# 75. Robin Boyd
# 76. Erika Mcbride
# 77. Michael Nicholson
# 78. Douglas Miranda
# 79. Alexandra Washington
# 80. Mark Weiss
# 81. Taylor Williams
# 82. Chris Lam
# 83. Amber Howard
# 84. Gary Silva
# 85. Amanda Rivas
# 86. Michael Harrington
# 87. Timothy Bauer
# 88. Robert Gomez
# 89. Jay Mercer Jr.
# 90. Whitney White
# 91. Christopher Torres
# 92. Joseph Reeves
# 93. Scott Jackson
# 94. Barbara Garcia
# 95. Mindy Brooks
# 96. Jeff Mitchell
# 97. Sara Tanner
# 98. Alexander Smith
# 99. Barbara King
# 100. David Cox
# 101. Jennifer Morse
# 102. Jessica Hamilton
# 103. Kayla Leach
# 104. James Flores
# 105. Timothy Giles

# 2nd run it used cart table to find the most trendy product
# ans - The list of user IDs who have ordered the most trendy product, "Pants", is: 53, 886, 217, 1203, 681, 1040, 369, 1022, 323, 146, 1175, 
# 240, 252, 480, 808, 970, 861, 639, 490, 1266, 91, 324, 724, 661, 1012, 1466, 1240, 1370, 1174, 422, 11, 302, 1027, 756, 410, 444, 1079, 134, 479, 1075, 1019, 1450, 406, 841, 575, 1173.

# 3rd run it used order_products table again to find the most trendy product


# inputPrompt = "Write an HTML report on the most ordered product." # report.html
inputPrompt = "Write an HTML guide describing the db also helping user write queries." # db_guide.html

final_state = app.invoke(
    input={'messages': getPrompt(inputPrompt)},
    config={"configurable": {"thread_id": 42}}
)

messages = final_state["messages"]

res = messages[-1].content
print(res)

tool_calls = [
     call
    for message in messages
    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs
    for call in message.additional_kwargs['tool_calls']
]

for call in tool_calls:
    print('\n',call)

print('\n*************execution completed*************\n')