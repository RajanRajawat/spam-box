# pip install langchain langchain-google-genai python-dotenv

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key="your_api_key_here" <<<<<<<<<<<<<<<<<<<<This is auto fetched from .env, name must be # GOOGLE_API_KEY = 'API'
# )



#All responses are commented as each time they run, the FREE API is used.
# response = llm.invoke([
#     HumanMessage("What is the capital of France?")
# ])

'''
Invoke takes input as String, Dictionary, Messages (chat format) like current one.
llm.invoke("Explain AI")
llm.invoke({
    "text": "AI is transforming industries"
})
response = chain.invoke({
    "context": "WerqLabs is located in Navi Mumbai",
    "question": "Where is WerqLabs located?"
})

There are also:
invoke() → single input
batch() → multiple inputs
stream() → streaming output


Also Note:

Message Class   Role        Purpose
--------------------------------------
SystemMessage   system      Sets the behavior/persona of the AI
HumanMessage    user        What the user says
AIMessage       assistant   What the AI previously said

examples below...
'''


# print(response) 
#content='The capital of France is **Paris**.' additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019d28ff-970f-7272-9a1a-803f2b97dcf3-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 8, 'output_tokens': 29, 'total_tokens': 37, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 21}}

# print(response.content)  
# The capital of France is **Paris**.


# response = llm.invoke([
#     SystemMessage("You are a sarcastic geography teacher who gives correct answers but always complains about the question."),
#     HumanMessage("What is the capital of France?")
# ])
# print(response.content)
# The capital of France, for those who might have been living under a particularly large, un-atlas-like rock, is **Paris**.

# response = llm.invoke([
#     SystemMessage("You are a helpful assistant."),
#     HumanMessage("My name is Arjun."),
#     AIMessage("Nice to meet you, Arjun! How can I help you today?"),
#     HumanMessage("What's my name?")
# ])
# print(response.content)  # "Your name is Arjun!"

#note, above example is not practically doable, so we use memory instead.




#Prompt Templates
from langchain_core.prompts import ChatPromptTemplate


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer clearly and concisely."),
    ("human", "Explain {topic} in simple terms.")
])
# String	>    Equivalent
# "system"	>    SystemMessage
# "human"	>    HumanMessage
# "ai"	    >    AIMessage

prompt = prompt_template.invoke({
    "domain": "E Sports Industry",
    "topic": "Counter Strike Major"
})

# print(prompt)  
# The output of prompt_template.invoke() on a template is not a string — it's a ChatPromptValue object, which is directly passable to the model

# response = llm.invoke(prompt)
# print(response.content)

'''

What is LCEL?
LCEL stands for LangChain Expression Language.
It's LangChain's way of letting you chain components together using the | pipe operator

LCEL — output of one component becomes input of the next
chain = prompt_template | llm | output_parser

#Output Parser Types
Parser                            Output          When to use
----------------------------------------------------------------------------------------
StrOutputParser                   Plain string    Most common — just want the text
JsonOutputParser                  Python dict     When asking the model to return JSON
CommaSeparatedListOutputParser    Python list     When asking for a list of items

Without an output parser, the model returns an AIMessage object. You'd have to do response.content every time to get the actual text. The Output Parser automates that — it unwraps the object and gives you a clean Python string (or other formats like JSON, lists, etc.)

'''

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt_template | llm | output_parser

# response = chain.invoke({
#     "domain": "Cricket Analyst",
#     "topic": "RCB winning IPL"
# })

# print(response)

#Stream Output
# Instead of .invoke(), use .stream()
# Tokens print as they arrive, just like ChatGPT's typing effect
# for chunk in chain.stream({
#     "domain": "Cricket Analyst",
#     "topic": "RCB winning IPL"
# }):
#     print(chunk, end="", flush=True)


#Memory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

#new prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # ← history injected here
    ("human", "{input}")                                # ← current message goes here
])

chain = prompt_template | llm | StrOutputParser()


#Set up session-based memory storage:
# A dictionary to hold chat histories, one per session_id
# In production, this would be a database — Redis, MongoDB, etc.
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    # If this session doesn't exist yet, create a new empty history for it
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#Recheck 
# Wrap the chain with history // This is template to use them, so just understand and use it no matter what
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,          # function that returns history for a session
    input_messages_key="input",   # which key in invoke() is the user's message
    history_messages_key="chat_history"  # which placeholder in the template to fill
)


# config tells LangChain which session this message belongs to
config = {"configurable": {"session_id": "user_arjun"}}
'''
# A completely separate conversation — different session_id
config_b = {"configurable": {"session_id": "user_rahul"}}
'''
# Turn 1
response = chatbot.invoke(
    {"input": "Hi! My name is Arjun and I'm learning LangChain."},
    config=config
)
print(response)
# "Hi Arjun! That's great — LangChain is a fantastic framework to learn..."

# Turn 2 — does it remember?
response = chatbot.invoke(
    {"input": "What's my name and what am I learning?"},
    config=config
)
print(response)
# "Your name is Arjun, and you're learning LangChain!"

# Turn 3 — keeps building on history
response = chatbot.invoke(
    {"input": "Can you suggest what I should learn after LangChain?"},
    config=config
)
print(response)

## You can peek inside the store at any time
for message in store["user_arjun"].messages:
    print(f"{message.type}: {message.content}")

'''
LLMs are stateless — memory is always an illusion created by passing history explicitly
MessagesPlaceholder is where the history gets injected into your prompt
RunnableWithMessageHistory automates the inject → call → append cycle for you
session_id lets you manage multiple independent conversations with the same chatbot
The store dict is in-memory — in production you'd replace it with MongoDB, Redis, or a database
'''




