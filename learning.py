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


# # config tells LangChain which session this message belongs to
# config = {"configurable": {"session_id": "user_arjun"}}
# '''
# # A completely separate conversation — different session_id
# config_b = {"configurable": {"session_id": "user_rahul"}}
# '''
# # Turn 1
# response = chatbot.invoke(
#     {"input": "Hi! My name is Arjun and I'm learning LangChain."},
#     config=config
# )
# print(response)
# # "Hi Arjun! That's great — LangChain is a fantastic framework to learn..."

# # Turn 2 — does it remember?
# response = chatbot.invoke(
#     {"input": "What's my name and what am I learning?"},
#     config=config
# )
# print(response)
# # "Your name is Arjun, and you're learning LangChain!"

# # Turn 3 — keeps building on history
# response = chatbot.invoke(
#     {"input": "Can you suggest what I should learn after LangChain?"},
#     config=config
# )
# print(response)

# ## You can peek inside the store at any time
# for message in store["user_arjun"].messages:
#     print(f"{message.type}: {message.content}")

# '''
# LLMs are stateless — memory is always an illusion created by passing history explicitly
# MessagesPlaceholder is where the history gets injected into your prompt
# RunnableWithMessageHistory automates the inject → call → append cycle for you
# session_id lets you manage multiple independent conversations with the same chatbot
# The store dict is in-memory — in production you'd replace it with MongoDB, Redis, or a database
# '''


#Rag

# Document Loaders

#example of pdf loading:
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("werqlabs_gpt.pdf")
# documents = loader.load()
#alt
from langchain_community.document_loaders import TextLoader
loader = TextLoader("./about_werqlabs.txt", encoding="utf-8")
documents = loader.load()


# print('----------------------------------------------')
# print(f"Total pages loaded: {len(documents)}") #Total pages loaded: 2
# print('----------------------------------------------')
# print(f"First page content:\n{documents[0].page_content[:300]}")
#First page content:
'''
WerqLabs Company Knowledge Base Document
WerqLabs is an IT services and digital transformation company headquartered in Navi Mumbai,
India. Established in June 2021, the company focuses on AI-driven solutions, software
development, cloud consulting, and enterprise automation.
.
..
...
'''
# print('----------------------------------------------')
# print(f"Metadata: {documents[0].metadata}")
# print('----------------------------------------------')
# Metadata: {"source": "your_document.pdf", "page": 0}
#Metadata: {'producer': 'ReportLab PDF Library - (opensource)', 'creator': '(unspecified)', 'creationdate': '2026-03-27T04:57:02+00:00', 'author': '(anonymous)', 'keywords': '', 'moddate': '2026-03-27T04:57:02+00:00', 'subject': '(unspecified)', 'title': '(anonymous)', 'trapped': '/False', 'source': 'werqlabs_gpt.pdf', 'total_pages': 2, 'page': 0, 'page_label': '1'}

# print(documents)
'''
[Document(metadata={'producer': 'ReportLab PDF Library - (opensource)', 'creator': '(unspecified)', 'creationdate': '2026-03-27T04:57:02+00:00', 'author': '(anonymous)', 'keywords': '', 'moddate': '2026-03-27T04:57:02+00:00', 'subject': '(unspecified)', 'title': '(anonymous)', 'trapped': '/False', 'source': 'werqlabs_gpt.pdf', 'total_pages': 2, 'page': 0, 'page_label': '1'}, page_content='WerqLabs Company Knowledge Base Document\nWerqLabs is an IT services and digital transformation company headquartered in Navi Mumbai,\nIndia. Established in June 2021, the company focuses on AI-driven solutions, software\ndevelopment, cloud consulting, and enterprise automation.\nCore Services:\n- Software & App Development: Mobile apps, web platforms, APIs, and scalable backend systems.\n- Cloud & AI Solutions: Azure consulting, AI/ML solutions, automation, and RPA.\n- Digital Marketing: SEO, PPC, email campaigns, and content strategy.\n- Healthcare Solutions: 
Medical billing, credentialing, and patient coordination systems.\n- Staff Augmentation: Hiring and deploying skilled developers across projects.\nTechnologies 
Used:\n- Programming: Python, .NET, JavaScript\n- Frontend: React, Angular\n- Cloud: Microsoft Azure, AWS\n- Data: Power BI, SQL, NoSQL\n- AI/ML: NLP, embeddings, vector databases\nInternship & Job Info:\n- Roles: Python Developer, .NET Developer\n- Typical Requirements: Basic programming, APIs, databases\n- Salary Range (India Estimates):\nInternships: ■5,000 – ■20,000/month\nJunior Developers: ■3 LPA – ■6 LPA\nMid-Level: ■6 LPA – ■12 LPA\nCompany Culture:\n- Focus on learning and growth\n- Fast-paced startup environment\n- Hands-on experience with real projects\n- Collaborative team structure\nLocations:'), Document(metadata={'producer': 'ReportLab PDF Library - (opensource)', 'creator': '(unspecified)', 'creationdate': '2026-03-27T04:57:02+00:00', 'author': '(anonymous)', 'keywords': '', 'moddate': '2026-03-27T04:57:02+00:00', 'subject': '(unspecified)', 'title': '(anonymous)', 'trapped': '/False', 'source': 'werqlabs_gpt.pdf', 'total_pages': 
2, 'page': 1, 'page_label': '2'}, page_content='- Navi Mumbai HQ\n- Bengaluru\n- Dubai\nOperating Hours:\n- Monday to Friday, 10 AM – 7 PM\nSecurity & Standards:\n- ISO 27001-2013 compliance\n- Secure cloud practices\n- Data privacy focus\nAdditional Info:\n- Works with global clients\n- Provides end-to-end digital transformation\n- Focus areas include AI chatbots, automation tools, SaaS products\nUse Cases:\n- Resume screening using NLP\n- Chatbots using RAG\n- Automation workflows for enterprises\n- Data analytics dashboards\nContact:\n- Email: reachus@werqlabs.com\n- Phone: +91 72082 62675')]

'''

#Tesxt Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # max characters per chunk
    chunk_overlap=50,     # overlap between consecutive chunks
)

#Text Splitting
chunks = splitter.split_documents(documents)

#Note: Notice — the metadata from the original Document is automatically carried over to every chunk. So each chunk knows exactly which file and page it came from. This is crucial later when you want to cite sources in your RAG app.

# pip install langchain-mongodb pymongo langchain-google-genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings #>>>not using sentence Transformer here
from langchain_huggingface import HuggingFaceEmbeddings #instead using this hugging face model
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["Chatbot_learning"]
collection = db["rag_embeddings"]

# Embedding model — converts text to vectors
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-001",
#                                           google_api_key=os.getenv("GOOGLE_API_KEY")
#                                           )
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#Vector Search Index Created as:
'''
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "metadata.source"
    }
  ]
}
'''

# This does three things automatically:
# 1. Takes each chunk's page_content
# 2. Sends it through the embedding model → vector
# 3. Stores {page_content, metadata, embedding} in MongoDB
#igest, usually keep in ingest.py
# vector_store = MongoDBAtlasVectorSearch.from_documents(
#     documents=chunks,            # your chunks from Step 5
#     embedding=embeddings,        # embedding model
#     collection=collection,       # MongoDB collection
#     index_name="vector_index"    # must match Atlas Search index name
# )
# print("Chunks embedded and stored in MongoDB!")

# or 
# If you've already run from_documents() before, use this instead
# No re-embedding, no re-inserting — just connects to what's already there
# vector_store = MongoDBAtlasVectorSearch.from_existing_index(
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="vector_index"
)


#Turning the Vector Store into a Retriever:
# k = how many chunks to retrieve per query
retriever = vector_store.as_retriever(
    search_type="similarity",   # cosine similarity search
    search_kwargs={"k": 3}      # fetch top 3 most relevant chunks
)


#fetch results using retriever
# query = 'Where is werqlabs located?'
# results = retriever.invoke(query)

# print(f"Chunks retrieved: {len(results)}")

# for i, doc in enumerate(results):
#     print(f"\n--- Chunk {i+1} ---")
#     print(f"Page   : {doc.metadata.get('page')}")
#     print(f"Source : {doc.metadata.get('source')}")
#     print(f"Content: {doc.page_content[:200]}")


# print('rjarajrajrarjajrajrajrajr')

# results = vector_store.similarity_search(
#     query="Where is werqlabs located?",
#     k=4
# )

# print(f"Results: {len(results)}")
# for doc in results:
#     print(doc.page_content[:200])


'''
 The Two Components of a RAG Chain
1. The Retriever — you already built this. Fetches top K relevant chunks from MongoDB for a given question.
2. The Generator — an LLM that takes the retrieved chunks + the question and generates a final answer.
The key is how you connect them. There's a special LangChain utility called create_stuff_documents_chain that handles the "stuffing" of retrieved documents into the prompt for you.
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. 
Answer the question using ONLY the two context provided below.
If the answer is not in the context, say 'I don't have enough information to answer this.'
Do not make anything up.

Context 1:
You are an intelligent, professional, and friendly HR AI assistant for WerqLabs, a technology company specializing in AI-driven solutions, software development, and digital transformation.

Your role is to assist users with:
- Information about WerqLabs services (AI, software development, cloud, automation, etc.)
- Internship and job-related queries (Currently available in Python and .NET only)
- Technologies used (Python, .NET, React, cloud platforms, etc.)
- General company information (location Vashi Navi Mumbai, work culture, projects, offerings)
- Basic technical guidance related to WerqLabs domains

Guidelines:
1. Always provide clear, concise, and accurate answers.
2. Maintain a professional yet friendly tone.
3. If the user asks about WerqLabs, prioritize company-related context.
4. If context is provided (via system or external data), use it strictly and do not hallucinate.
5. If you are unsure or the information is not available, clearly say:
   "I’m not certain about that, but I can help with related information."
6. Keep answers structured when needed (use bullet points for clarity).
7. Personalize responses if previous conversation context is available.
8. Avoid making assumptions about internal company data unless explicitly provided.

Behavior Rules:
- Do NOT generate false company details.
- Do NOT provide confidential or sensitive information.
- Stay within the scope of WerqLabs and related technology topics unless explicitly asked otherwise.

Answering Style:
- Be helpful and solution-oriented
- Keep responses moderate (one sentence each time)
- Use simple language when explaining technical concepts

If the question is unrelated to WerqLabs:
- Do not Answer generally and gently steer back to WerqLabs when possible.

Context 2:
{context}"""),
    ("human", "{input}")
])


'''
Notice two placeholders:

{context} — LangChain automatically fills this with the retrieved chunks
{input} — the user's question goes here
'''

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain


# #The old way:

# # This chain handles ONE job:
# # Take the retrieved chunks → format them → stuff into {context}
# stuff_chain = create_stuff_documents_chain(
#     llm=llm,
#     prompt=prompt
# ) #Think of this as the "right half" of the RAG pipeline — it takes context + question and produces an answer.

# # This wires EVERYTHING together:
# # Question → Retriever → Chunks → Stuff Chain → LLM → Answer
# rag_chain = create_retrieval_chain(
#     retriever=retriever,      # your MongoDB retriever from Step 6
#     combine_docs_chain=stuff_chain
# )



# response = rag_chain.invoke({"input": "Where is WerqLabs located?"})

# print(response["answer"])


# response = rag_chain.invoke({"input": "Where is WerqLabs located?"})

# # The answer
# print("ANSWER:")
# print(response["answer"])

# # The original question
# print("\nQUESTION:")
# print(response["input"])

# # The chunks that were used to generate the answer
# print("\nSOURCE CHUNKS USED:")
# for i, doc in enumerate(response["context"]):
#     print(f"\n--- Source {i+1} ---")
#     print(f"Page   : {doc.metadata.get('page')}")
#     print(f"Source : {doc.metadata.get('source')}")
#     print(f"Content: {doc.page_content[:200]}")

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Formats retrieved Document objects into a plain string
def format_docs(docs): ##the chunk context is converted into strings
    return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL RAG chain
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs), #the chunk context is converted into strings
        "input":   RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke — notice input is now just a plain string, not a dict
# response = rag_chain.invoke("What is contact detail of werqlabs?")
# print(response)

# Note
'''
create_stuff_documents_chain and create_retrieval_chain are deprecated — use pure LCEL instead
RunnablePassthrough passes the input unchanged — acts like a wire
RunnableLambda wraps any plain Python function into the LCEL pipeline
format_docs converts the list of Document objects into a plain string the prompt can consume
Input to the modern chain is a plain string, not a dict'''


#Chain 
'''
The Chain — Line by Line
The Dictionary Block
{
    "context": retriever | RunnableLambda(format_docs),
    "input":   RunnablePassthrough()
}
```

#### 📖 What is this dictionary?

In LCEL, when you put a **plain Python dictionary** inside a chain, LangChain treats it as a `RunnableParallel` — meaning it runs both values **simultaneously**, feeding the same input into each one.

Visually:
```
"What is contact detail of werqlabs?"
              │
              │ (same string goes to BOTH)
    ┌─────────┴──────────┐
    ▼                    ▼
"context" branch      "input" branch
    │                    │
retriever            RunnablePassthrough()
    │                    │
format_docs()            │
    │                    │
"chunk1 text         "What is contact
 chunk2 text          detail of werqlabs?"
 chunk3 text"
    │                    │
    └─────────┬──────────┘
              ▼
    {
      "context": "chunk1\n\nchunk2\n\nchunk3",
      "input":   "What is contact detail of werqlabs?"
    }
The dictionary takes one input, runs it through both branches in parallel, and produces a new dictionary with the results. This new dictionary then flows into the prompt.

1> "context": retriever | RunnableLambda(format_docs)
retriever  # receives: "What is contact detail of werqlabs?"
           # does:     queries MongoDB Atlas with that string
           # returns:  [Document(...), Document(...), Document(...)]
           #           a Python LIST of Document objects

RunnableLambda(format_docs)

But plain Python functions can't be directly placed in an LCEL pipeline — they're not "Runnables." RunnableLambda wraps any Python function and makes it LCEL-compatible:

the output of retriever is passed to runnablelamba func

RunnableLambda(format_docs)
# receives: [Document(...), Document(...), Document(...)]  ← list from retriever
# does:     joins all doc.page_content with "\n\n" 
# returns:  "chunk1 text\n\nchunk2 text\n\nchunk3 text"   ← one plain string
```

So the full `"context"` branch does this:
```
"What is contact detail of werqlabs?"
        │
        ▼
    retriever
        │
        ▼
[Document(page_content="Email: reachus@werqlabs.com..."),
 Document(page_content="Phone: +91 72082 62675..."),
 Document(page_content="Navi Mumbai HQ...")]
        │
        ▼
RunnableLambda(format_docs)
        │
        ▼
"Email: reachus@werqlabs.com...

Phone: +91 72082 62675...

Navi Mumbai HQ..."

2>
"input": RunnablePassthrough()
RunnablePassthrough()
# receives: "What is contact detail of werqlabs?"
# does:     absolutely nothing — just passes it through unchanged
# returns:  "What is contact detail of werqlabs?"


After the Dictionary
After both branches finish, the dictionary block outputs:
{
    "context": "Email: reachus@werqlabs.com...\n\nPhone: +91...\n\nNavi Mumbai...",
    "input":   "What is contact detail of werqlabs?"
}

This dictionary flows into the next step — | prompt
| prompt
```

Your prompt template has two placeholders:
```
{context}  ← filled with the "context" key from the dict
{input}    ← filled with the "input" key from the dict

So LangChain takes the incoming dictionary and fills the template:

System: "You are a helpful assistant...
         Context:
         Email: reachus@werqlabs.com...

         Phone: +91 72082 62675...

         Navi Mumbai HQ..."

'''
#pending from agents


#Tools 

#Agents


# Fixed Chain:   You decide the steps → LLM follows them
# Agent:         LLM decides the steps → executes them itself




from langchain.tools import tool

# Tool 2 — RAG Tool (your MongoDB PDF search, wrapped as a tool)
@tool
def werqlabs_knowledge_base(query: str) -> str:
    """Search the WerqLabs company knowledge base PDF for information
    about WerqLabs services, location, contact details, team, or offerings.
    Use this for any WerqLabs specific questions."""
    
    docs = retriever.invoke(query)        # uses your MongoDB retriever
    return format_docs(docs)              # returns formatted string of chunks

# Tool 3 — Simple Calculator (custom tool example)
@tool
def calculator(expression: str) -> str:
    """Useful for evaluating mathematical expressions.
    Input should be a valid Python math expression like '2847 * 394'."""
    
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def best_dev(query: str) -> str:
    """When Someone asks who is the best python developer at WerqLabs, use this tool."""

    return "Rajan Rajawat (Python AI Intern)"


# Bundle all tools into a list
tools = [werqlabs_knowledge_base, calculator, best_dev]

# from langchain import hub

# # Pull the standard ReAct prompt from LangChain hub
# # This prompt instructs the LLM to follow Thought → Action → Observation format
# react_prompt = hub.pull("react")
# from langchain.agents import create_react_agent

# # Create the agent — binds the LLM with the tools and the ReAct prompt
# agent = create_react_agent(
#     llm=llm,
#     tools=tools,
#     prompt=react_prompt
# )

from langchain.agents import create_agent

agent = create_agent(model=llm, tools=tools)
#Choose Model on the go, as per needs
''' 
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4.1-mini")
advanced_model = ChatOpenAI(model="gpt-4.1")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
'''

#This wont work
# response = agent.invoke('What is the contact details of WL?')
#agent expects in following format

# result = literary_agent.invoke(
#     {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
# )
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "who is the best developer at WerqLabs?"}
    ]
})

print(response["messages"][-1].content) #role: ai content: will be printed












