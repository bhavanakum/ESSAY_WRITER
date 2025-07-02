from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from typing import Annotated
from dotenv import load_dotenv
import os
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
import sqlite3
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")

class AgentState(TypedDict):
    task : str
    plan : str
    draft : str
    critique : str
    content : list[str]
    revision_number : int
    max_revisions : int
    
from langchain.chat_models import init_chat_model
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature = 0)
# result that we are getting back from llms
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: list[str]
    
# Prompt for llm which writes out the plan for essay
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

# Writes the essay given all the content that was researched
WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 3-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts.\
In the final draft at the end of the essay with a heading as reference links : include reference links from credible sources where readers can find more information about the topic.\
Utilize all the information below as needed: 
------

{content}"""

#controlling the critique of the draft of the essay
REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

# Creates the research after the planning stuff, we will create a bunch of queries and pass to tavily
RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

# after critique is generated it is generated it is sent to tavily 
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.\
In the final draft at the end of the essay with a heading as reference links : include reference links from credible sources where readers can find more information about the topic."""

from dotenv import load_dotenv
load_dotenv()

# Access the API key
api_key = os.getenv("tvly_api_key") 
from tavily import TavilyClient
tavily = TavilyClient()

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state.get('task', 'No task provided'))
    ]
    
    response = llm.invoke(messages)
    
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = llm.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state.get('content', []) 
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
        return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = llm.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content = REFLECTION_PROMPT),
        HumanMessage(content=state["draft"])
    ]
    response = llm.invoke(messages)
    return {"critique" : response.content}

def research_critique_node(state: AgentState):
    queries = llm.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state.get('critique', 'No critique provided'))
    ])
    content = state.get('content', []) 
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"