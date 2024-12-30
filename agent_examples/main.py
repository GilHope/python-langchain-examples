from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool

# Load environment variables
load_dotenv()

# Initialize Chat Model
chat = ChatOpenAI()

# Create Prompt Template
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# Define Tools
tools = [run_query_tool]

# Create Agent
agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools  
)

# Execute Agent
response = agent_executor({"input": "How many users are in the database?"})
print(response)
