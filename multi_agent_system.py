from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import datetime
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# Initialize LLM
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-8b-instruct:free",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)


# Research Tool
class ResearchTool(BaseTool):
    name: str = "research_tool"
    description: str = "Gathers detailed information about a topic using the LLM."

    def _run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. Provide detailed and factual information."),
            ("human", f"Research the following topic: {query}")
        ])
        response = llm.invoke(prompt.format())
        return response.content

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")


# Summarizer Tool
class SummarizerTool(BaseTool):
    name: str = "summarizer_tool"
    description: str = "Summarizes long text into a concise, clear summary."

    def _run(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a summarizer. Summarize the content into concise and clear form."),
            ("human", f"Summarize this: {text}")
        ])
        response = llm.invoke(prompt.format())
        return response.content

    def _arun(self, text: str):
        raise NotImplementedError("Async not supported.")


# Notifier Tool
class NotifierTool(BaseTool):
    name: str = "notifier_tool"
    description: str = "Prints and saves the final summary to a file."

    def _run(self, summary: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"summary_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n📢 Final Summary:\n{summary}\n💾 Saved to {filename}")
        return "Notification complete."

    def _arun(self, summary: str):
        raise NotImplementedError("Async not supported.")


# Setup Agent Tools
tools = [ResearchTool(), SummarizerTool(), NotifierTool()]


# Create the Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coordinator AI. You decide which tool to use to complete the user's task."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Run Multi-Agent Workflow
query = "Recent advancements in quantum computing"
result = agent_executor.invoke({"input": query})

print("\n Final Output:")
print(result["output"])
