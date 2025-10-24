import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser



# ----------------------------------------------------------
# 1. Load environment variables
# ----------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_KEY missing in .env")

# ----------------------------------------------------------
# 2. Initialize model
# ----------------------------------------------------------
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)

# ----------------------------------------------------------
# 3. Setup memory
# ----------------------------------------------------------
# Memory stores the running conversation history
memory = ChatMessageHistory(return_messages=True)

# ----------------------------------------------------------
# 4. Create a prompt template with memory placeholder
# ----------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that remembers past messages."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ----------------------------------------------------------
# 5. Define a function to get session-based memory
# ----------------------------------------------------------
def get_session_history(session_id: str):
    """Retrieve memory for a specific session ID."""
    return memory

# ----------------------------------------------------------
# 6. Create a Runnable with message history (tracks per user session)
# ----------------------------------------------------------
parser = StrOutputParser()
chain_with_memory = RunnableWithMessageHistory(
    prompt | llm | parser,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ----------------------------------------------------------
# 7. Chat loop
# ----------------------------------------------------------
print("=== Chat with Memory Enabled ===")
print("Type 'exit' to quit.\n")

session_id = "user-session-1"  # static session; can be dynamic per user
os.makedirs("logs",exist_ok=True)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = chain_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Assistant: {response}\n")

        # ---- JSONL logging ----
        log_entry = {
            "session_id": session_id,
            "user_input": user_input,
            "assistant_response": response,
            "timestamp": datetime.now().isoformat()
        }

        with open("logs/chat_memory_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print("Assistant: Sorry, something went wrong.\n")
