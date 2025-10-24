import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------
# 1. Load environment variables
# ----------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# ----------------------------------------------------------
# 2. Initialize model
# ----------------------------------------------------------
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    max_tokens=300,
    api_key=api_key,
    base_url=base_url,
)

parser = StrOutputParser()

# ----------------------------------------------------------
# 3. Define prompt templates
# ----------------------------------------------------------
summary_prompt = ChatPromptTemplate.from_template(
    "<s>[INST] Summarize the topic '{topic}' in simple terms for a beginner. Keep it concise and clear. [/INST]"
)

quiz_prompt = ChatPromptTemplate.from_template(
    "<s>[INST] Based on the topic '{topic}', generate 3 short quiz questions that test understanding of the summary. [/INST]"
)

# ----------------------------------------------------------
# 4. Define chain functions
# ----------------------------------------------------------
def generate_summary(topic):
    chain = summary_prompt | llm | parser
    return chain.invoke({"topic": topic}).strip()

def generate_quiz(topic):
    chain = quiz_prompt | llm | parser
    return chain.invoke({"topic": topic}).strip()

# ----------------------------------------------------------
# 5. Run dynamically
# ----------------------------------------------------------
user_topic = input("Enter a topic to summarize and generate quiz: ").strip()

summary = generate_summary(user_topic)
quiz = generate_quiz(user_topic)

print(f"\n--- SUMMARY ---\n{summary}\n")
print(f"--- QUIZ QUESTIONS ---\n{quiz}\n")

os.makedirs("logs", exist_ok=True)
log_entry = {
    "topic": user_topic,
    "summary": summary,
    "quiz": quiz,
    "timestamp": datetime.now().isoformat(),
}

with open("logs/quiz_chain_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")

print("Results logged to logs/quiz_chain_log.jsonl")