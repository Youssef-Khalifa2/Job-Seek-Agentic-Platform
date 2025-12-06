from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from LLM import run_rag_pipeline

load_dotenv()

# 1. Setup Models (Explicitly!)
# We must tell Ragas to use Mistral, otherwise it defaults to GPT-4
llm = ChatOpenAI(
    model="mistral-medium-latest",
    temperature=0,
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1"
)

# We must tell Ragas to use your local model, otherwise it defaults to OpenAI Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. The Test Questions
test_questions = [
    {
        "question": "Who has experience with Computer Vision?",
        "ground_truth": "Youssef Khalifa has experience with Computer Vision, specifically architecting real-time facial recognition platforms. Gabriel Marc also lists Computer Vision and Deep Learning as skills."
    },
    {
        "question": "Find me a Civil Engineer.",
        "ground_truth": "Osama Nader is a Civil Engineer."
    },
    {
        "question": "Does anyone know Snowflake?",
        "ground_truth": "Yes, Youssef Khalifa lists Snowflake as a skill."
    },
    {
        "question": "Who has experience with Flutter development?",
        "ground_truth": "Ahmed Tarek Fahim has experience with Flutter Development, Dart, and State Management (GetX, Bloc)."
    },
    {
        "question": "Find a candidate with Digital Marketing experience.",
        "ground_truth": "Ahmed Maged and Ehab Maged both have experience in Digital Marketing, SEO/SEM, and Social Media Ads."
    },
    {
        "question": "Who is a Purchasing Supervisor?",
        "ground_truth": "Ahmed Ashraf Ibrahem is a Purchasing Supervisor with experience in cost reduction and vendor management."
    },
    {
        "question": "Find someone with experience in Retail Risk Modeling.",
        "ground_truth": "Gabriel Marc is a Retail Risk Modeling Officer at EGBank."
    },
    {
        "question": "Who has skills in CRM and Statistics?",
        "ground_truth": "Jumana Khaled Ezz is a CRM Manager and Statistician with skills in R, Python, Tableau, and various CRM tools."
    },
    {
        "question": "Find a Full Stack Developer experienced with MERN stack.",
        "ground_truth": "Eslam Khaled is a Full Stack Developer skilled in MongoDB, Express (implied by Node.js), React, and Node.js."
    }
]

# 3. Run the Pipeline
results = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

print("🚀 Running pipeline...")
for item in test_questions:
    print(f"Asking: {item['question']}")
    output = run_rag_pipeline(item['question'])
    
    results["question"].append(item['question'])
    results["ground_truth"].append(item["ground_truth"])
    # Handle the dict return from your pipeline
    results["answer"].append(output["answer"])
    results["contexts"].append(output["contexts"])

dataset = Dataset.from_dict(results)

# 4. Evaluate (The Fix is Here!)
print("\n📊 Calculating Scores (Using Mistral + MiniLM)...")
scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,                # <--- Pass your LLM
    embeddings=embeddings   # <--- Pass your Embeddings
)

print("\n🏆 Final Report:")
print(scores)
scores.to_pandas().to_csv("evaluation_results.csv")