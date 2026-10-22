# Import Google Gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load variables from .env
from dotenv import load_dotenv

# Used to calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Used to read the API key
import os


# Load .env file
load_dotenv()


# Get Google Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")


# Create Gemini embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# Australian cricket documents
documents = [
    "Steve Smith is an Australian cricketer known for his exceptional batting technique and Test cricket performances.",

    "Pat Cummins is an Australian fast bowler and captain known for his pace, accuracy, leadership, and ability to take important wickets.",

    "Mitchell Starc is an Australian left-arm fast bowler famous for his high pace, swing, and dangerous yorkers.",

    "David Warner is a former Australian opening batsman known for his aggressive batting and explosive performances.",

    "Glenn Maxwell is an Australian all-rounder known for his powerful batting, innovative shots, and useful off-spin bowling.",

    "Nathan Lyon is an Australian off-spin bowler and one of Australia's most successful Test spinners.",

    "Travis Head is an Australian batsman known for his aggressive style and important performances in Test and limited-overs cricket."
]


# Ask the user for a query
query = input("Ask about Australian cricket: ")


# Convert all documents into embeddings
doc_embeddings = embedding.embed_documents(documents)


# Convert user's query into an embedding
query_embedding = embedding.embed_query(query)


# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]


# Find document with highest similarity
index, score = sorted(
    enumerate(scores),
    key=lambda x: x[1]
)[-1]


# Display result
print("\nYour Query:")
print(query)

print("\nMost Relevant Information:")
print(documents[index])

print("\nSimilarity Score:")
print(score)