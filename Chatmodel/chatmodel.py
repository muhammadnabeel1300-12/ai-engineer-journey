from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

user = input("You: ")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=user
)

print(response.text)
