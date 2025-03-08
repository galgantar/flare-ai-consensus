# from google import genai

# client = genai.Client(api_key="AIzaSyALyBK7V0mY-yF50LqTV_wayIMizwyRs-8")

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="Explain how AI works",
# )

# print(response.text)

from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyALyBK7V0mY-yF50LqTV_wayIMizwyRs-8")
text = "Hello World! Tell me about AI"
result = client.models.embed_content(
    model="text-embedding-004",
    contents=text,
    config=types.EmbedContentConfig(output_dimensionality=10),
)
print(result.embeddings)