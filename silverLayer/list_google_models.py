import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Available Google Generative AI Models:")
print("=" * 80)

for model in genai.list_models():
    print(f"\nModel Name: {model.name}")
    print(f"Display Name: {model.display_name}")
    print(f"Supported Methods: {model.supported_generation_methods}")
    
print("\n" + "=" * 80)
print("\nModels that support generateContent:")
print("=" * 80)

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"✓ {model.name}")
