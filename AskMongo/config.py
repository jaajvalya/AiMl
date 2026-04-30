import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the parent directory
load_dotenv()

# Centralized configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Default database and collection from the user's requirements
DATABASE_NAME = "chatterbot"
COLLECTION_NAME = "palbooks"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")
