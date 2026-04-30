import os
from dotenv import load_dotenv
from google import genai

# Load environment variables (it will automatically find the .env file in the parent folder)
load_dotenv()

def start_chat():
    print("Welcome to Chatterbox! Type 'quit' or 'exit' to end the conversation.\n")
    
    # Initialize the client (it automatically picks up GEMINI_API_KEY from the environment)
    client = genai.Client()
    
    # Start a chat session that remembers conversation history
    # gemini-2.5-flash is the recommended default model for text tasks
    chat = client.chats.create(model="gemini-2.5-flash")
    
    while True:
        try:
            user_input = input("You: ")
            
            # Check if user wants to quit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Chatbot: Goodbye!")
                break
                
            # Skip empty inputs
            if not user_input.strip():
                continue
                
            # Send message to the LLM and get the response
            response = chat.send_message(user_input)
            print(f"Chatbot: {response.text}\n")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    start_chat()
