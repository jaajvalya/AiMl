import os
import sys

# Ensure the parent directory is in the path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AskMongo.agent import process_user_query

def start_chat():
    print("\n========================================================")
    print("Welcome to AskMongo!")
    print("I can translate your questions into MongoDB queries and")
    print("fetch the answers directly from your database.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("========================================================\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nAskMongo: Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("AskMongo: Let me check the database...")
            response = process_user_query(user_input)
            
            print(f"\nAskMongo: {response}\n")
            
        except KeyboardInterrupt:
            print("\nAskMongo: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    start_chat()
