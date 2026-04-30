import json
from google import genai
from google.genai import types
from .config import COLLECTION_NAME, GEMINI_API_KEY
from .database import db

# Initialize the Gemini client
client = genai.Client()

def get_mongodb_query(user_question, schema):
    """
    Step 1: Ask Gemini to generate a MongoDB aggregation pipeline based on the user's question.
    We force the model to output JSON using response_mime_type.
    """
    system_instruction = f"""
You are an expert MongoDB developer. Your task is to translate natural language questions into MongoDB aggregation pipelines.
The target collection is '{COLLECTION_NAME}'.
Here is a sample document representing the schema of the collection:
{schema}

Instructions:
1. Output ONLY a valid JSON array representing the MongoDB aggregation pipeline.
2. Do not include any explanations, markdown blocks, or comments in the output.
3. Make sure the query effectively answers the user's question based on the provided schema.
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_question,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        # Parse the JSON response
        pipeline = json.loads(response.text)
        return pipeline
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output into JSON: {response.text}")
        return None
    except Exception as e:
        print(f"Error generating query: {e}")
        return None

def generate_final_answer(user_question, query_results):
    """
    Step 2: Ask Gemini to generate a natural language response based on the query results.
    """
    system_instruction = """
You are a helpful data assistant. You have queried a database to answer a user's question.
Given the user's question and the raw JSON results from the database query, provide a clear, concise, and natural language answer.
Do not mention the raw JSON or the database query directly, just provide the final answer to the user.
"""
    prompt = f"User Question: {user_question}\n\nDatabase Results: {json.dumps(query_results, default=str)}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        return response.text
    except Exception as e:
        return f"Error generating final response: {e}"

def process_user_query(user_question):
    """
    Main agent pipeline:
    1. Get Schema
    2. Generate Query
    3. Execute Query
    4. Generate Answer
    """
    # 1. Get the schema
    schema = db.get_schema_summary()
    
    # 2. Generate the MongoDB query
    pipeline = get_mongodb_query(user_question, schema)
    if not pipeline:
         return "I'm sorry, I couldn't formulate a valid database query for your question."
         
    # print(f"\n[Debug] Generated Pipeline: {json.dumps(pipeline, indent=2)}")
         
    # 3. Execute the query
    results = db.execute_aggregation(pipeline)
    if isinstance(results, str) and results.startswith(("Database Query Error", "Unexpected Error")):
        # It's an error message
        return f"I encountered an error querying the database: {results}"
        
    # print(f"\n[Debug] Query Results: {json.dumps(results, indent=2, default=str)}")
        
    # 4. Generate the final answer
    final_answer = generate_final_answer(user_question, results)
    
    return final_answer
