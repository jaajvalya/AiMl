import json
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from .config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME

class MongoConnector:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DATABASE_NAME]
            self.collection = self.db[COLLECTION_NAME]
            # Ping to check connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

    def get_schema_summary(self):
        """
        Retrieves a single document to help the LLM understand the schema.
        We return it as a formatted JSON string or dictionary.
        """
        try:
            sample_doc = self.collection.find_one()
            if not sample_doc:
                return "The collection is currently empty. No schema available."
            
            # Convert ObjectId to string for JSON serialization
            if '_id' in sample_doc:
                sample_doc['_id'] = str(sample_doc['_id'])
                
            return json.dumps(sample_doc, indent=2, default=str)
        except PyMongoError as e:
            return f"Error retrieving schema: {e}"

    def execute_aggregation(self, pipeline):
        """
        Executes a MongoDB aggregation pipeline and returns the results.
        """
        try:
            results = list(self.collection.aggregate(pipeline))
            
            # Convert ObjectId and other non-serializable types to strings
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return results
        except PyMongoError as e:
            return f"Database Query Error: {e}"
        except Exception as e:
            return f"Unexpected Error executing query: {e}"

# Singleton instance for easy importing
db = MongoConnector()
