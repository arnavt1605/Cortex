import json
import ollama

class MemoryExtractor:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name= model_name

    def extract_facts(self, transcript):
        # Pass the chat log to the local model to extract JSON facts

        prompt="""
        You are an advanced AI memory extraction system. 
        Read the chat transcript between a User and an AI Assistant.
        Extract permanent, highly-relevant factual information about the user.
        
        RULES:
        - Ignore transient states (e.g., "I am tired today", "Help me debug this error").
        - Focus on skills, preferences, background, constraints, and relationships.
        - Write the facts from the perspective of an observer (e.g., "The user is...")
        
        Return the extracted facts strictly as a JSON object with a single key "facts" containing a list of strings.
        Example: {"facts": ["The user prefers coding in Java", "The user uses a Mac"]}
        If no relevant facts are found, return {"facts": []}.
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                format='json',
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"TRANSCRIPT:\n{transcript}"
                    }
                ],
                options={
                    "temperature": 0.1
                }
            )

            raw_content = response['message']['content']
            parsed_json = json.loads(raw_content)

            return parsed_json.get("facts", [])

        except json.JSONDecodeError:
            print("Failed to parse JSON")
            return []

        except Exception as e:
            print(f"Extraction failed: {e}")
            return []


