import sys
import argparse
import ollama
from cortex.storage import SecureMemoryDB
from cortex.extractor import MemoryExtractor
from pyfiglet import  figlet_format

class MemoryAgent:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name= model_name
        self.db= SecureMemoryDB()
        self.extractor = MemoryExtractor(model_name= model_name)

        self.transcript= []


    def build_system_prompt(self, user_input):
        # Searches the memory related to the user prompt and build a hidden system prompt
    
        relevant_memories= self.db.search_memories(user_input, top_k=3)

        system_context = "You are a helpful AI assistant."

        if relevant_memories:
            memory_points = "\n".join([f"- {mem[0]}" for mem in relevant_memories])
            system_context += f"Here are some things you know about the user:\n{memory_points}"
            print(f"[System: Silently injected {len(relevant_memories)} memories into context]")

        return system_context
    
    def get_recent_memory(self):
        # Gathering the context of the last 5 chats
        messages= []
        for line in self.transcript[-5:]:
            role= "user" if line.startsWith("User:") else "assistant"

            if ": " in line:
                content = line.split(": ", 1)[1]  #string.split(separator, maxsplit)
            else:
                content = line

            messages.append({"role": role, "content": content})
        return messages
    
    def stream_response(self, messages):
        #Sends the prompt to Ollama and streams the response to the terminal
        print("AI: ", end="", flush=True)
        response_text = ""

        try:
            stream = ollama.chat(model=self.model_name, messages=messages, stream=True)

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end="", flush=True)
                response_text += content
            
            print() # Print a final newline when done
            return response_text
        except Exception as e:
            print(f"\nError connecting to Ollama: {e}")
            return ""


    def chat_loop(self):
        # Main interactive loop
        print(figlet_format("cortexDB", font="slant"))

        print("Local AI with True Memory")
        print("Type 'exit' or 'quit' to end the session.")
    

        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                self.end_session()
                break


            elif user_input.lower() == 'show memories':
                memories= self.db.get_all_memories()
                print("\n--- Your Stored Memories ---")
                if not memories:
                    print("Your memory is currently empty")
                else:
                    for idx, mem in enumerate(memories, 1):
                        print(f"{idx}. {mem}")
                print("-------------------------------\n")
                continue

            elif user_input.lower() == 'clear memories':
                confirm = input("Are you sure you want to permanently delete all memories? (y/n): ")
                if confirm.lower() == 'y':
                    self.db.clear_all_memories()
                    print("Memories cleared successfully.")
                else:
                    print("Action cancelled.")
                continue 




            self.transcript.append(f"User: {user_input}")

            system_prompt = self.build_system_prompt(user_input)
            chat_history = self.get_recent_memory()
            
            # Combine the system prompt and chat history
            messages = [{"role": "system", "content": system_prompt}] + chat_history

            ai_response = self.stream_response(messages)
            
            if ai_response:
                self.transcript.append(f"AI: {ai_response}")
    

    def end_session(self):
        #Runs when the user types exit or quit and triggers the Extractor 
        print("\n\nClosing session. Analyzing chat for permanent facts...")
        if not self.transcript:
            return
            
        full_transcript = "\n".join(self.transcript)
        
        new_facts = self.extractor.extract_facts(full_transcript)
        
        # Storing the facts
        if new_facts:
            added_count = 0
            for fact in new_facts:
                if not self.db.is_duplicate(fact):
                    self.db.add_memory(fact)
                    added_count += 1
                else:
                    print(f"Skipping duplicate fact: '{fact}'")
            
            if added_count > 0:
                print(f"Learned {added_count} new things about you!")
            else:
                print("No *new* permanent facts found this session (duplicates ignored).")
        else:
            print("No new permanent facts found this session.")


def main():
    parser = argparse.ArgumentParser(description="CortexDB: Local AI with True Memory")

    parser.add_argument(
        "-m", "--model", 
        type=str, 
        default="llama3.1:8b", # default fallback model
        help="The name of the local Ollama model you want to use."
    )
    
    args = parser.parse_args()

    try:
        available_models = [m['model'] for m in ollama.list()['models']]
        
        # If the model isnt installed
        if args.model not in available_models and f"{args.model}:latest" not in available_models:
            print(f" Warning: Model '{args.model}' was not found in your local Ollama.")
            print(f"Installed models found: {', '.join(available_models)}")
            print("Please pull the model first using: 'ollama pull <model_name>'")
            sys.exit(1) # Exit gracefully
            
    except Exception as e:
        print("Could not connect to Ollama daemon. Is Ollama running?")
        sys.exit(1)

    agent = MemoryAgent(model_name=args.model)
    agent.chat_loop()

if __name__ == "__main__":
    main()