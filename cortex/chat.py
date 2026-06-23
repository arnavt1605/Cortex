import sys
import os
import json
import time
import subprocess
from pathlib import Path
import argparse
import ollama
from cortex.storage import SecureMemoryDB
from cortex.extractor import MemoryExtractor
from pyfiglet import  figlet_format

CORTEX_DIR = Path.home() / ".cortex"
QUEUE_DIR = CORTEX_DIR / "queue"
LOCK_FILE = QUEUE_DIR / "worker.lock"

class MemoryAgent:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name= model_name
        self.db= SecureMemoryDB()
        self.extractor = MemoryExtractor(model_name= model_name)

        self.transcript= []

        self.memory_enabled= True  # To toggle incognito mode on and off

        QUEUE_DIR.mkdir(parents=True, exist_ok=True)


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
            role= "user" if line.startswith("User:") else "assistant"

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
                    for id, mem in memories:
                        print(f"{id}. {mem}")
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

            elif user_input.lower().startswith('delete memory '):
                id= user_input[14:].strip()

                if not id.isdigit():
                    print("Please specify a valid memory ID number.")
                    continue
                    
                id_int = int(id)

                if self.db.delete_memory(id_int):
                    print(f"Successfully deleted memory entry [{id_int}].")
                else:
                    print(f"Error: Memory ID [{id_int}] does not exist")
                continue

            elif user_input.lower() == 'memory off':
                self.memory_enabled = False
                print("Incognito Mode Active: Memory retrieval and storage are suspended for this session")
                continue
                
            elif user_input.lower() == 'memory on':
                self.memory_enabled = True
                print("Memory Mode Active: Long term memory tracking resumed")
                continue




            self.transcript.append(f"User: {user_input}")

            if self.memory_enabled:
                system_prompt = self.build_system_prompt(user_input)
            else:
                system_prompt = "You are a helpful AI assistant. (Note: Personal memory access is currently disabled by the user)"
            chat_history = self.get_recent_memory()
            
            # Combine the system prompt and chat history
            messages = [{"role": "system", "content": system_prompt}] + chat_history

            ai_response = self.stream_response(messages)
            
            if ai_response:
                self.transcript.append(f"AI: {ai_response}")
    

    def end_session(self):
        #Runs when the user types exit or quit and triggers the Extractor 

        if not self.memory_enabled:
            print("\nSession closed. Incognito mode was active so no chat analysis performed.")
            return
        if not self.transcript:
            print("\n\nSession closed. No conversation took place.")
            return
            
        print("\n\nClosing session instantly")

        timestamp= int(time.time())
        payload= {
            "model_name": self.model_name,
            "transcript": self.transcript
        }

        transcript_path= QUEUE_DIR/f"transcript_{timestamp}.json"

        # Write the file to the queue folder instead of sending the transcript to the LLM
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        
        if LOCK_FILE.exists():
            print("Background sync active. Transcripts saved")
            return
        

        print("Spawning background memory worker...")
        try:
            #Popen() creates a second proces and start_new_session=True makes that second process independent from the parent
            subprocess.Popen(
                ["python", "-m", "cortex.worker"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True 
            )
        except Exception as e:
            print(f"Failed to spawn background worker automatically: {e}")


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
            sys.exit(1) 
            
    except Exception as e:
        print("Could not connect to Ollama daemon. Is Ollama running?")
        sys.exit(1)

    agent = MemoryAgent(model_name=args.model)
    agent.chat_loop()

if __name__ == "__main__":
    main()