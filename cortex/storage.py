import sqlite3
import os
import re
import json
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from sentence_transformers import SentenceTransformer, CrossEncoder

class SecureMemoryDB:
    def __init__(self, folder_name=".cortex", db_name= "memory.db", key_name= "secret.key"):
        home_dir= os.path.expanduser("~") # return home directory
        base_path= os.path.join(home_dir, folder_name)

        os.makedirs(base_path, exist_ok=True)

        self.db_path= os.path.join(base_path, db_name)
        self.key_path= os.path.join(base_path, key_name)
        self.key= self.initialize_key()
        self.initialize_db()

        self.encoder= SentenceTransformer('all-MiniLM-L6-v2')

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def initialize_key(self):
        # Load a 256 bit AES key or generate a new one
        if not os.path.exists(self.key_path):
            key= get_random_bytes(32)
            with open(self.key_path, "wb") as key_file:
                key_file.write(key)
            return key

        else:
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
            
    
    def initialize_db(self):
        # Create the tables 
        self.conn= sqlite3.connect(self.db_path, check_same_thread= False)
        cursor= self.conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encrypted_text BLOB NOT NULL,
            vector_embedding TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()


    def encrypt_data(self, plaintext):
        # Encrypt using AES excryption
        cipher= AES.new(self.key, AES.MODE_GCM)
        ciphertext, tag= cipher.encrypt_and_digest(plaintext.encode('utf-8'))

        # [Nonce (16 bytes)] + [Tag (16 bytes)] + [Ciphertext (variable)]
        return cipher.nonce + tag + ciphertext
    
    def decrypt_data(self, encrypted):
        # Decrypt the ciphertext using AES
        nonce= encrypted[:16]
        tag= encrypted[16:32]
        ciphertext= encrypted[32:]

        cipher= AES.new(self.key, AES.MODE_GCM, nonce= nonce)

        try:
            decrypted_data= cipher.decrypt_and_verify(ciphertext, tag)
            return decrypted_data.decode('utf-8')
        except ValueError:
            raise ValueError("Data is corrupted")
        
    
    def add_memory(self, text):
        # Encrypts the text, generate an embedding vector and save both to the db
        encrypted= self.encrypt_data(text)

        embedding= self.encoder.encode(text).tolist()

        # Serialize the list to json so SQLite can store it in a TEXT column
        embedding_json= json.dumps(embedding)

        cursor= self.conn.cursor()
        cursor.execute('INSERT INTO memories (encrypted_text, vector_embedding) VALUES (?, ?)', 
                       (encrypted, embedding_json))
        self.conn.commit()

    def search_memories(self, query, top_k=3, fetch_k=10, threshold= 0.3):
        # Two stage search from vector search to cross encoder

        query_vector= self.encoder.encode(query)

        cursor= self.conn.cursor()
        cursor.execute('SELECT id, encrypted_text, vector_embedding FROM memories')
        rows= cursor.fetchall()

        if not rows:
            return []

        results= []
        for row_id, encrypted, embedding_json in rows:
            stored_vector= np.array(json.loads(embedding_json))

            similarity = np.dot(query_vector, stored_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(stored_vector))

            results.append({
                "id": row_id,
                "blob": encrypted,
                "score": float(similarity)
            })

        results.sort(key= lambda x: x["score"], reverse= True)

        broad_candidates= results[:fetch_k]

        # Semantic reranking using CrossEncoder
        candidate_texts= []
        for cand in broad_candidates:
            cand["text"]= self.decrypt_data(cand["blob"])
            candidate_texts.append(cand["text"])


        # Format pairs for the Cross-Encoder: [[query, text1], [query, text2], ...]
        cross_inputs = [[query, text] for text in candidate_texts]
        
        # The model reads the query and the text together and scores their logical connection
        cross_scores = self.reranker.predict(cross_inputs)

        # Attach the new scores and resort
        for i in range(len(broad_candidates)):
             broad_candidates[i]["rerank_score"] = float(cross_scores[i])

        broad_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_memories = []
        for cand in broad_candidates:
            # Check teh cross encoder score against the threshold
            if cand["rerank_score"] >= threshold:
                final_memories.append((cand["text"], cand["rerank_score"]))
                
            # Stoppped once we hit our top_k limit
            if len(final_memories) == top_k:
                break
            
        return final_memories
    
    def word_overlap(self, a, b):
        #Calculates Jaccard similarity (word overlap) between two strings
        words_a = set(re.findall(r'\w+', a.lower()))
        words_b = set(re.findall(r'\w+', b.lower()))
        union = words_a | words_b
        if not union:
            return 0.0
        return len(words_a & words_b) / len(union)

    def is_duplicate(self, text, threshold=0.85):
        #Advanced Deduplication: Vector Math + Word Overlap + Substring
        query_vector = self.encoder.encode(text)
        text_norm = text.lower().strip()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT encrypted_text, vector_embedding FROM memories')
        rows = cursor.fetchall()
        
        for encrypted_blob, embedding_json in rows:
            existing_text = self.decrypt_data(encrypted_blob)
            existing_norm = existing_text.lower().strip()
            
            # check 1 - substring in the other string
            if text_norm in existing_norm or existing_norm in text_norm:
                return True
                
            # check 2- jaccard overlap 
            if self.word_overlap(text_norm, existing_norm) > 0.60:
                return True
            
            # check 3 - standard vector math
            stored_vector = np.array(json.loads(embedding_json))
            similarity = np.dot(query_vector, stored_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(stored_vector))
            
            if similarity >= threshold:
                return True
                
        return False
            
    def get_all_memories(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, encrypted_text FROM memories ORDER BY timestamp ASC')
        rows = cursor.fetchall()
        
        memories = []
        for row_id, encrypted_blob in rows:
            plain_text = self.decrypt_data(encrypted_blob)
            memories.append((row_id, plain_text))
        
        return memories

    def clear_all_memories(self):
        #Permanently deletes all memories from the database
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM memories')
        self.conn.commit()

    def delete_memory(self, mem_id):
        cursor= self.conn.cursor()

        #Check if id exists
        cursor.execute("SELECT id from memories WHERE id = ?",(mem_id,))
        if not cursor.fetchone():
            return False
        
        cursor.execute("DELETE FROM memories WHERE id= ?",(mem_id,))
        self.conn.commit()

        return True

    



