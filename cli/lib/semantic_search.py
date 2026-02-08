from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("must have text")
        return self.model.encode([text])[0]

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(embedding.shape)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    ss = SemanticSearch()
    print(f"model loaded: {ss.model}")
    print(f"max seq length: {ss.model.max_seq_length}")