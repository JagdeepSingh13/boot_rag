from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from lib.search_utils import load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None  # movies
        self.document_map = {}

        self.embeddings_path = Path("cache/movie_embeddings.npy")

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("must have text")
        return self.model.encode([text])[0]
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qry_emb = self.generate_embedding(query)

        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(qry_emb, doc_emb)
            similarities.append((_similarity, doc))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        # [{sc, doc['title'], doc['description']} for sc, doc in similarities[:limit]]
        res = []
        for sc, doc in similarities[:limit]:
            res.append({'score': sc, 
                        'title': doc['title'],
                        'description':  doc['description']
                    })
        return res

def search(query, limit=5):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_res = ss.search(query, limit)
    for idx, r in enumerate(search_res):
        print(f"{idx}. {r['title']} (score: {r['score']})")
        print(r['description'][:100])

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Dimensions: {embedding.shape}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")    
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model():
    ss = SemanticSearch()
    print(f"model loaded: {ss.model}")
    print(f"max seq length: {ss.model.max_seq_length}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)