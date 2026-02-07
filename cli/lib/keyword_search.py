from lib.search_utils import load_movies
import string

def clean_text(text):
  text = text.lower()
  text = text.translate(str.maketrans("", "", string.punctuation))
  return text

def tokenize_text(text):
  text = clean_text(text)
  # only if token exists
  tokens = [tok for tok in text.split() if tok]
  return tokens

def has_matching_token(query_toks, movie_toks):
  for query_tok in query_toks:
    for movie_tok in movie_toks:
      if query_tok in movie_tok:
        return True
  return False

def search_command(query, n_results):
  movies = load_movies()
  res = []
  query = clean_text(query)
  query_toks = tokenize_text(query)
  for movie in movies:
    movie_toks = tokenize_text(movie['title'])
    if has_matching_token(query_toks, movie_toks):
      res.append(movie)
    if len(res) == n_results:
      break
  return res
