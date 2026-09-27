import json
from pprint import pprint

from .search import Retrieval

from .index import Indexer

query = "open ai"
# Indexer.index()
Retrieval.search(query, 1)
