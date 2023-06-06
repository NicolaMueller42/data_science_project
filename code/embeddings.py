"""
This script contains the functions needed to transform a company's description into low-dimensional embedding.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# Computes embeddings for descriptions by transforming each sentence in a description using a sentence transformer model
# and then taking the average or maximum across all dimensions.
def get_description_embeddings(descriptions, max=True):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    description_embeddings = []
    for description in descriptions:
        sentences = description.split(".")
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embedding = model.encode(sentence)
            sentence_embeddings.append(sentence_embedding)
        if max:
            description_embedding = np.max(sentence_embeddings, axis=0)
        else:
            description_embedding = np.mean(sentence_embeddings, axis=0)
        description_embeddings.append(description_embedding)
    return description_embeddings