"""
This script contains the functions needed to transform a company's description into low-dimensional embedding.
"""
import numpy as np
import re
from sentence_transformers import SentenceTransformer


# Computes embeddings for descriptions by transforming each sentence in a description using a sentence transformer model
# and then taking the average or maximum across all dimensions.
def get_description_embeddings(descriptions, max=True):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    description_embeddings = []
    for description in descriptions:
        sentences = re.split('\.|\!|\?', description)
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


def get_full_embeddings(descriptions):
#def get_description_embeddings(descriptions):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    description_embeddings = []
    min_sentences = 100
    max_sentences = 0
    for description in descriptions:
        sentences = re.split('\.|\!|\?', description)
        min_sentences = min(min_sentences, len(sentences))
        max_sentences = max(max_sentences, len(sentences))
        sentence_embeddings = []
        embedding_shape = None
        for sentence in sentences:
            sentence_embedding = model.encode(sentence)
            embedding_shape = sentence_embedding.shape
            sentence_embeddings.append(sentence_embedding)
        counter = 0
        while len(sentence_embeddings) < 15:
            sentence_embeddings.append(sentence_embeddings[counter])
            counter += 1
        flattened_embeddings = []
        for embedding in sentence_embeddings:
            flattened_embeddings += list(embedding)
        description_embeddings.append(flattened_embeddings)
    print(f"Min number of sentences: {min_sentences}")
    print(f"Max number of sentences: {max_sentences}")
    return description_embeddings
