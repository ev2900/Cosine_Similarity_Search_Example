# Hugging Transformer Based Embeddings and Cosine Similarity Search Example

<img width="85" alt="map-user" src="https://img.shields.io/badge/views-313-green"> <img width="125" alt="map-user" src="https://img.shields.io/badge/unique visits-108-green">

This example demonstrates how to transform text into embeddings via. Hugging Face sentence transform library. Once the text is represented as embeddings cosine similarity search can determine which embeddings are most similar to a search query

## What are Text Embeddings

Embeddings are numerical representations of text data that capture the semantic meaning and relationships between words, sentences, or documents. The numerical representation (vectors in a high-dimensional space) of the text data (sentences, paragraphs, documents ...) are constructed in a way that words or texts with similar meanings/contexts are represented closer to each other in the embedding space

There are several methods to generate text embeddings

1. **Word Embeddings** represent of individual words as vectors. Popular techniques include [Word2Vec](https://www.tensorflow.org/text/tutorials/word2vec), [GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation), and [FastText](https://fasttext.cc/)

2. **Sentence and Document Embeddings** aim to represent the meaning of entire sentences or documents. Sentence/document embeddings are generated using methods like averaging the word embeddings, using recurrent neural networks (RNNs) or convolutional neural networks (CNNs) or more recently, using transformer based models

3. **Transformer Based Embeddings** models, such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), [GPT](https://en.wikipedia.org/wiki/GPT-3) can create contextual embeddings. These embeddings consider the surrounding context of each word in a sentence and can result in richer, more nuanced representations

The example in this repository uses a transformer based approach for converting text to embeddings. The example uses the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) provided by hugging face.

## What is Cosine Similarity Search

Cosine similarity measures the cosine of the angle between two vectors. This indicates how similar their directions are regardless of their magnitudes. Explained simply cosine similarity can quantify how similar two vectors are.

In search applications if we represent our search query as vector(s) and our search-able text as vector(s). we can determine which search-able text is most similar to our search query.

## Example in Python

### Install sentence_transformers library

```pip install sentence_transformers``` or in a Juypter notebook run ```!pip install sentence_transformers```

To confirm that the installation was successful import the BM250kapi item from the rank_bm25 library run

```from sentence_transformers import SentenceTransformer, util```

### Create corpus of documents

In this example we have a corpus of 4 documents

```
corpus = [
    "does this work with xbox?",
    "Does the M70 work with Android phones?",
    "does this work with iphone?",
    "Can this work with an xbox "
]
```

### Download transformer model and encode documents
```
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(corpus)
```

### Encode a search query
```
search_query = "M70 Android"
search_query_encode = model.encode(search_query)
```

### Compute cosine similarity between search query and all sentencase
```
cos_sim = util.cos_sim(search_query_encode, embeddings)
```

# Return the most similar document to the search query
```
similarity_scores = []
for i in range(list(cos_sim.size())[1]):
	similarity_scores.append(cos_sim[0][i].item())

print(corpus[similarity_scores.index(max(similarity_scores))])

---- result below ----

Does the M70 work with Android phones?

```
