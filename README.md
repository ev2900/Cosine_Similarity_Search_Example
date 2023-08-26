# Hugging Transformer Based Embeddings and Cosine Similarity Search Example

This example demonstrates how to transform text into embeddings via. Hugging Face sentence transform library. Once the text is represented as embeddings cosine similarity search can determine which embeddings are most similar to a search query

## What are Text Embeddings

Embeddings are numerical representations of text data that capture the semantic meaning and relationships between words, sentences, or documents. The numerical representation (vectors in a high-dimensional space) of the text data (sentences, paragraphs, documents ...) are constructed in a way that words or texts with similar meanings/contexts are represented closer to each other in the embedding space

There are several methods to generate text embeddings

1. **Word Embeddings** represent of individual words as vectors. Popular techniques include [Word2Vec](https://www.tensorflow.org/text/tutorials/word2vec), [GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation), and [FastText](https://fasttext.cc/)
   
2. **Sentence and Document Embeddings** aim to represent the meaning of entire sentences or documents. Sentence/document embeddings are generated using methods like averaging the word embeddings, using recurrent neural networks (RNNs) or convolutional neural networks (CNNs) or more recently, using transformer based models
   
3. **Transformer Based Embeddings** models, such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), [GPT](https://en.wikipedia.org/wiki/GPT-3) can create contextual embeddings. These embeddings consider the surrounding context of each word in a sentence and can result in richer, more nuanced representations

The example in this repository uses a transformer based approach for converting text to embeddings. The example uses the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) provided by hugging face.

## What is Cosine Similarity Search

## Example in Python
