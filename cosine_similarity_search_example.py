from sentence_transformers import SentenceTransformer, util

# Create corpus of documents
corpus = [
	"does this work with xbox?",
    "Does the M70 work with Android phones?", 
    "does this work with iphone?",
    "Can this work with an xbox "
]

# Download model for document encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all documents in corpus
embeddings = model.encode(corpus)
#print(embeddings)

# Encode search query
search_query = "M70 Android"
search_query_encode = model.encode(search_query)

# Compute cosine similarity between search query and all sentences 
cos_sim = util.cos_sim(search_query_encode, embeddings)

# Show similarity score of each document to the search query
'''
for i in range(list(cos_sim.size())[1]):
	print("Document: " + corpus[i])
	print("Similarity score to query: " + str(cos_sim[0][i].item()) + "\n")
'''

# Return the most similar document to the search query
similarity_scores = []
for i in range(list(cos_sim.size())[1]):
	similarity_scores.append(cos_sim[0][i].item())

print(corpus[similarity_scores.index(max(similarity_scores))])
