from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

#input sentences
sentences = ['The cat sat on a mushy mat',
             'The cat stretched and laid down on a furry mat']

#encode sentences
embeddings = model.encode(sentences)

#output embeddings
for phrase, embedding in zip(sentences, embeddings):
    print("Phrase:", phrase)
    print("Embedding shape:", embedding.shape) 
    print("Embedding:", embedding[0:5])
print("--" * 40)

# compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cs = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print (f"Cosine similarity: {cs:.4f}")



