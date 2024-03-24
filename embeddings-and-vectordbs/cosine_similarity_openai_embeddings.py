import sys
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# Function to get the embeddings
def get_embedding(clnt, params):
    response = clnt.embeddings.create(input=params["documents"],
                                      model=params["model"])
    print(response)
    return response["embeddings"]

if __name__ == '__main__':
    
    # Load the environment variables
    _ = load_dotenv(find_dotenv())
    open_api_key = os.getenv("OPENAI_API_KEY")
    open_base_url = os.getenv("OPENAI_API_BASE")
    MODEL = os.getenv("MODEL")

    # Initialize the OpenAI API
    client = OpenAI(
    api_key = open_api_key,
    )

    print(f"Using Model: {MODEL}")
    print(f"Using Base URL: {open_base_url}")

    # input sentences
    sentences = ['The cat sat on a mushy mat',
                 'The cat stretched and laid down on a furry mat']
    
    # encode sentences
    params = {
        "model": "text-embedding-3-small",  # Specify the model
        "documents": sentences,       # Specify the text for which embeddings are to be generated
}
    embeddings = get_embedding(client, params)

    # output embeddings
    for phrase, embedding in zip(sentences, embeddings):
        print("Phrase:", phrase)
        print("Embedding shape:", embedding.shape) 
        print("Embedding:", embedding[0:5])

    print("--" * 40)

    # compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cs = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print (f"Cosine similarity: {cs:.4f}")
    