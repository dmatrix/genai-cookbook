
import os
from rag_utils import extract_and_print_matches
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

#
# Example code to search the Pinecode index for similarity search
# of the PDF document indexed.
#
if __name__ == "__main__":
    TOP_K = 5
    # Set up Pinecone environment. Use the .env file to load the Pinecone API key
    # and the environment name, which is "gcp-starter" in this case, for the GCP starter environment.
    # a community edition of Pinecone is also available for free.
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key is None:
        raise ValueError("Please set the PINECONE_API_KEY environment")
    pc = Pinecone(
        api_key=api_key,
        environment="gcp-starter",
        spec=PodSpec(environment="gcp-starter")
    ) 

    # Our starter pinecone index
    index_name = "starter-index"

     # Connect or get a handle to the index
    pindex = pc.Index(index_name)

    # create our model instance for encodin the query
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('-' * 50)

     # Start a semantic search
    print("Running a semantic search...")
    query = "What are the key takeaways for AI in 2023?"
    print(f"Query: {query}")
    query_embedding = model.encode(query).tolist()
    results = pindex.query(vector=query_embedding, top_k=TOP_K,
                           include_values=False, 
                           include_metadata=True)

    print('-' * 50)
    print(f"Top {TOP_K} results for the query:")
    extract_and_print_matches(results)
