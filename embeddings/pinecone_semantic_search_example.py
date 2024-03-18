import os
from pinecone import Pinecone, PodSpec
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm

def extract_and_print_matches(results):
    for result in results['matches']:
        print(f"Score  : {round(result['score'], 2)}")
        print(f"Matches: {result['metadata']['text']}")
        print('-' * 50)
    
if __name__ == "__main__":
    
    # Load the dataset, only the first 50k samples
    dataset = load_dataset("imdb", split='train[:50000]')
    print(dataset[:1])

    reviews = []
    for record in dataset['text']:
        reviews.extend(record.split('\n'))
    reviews = list(set(reviews))
    print('\n'.join(reviews[:2]))
    print('-' * 50)
    print(f'Number of reviews: {len(reviews)}')

    # Load the sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Try encoding a sample review
    embeddings = model.encode(reviews[0:1])
    print(f"vector shape: {embeddings.shape}; vector length:{len(embeddings[0])}")

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

    # check if index exists in pinecone
    index_name = "starter-index"
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    if index_name in existing_indexes:
        print(f"Index {index_name} already exists. Deleting it.")
        pc.delete_index(index_name)

    print('-' * 50)
    # Create a new index
    print(f"Creating a new index {index_name}...")
    pc.create_index(name=index_name,
            metric="cosine",
            dimension=embeddings.shape[1],
            spec=PodSpec(environment="gcp-starter")
    )
    # Connect to the index
    pindex = pc.Index(index_name)

    # Insert the embeddings into the index
    print("Upserting the embeddings into the index...")
    batch_size = 500
    for i in tqdm(range(0, len(reviews), batch_size)):
        i_end = min(i+batch_size, len(reviews))
        # create IDs each batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'text': text} for text in reviews[i:i_end]]
        batch = reviews[i:i+500]
        embeddings = model.encode(batch)
        records = zip(ids, embeddings, metadatas)
        print(f"Upserting {i} to {i_end} records...")
        # upsert to Pinecone
        pindex.upsert(vectors=records)

    print('-' * 50)
    # Check the index stats
    print(pindex.describe_index_stats())

    # Start a semantic search
    print('-' * 50)
    print("Running a semantic search...")
    query = """This is a classic espionage thriller. I loved the movie, it was capitivating, 
            the plot brilliant, based on a true story, the characters were well developed,
            and their actions unpredictable. The actors were amazing, and the direction of plot was
            very well thought out. Recommended to everyone if you love clock and dagger
            twists and turns of cold war dramas and betrayals, and if you relish how John Le Carre spins his plots 
            in his absorbing novels on cold war espionage tales of spooks and crooks, you shall throughly enjoy this one!"""
    print(f"Query: {query}")
    query_embedding = model.encode(query).tolist()
    results = pindex.query(vector=query_embedding, top_k=5,
                           include_values=False, include_metadata=True)
    

    print('-' * 50)
    print("Top 5 results for the query:")
    extract_and_print_matches(results)

    # Delete the index
    print(f"Deleting the index {index_name}...")
    pc.delete_index(index_name)
    print("Done!")