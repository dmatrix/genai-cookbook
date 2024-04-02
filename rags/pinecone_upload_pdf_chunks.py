from rag_utils import read_pdf_chunks, extract_matches
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm
import os

# 
# Usage example of how to upload PDF to Pinecone, index it,
# and use it for semantic search query
#  

if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    DIR_PATH = os.path.join(home_dir, 'rags/pdfs')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    verbose = True
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 20

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
            dimension=384,
            spec=PodSpec(environment="gcp-starter")
    )

    # Connect or get a handle to the index
    pindex = pc.Index(index_name)

    # read each file in the directory
    for filename in tqdm(os.listdir(DIR_PATH)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(DIR_PATH, filename)
            print(f"Processing file: {file_path}")
            for i, chunk in enumerate(read_pdf_chunks(file_path, CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)):
                
                # Process each chunk (e.g., create vector embeddings)
                embeddings = model.encode(chunk)

                # create a metadata batch
                c_id = "".join([str(i), '-', filename])
                sample_doc = [
                    { "id":  c_id ,
                      "values": embeddings.tolist(),
                      "metadata": {
                          "text": chunk
                        }
                     }
                ]
                if verbose:
                    if i % 100 == 0:
                        print(f"Upserting batch id: { c_id}")
                        
                 # upsert to Pinecone
                pindex.upsert(sample_doc)
            print('---')

    print('-' * 50)

    # Check the index stats
    print(pindex.describe_index_stats())
    print(f"Creating start-index done!")

    # Send a query to the index to fetch the top 3 similar documents
    query = "What are the key takeaways for AI in 2023?"
    print(f"Query: {query}")    
    query_embedding = model.encode(query).tolist()
    results = pindex.query(vector=query_embedding, top_k=3,
                           include_values=False, 
                           include_metadata=True)
    print('-' * 50)
    print(f"Top 3 results for the query:")
    for result in results['matches']:
        print(f"Score  : {round(result['score'], 2)}")
        print(f"Matches: {result['metadata']['text']}")
        print('-' * 50)

    
