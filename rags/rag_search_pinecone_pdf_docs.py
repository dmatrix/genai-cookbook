
import sys 
sys.path.insert(0, "llm-prompts")
import os
from anthropic import Anthropic

from  llm_clnt_factory_api import ClientFactory, get_commpletion
from rag_utils import print_matches, extract_matches
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv

#
# Example code to search the Pinecode index for similarity search
# of the PDF document indexed.
#
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOP_K = 5
    # Set up Pinecone environment. Use the .env file to load the Pinecone API key
    # and the environment name, which is "gcp-starter" in this case, for the GCP starter environment.
    # a community edition of Pinecone is also available for free.
    _ = load_dotenv(find_dotenv())
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key is None:
        raise ValueError("Please set the PINECONE_API_KEY environment")
    pc = Pinecone(
        api_key=pinecone_api_key,
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
    print_matches(results)
    print('-' * 50)

    # Extract the context from the results for our LLM query
    context = "".join(extract_matches(results))

    # Construct our next query for the LLM model
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    MODEL = os.getenv("MODEL")
    print('-' * 50)
    print(f"Using MODEL={MODEL}; base={'Anthropic'}")

    # create an Anthropic client instance using our
    # factor method 
    client_factory = ClientFactory()
    client_type = "anthropic"
    client_factory.register_client(client_type, Anthropic)
    client_kwargs = {"api_key": anthropic_api_key}
    # create the client
    client = client_factory.create_client(client_type, **client_kwargs)

    # create system and user prompt
    system_content = """You are master of all knowledge, and a helpful sage.
                    You must summarize content given to you by drawing from your vast
                    knowledge about history, literature, science, social science, philosophy, religion, economics, 
                    sports, etc. Do not make up any responses. Only provide information that is true and verifiable.
                  """
    
    user_content = f"""What are the key takeaways for AI in 2023?,
                        given the {context}. Only provide information that is true and verifiable and use
                        the given context to provide the answer.
                        Summary: 
                     """
    
    # get the response
    BOLD_BEGIN = "\033[1m"
    BOLD_END   =   "\033[0m"
    response = get_commpletion(client, MODEL, system_content, user_content)
    response = response.replace("```", "")
    print(f"\n{BOLD_BEGIN}Prompt:{BOLD_END} {user_content}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
    