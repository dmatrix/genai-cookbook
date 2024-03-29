from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_and_print_matches(results):
    """
    Extract and print the matches from the results
    """
    for result in results['matches']:
        print(f"Score  : {round(result['score'], 2)}")
        print(f"Matches: {result['metadata']['text']}")
        print('-' * 50)

def read_pdf_chunks(file_path, chunk_size, chunk_overlap=0):
        """
        Read the pdf document and break them into its
        text chunks of chunk_size with a specified
        overlap
        """
        pdf_reader = PyPDFLoader(file_path)
        data = pdf_reader.load()
        # print (f'You have {len(data)} document(s) in your pdf')
        # print (f'You have {len(data[0].page_content)} characters in your first document')

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(data)
        
        # return a list of chucked texts from the spliited 
        # text
        return [t.page_content for t in texts]