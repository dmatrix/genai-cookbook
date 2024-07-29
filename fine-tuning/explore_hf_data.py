import pandas as pd
from datasets import load_dataset
import uuid

"""
Example code to explore the dataset from the HuggingFace Hub, derived
from the Red Dot Design Award Product Description dataset and convert it into a pandas dataframe.
Code is from the following source:
https://huggingface.co/datasets/xiyuez/red-dot-design-award-product-description
Databricks Book on GenAI. (2021). Fine-tuning a pre-trained model with the Hugging Face library.
"""

if __name__ == "__main__":

    #Load the dataset from the HuggingFace Hub
    rd_ds = load_dataset("xiyuez/red-dot-design-award-product-description")
    pd.set_option('display.max_columns', None)  
    # convert the dataset into a pandas dataframe
    df = pd.DataFrame(rd_ds['train'])
    # print column names
    print(df.columns)
    print("------------------")

    # print the first product, category, description, and text
    print(f"product:{df['product'][0]}")
    print(f"category:{df['category'][0]}")
    print(f"description:{df['description'][0]}")
    print("------------------")
    print(f"text: {df['text'][0]}")
    print("------------------")

    # Combine the two attributes into an instruction string
    df['instruction'] = 'Create a detailed description for the following product: '+ df['product']+', belonging to category: '+ df['category']

    df = df[['instruction', 'description']]
    # Get a 5000 sample subset for fine-tuning purposes
    df_sample = df.sample(n=5000, random_state=42)
    print(df_sample.columns)
    print("------------------")
    print(df_sample.head())
    print("------------------")

    # Define template and format data into the template 
    # for supervised fine-tuning
    template = """Below is an instruction that describes a task. 
                  Write a response that appropriately completes the
                  request.
                  ### Instruction:
                  {}
                  ### Response:\n"""

    df_sample['prompt'] = df_sample["instruction"].apply(lambda x: template.format(x))
    df_sample.rename(columns={'description': 'response'}, inplace=True)
    df_sample['response'] = df_sample['response'] + "\n### End"
    df_sample = df_sample[['prompt', 'response']]
    # print df_sample column names
    print(f" Modified column names: {df_sample.columns}")
    print("------------------")
    print(df_sample.head(2))
    print("------------------")
    print(f"prompt:{df_sample['prompt'][0]}")
    print(f"response:{df_sample['response'][0]}")
    print("------------------")

    # # test putting garbage in one of the original columns
    # # Create a copy of the original dataframe
    # df_copy = df.copy()
    # # Put garbage in the first row of the prompt column
    # # using a lambda function
    # print("before lambada ------------------")
    # print(df_copy.columns)
    # print("after lambada ------------------")
    # df_copy['prompt'] = df_copy['description'].apply(lambda x: len(x))
    # print(df_copy.columns)
    # print(df_copy.head(3))

    
    print(f"dataset-{uuid.uuid4()}")
    