
# A brief intuitive guide to vector embeddings

Embeddings, particulary text or word embeddings, are not novel. Use of the term word embeddings was originally coined by Bengio et al. in 2003, who trained them in a neural language model together with the model’s parameters.[1] Since then,
their representation, algorithms, and models to compute embeddings are predomintly used in natural language processing (NLP) tasks, and more recently,
in generative AI applications. Originally confined to use cases for semantic textual searches, rather than traditional string or word pattern matching, embeddings are used in other semantic searches for images, audio, and videos.

Because computers deal efficiently with numerical representation of any structured or unstructured data, embeddings are created as data's numerical vectors, projected into a latent multi-dimensional space, capturing and preserving data's semantic meaning. 

Another way to put it, in mathematical terms, Kevin Henner describes vectors as "an embedding [in vector] space or latent space, as a manifold in which similar items are positioned closer to one another than less similar items. In this case, sentences [or data entities] that are semantically similar should have similar embedded vectors and thus be closer together in the space." [2]

<img src="images/vector_space.png">

[source](https://causewriter.ai/courses/ai-explainers/lessons/vector-embedding/)

As projected vectors into a latent space, computers can then perform fast mathematical operations on an array or a matrix of vectors. Python's `numpy`
library offers efficient vector operations, such as computing the distance or proximity of vectors close to each other.

## What are vectors embedding 

From above, it follows that a vector is a single dimensional or multi-dimensional array of sequence of numbers, where each number may corrospond to a data's semantic meaning in relation to the other numbers in the sequence. For example, `vector_1 = [0.5, 1.75, -2, ..., 0.9]` could caputure the semantic meaning of "The cat sat on a cushy mat." Another vector,  `vector_2 = [0.5, 1.65, -1, ..., 0.9]`, could capture the meaning of "The cat sprawled and laid down on a furry mat."

The above two vector embeddings are deemed relatively similar if they capture semantic similarity. In the above case, this semantic similarity of two vectors is expressed as the shortest distance between them when they are projected in a vector space. From above we can confidently derive that similar vector embeddings will congregate into clusters, and, therefore, we can compute all its K=N nearest neighbors within a cluster of similar vector embeddings.

To compute the shortest distance between two vectors, cosine similarity is a common method; another one is Euclidean distance. [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) measures the cosine of the angle between two vectors and ranges from -1 to 1, where 1 means the vectors are identical, -1 means they are opposites, and 0 means they are orthogonal (i.e., unrelated).[3]

<img src=images/cosine_similarity.png> 

[source](https://www.learndatasci.com/glossary/cosine-similarity)

Let's take our above simple examples of the "cat on the mat", and compute its
cosine similarity to ascertain if they are semantically similar.

```
import numpy as np

v1 = np.array([0.5, 1.75, -2, ..., 0.9])  # Replace '...' with the remaining elements
v2 = np.array([0.5, 1.65, -1, ..., 0.9])   # Replace '...' with the remaining elements

# Compute the length of each vector
length_v1 = np.linalg.norm(v1)
length_v2 = np.linalg.norm(v2)

print("Length of v1:", length_v1)  # Length of v1: 2.85
print("Length of v2:", length_v2)  # Length of v2: 2.17

# Compute cosine similarity
cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Cosine similarity:", cos_sim) # Cosine similarity: 0.954
```
The cosine similarity 0.954 ~= 1, hence they are similar in semantics.

## How vector embeddings are created
Vector embeddings are generated using machine learning, first, by creating a model. A model is trained to convert various types of training data into numerical vectors. Then, finally, once the model is trained, it can be
used to generate a vector embedding for your data as its input and vector
embedding as its output.

The acutal process of training an embeddinbg model is beyond the scope of this discussion. For your edification, plenty of literature describe, at length and mathemetical foundations, how the models are trained, how the training data is employed and converted into pairs of vector embeddings, and neural network architecture used for training the model.

Suffice it to say, some of these pretrained models are publicly available for you to use. That is, convert your data a vector embeddings and vice versa.

 * [word2vec](https://codeblockhub.com/python/word2vec-models-python/)
 * [GloVe](https://codeblockhub.com/python/glove-models-python/)
 * [FastText](https://codeblockhub.com/python/fasttext-text-embedding-models/)
 * [BERT](https://blog.research.google/2018/11/open-sourcing-bert-state-of-art-pre.html)
 * [ELMo](https://en.wikipedia.org/wiki/ELMo)

 Other model vendors such as [HuggingFace](https://huggingface.co/models?other=embeddings), [Cohere](https://txt.cohere.com/introducing-embed-v3/), [Databricks](https://docs.databricks.com/en/generative-ai/vector-search.html), [Pinecone](https://www.pinecone.io/models/), [OpenAI](https://www.pinecone.io/learn/openai-embeddings-v3/) offer embedding models as a service.

## How vector embeddings are used today

It's no secret that embeddings have become popular because of deep learning and large language models. More recently, they've become essential in generative AI applications. They are at the core of how data are representated in these technologies.

Today, its use cases extends to: 
*  **semantic searches**: search engines go beyond just keyword or pattern matching; they go beyond text and include other data types--images, audio and video--allowing to search across multiple data formats for semantic similarities.
 * **Recommendation systems**: capture embeddings of articles, products, images, and recommend matching items selected by the user.
 * **Anomaly detection**: create an embedding of a target entity that does not match any entities seen before, indicating the target item as an anomaly.
 * **image search**: image similarity is a common use case in various scenarios, including nefarious surviellance. 
 * **Content moderation**: detect similarity of a social post to known examples of abuse stored as embeddings in an indexed vector database.
 * **Spam filtering or classification**: classify an email as an examples of spam mail.
 * **Conversational agent**: match existing embeddings in an indexed vector database that are semantically close to the user’s message or queries.
 * **Retrieval Augmentation Generation**: provide augmented proprietary data, combined with LLMs, for your generative AI, with more factual accuracy and less hallucination.

 All in all, vector embeddings are immensely poweful and useful, and their applications are at the core of numerous generative AI applications. For example, Pinecone tabulates some of the examples use cases.[6]

For a quick, getting started version of how to use vector embeddings for
semantic search, peruse below both notebook and Python application. 
 * Use the IMDB dataset reviews for a semantic search application. 
 * Understand how to use Pinecone Community Edition, aka Starter free edition. 
 
 To run either example, you'll need an account on Pinecone and an API key.

| Notebook Description| Open with Colab |
|--------------------|-----------------|
| Pinecone Semantic Search Example | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmatrix/genai-cookbook/blob/main/embeddings/1_pinecone_semantic_search_example.ipynb) |

| Python file  Description| View it on Github |
|-------------------------|-------------------|
| Pinecone Semantic Search Example | [Python App](https://github.com/dmatrix/genai-cookbook/blob/main/embeddings/pinecone_semantic_search_example.py) |

## Summary
To sum up, vector embeddings are numerical representations of data that capture semantic meaning. They are created using machine learning models and can be used for various applications such as semantic searches, recommendation systems, anomaly detection, image search, content moderation, spam filtering, conversational agents, and retrieval augmentation generation. Vector embeddings are powerful tools in generative AI applications. 

To get a flavor and intution of how to use with embeddings with a vector database such as Pinecone, have a go at the examples above.


## References and Resources

[1] https://aylien.com/blog/overview-word-embeddings-history-word2vec-cbow-glove

[2]https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/

[3] https://www.pinecone.io/learn/vector-embeddings-for-developers/

[4] https://en.wikipedia.org/wiki/Cosine_similarity

[5] https://www.learndatasci.com/glossary/cosine-similarity/

[6] https://docs.pinecone.io/page/examples

[7] https://realpython.com/chromadb-vector-database/

[8] https://causewriter.ai/courses/ai-explainers/lessons/vector-embedding/
