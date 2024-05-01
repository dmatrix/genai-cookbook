import dspy

SENTIMENTS = [
        "This movie is a true cinematic gem, blending an engaging plot with superb performances and stunning visuals. A masterpiece that leaves a lasting impression.",
         "Regrettably, the film failed to live up to expectations, with a convoluted storyline, lackluster acting, and uninspiring cinematography. A disappointment overall.",
         "The movie had its moments, offering a decent storyline and average performances. While not groundbreaking, it provided an enjoyable viewing experience.",
         "This city is a vibrant tapestry of culture, with friendly locals, historic landmarks, and a lively atmosphere. An ideal destination for cultural exploration.",
         "The city's charm is overshadowed by traffic congestion, high pollution levels, and a lack of cleanliness. Not recommended for a peaceful retreat.",
         "The city offers a mix of experiences, from bustling markets to serene parks. An interesting but not extraordinary destination for exploration.",
         "This song is a musical masterpiece, enchanting listeners with its soulful lyrics, mesmerizing melody, and exceptional vocals. A timeless classic.",
         "The song fails to impress, featuring uninspiring lyrics, a forgettable melody, and lackluster vocals. It lacks the creativity to leave a lasting impact.",
         "The song is decent, with a catchy tune and average lyrics. While enjoyable, it doesn't stand out in the vast landscape of music.",
         "A delightful cinematic experience that seamlessly weaves together a compelling narrative, strong character development, and breathtaking visuals.",
         "This film, unfortunately, falls short with a disjointed plot, subpar performances, and a lack of coherence. A disappointing viewing experience.",
         "While not groundbreaking, the movie offers a decent storyline and competent performances, providing an overall satisfactory viewing experience.",
         "This city is a haven for culture enthusiasts, boasting historical landmarks, a rich culinary scene, and a welcoming community. A must-visit destination.",
         "The city's appeal is tarnished by overcrowded streets, noise pollution, and a lack of urban planning. Not recommended for a tranquil getaway.",
         "The city offers a diverse range of experiences, from bustling markets to serene parks. An intriguing destination for those seeking a mix of urban and natural landscapes.",
         "Mr. Xxx Yyy Zzz was curious and dubious"
]

SUMMARY = """
         The emergence of large language models (LLMs) has marked a significant 
         breakthrough in natural language processing (NLP), leading to remarkable 
         advancements in text understanding and generation. 
         
         Nevertheless, alongside these strides, LLMs exhibit a critical tendency 
         to produce hallucinations, resulting in content that is inconsistent with 
         real-world facts or user inputs. This phenomenon poses substantial challenges 
         to their practical deployment and raises concerns over the reliability of LLMs 
         in real-world scenarios, which attracts increasing attention to detect and 
         mitigate these hallucinations. In this survey, we aim to provide a thorough and 
         in-depth  overview of recent advances in the field of LLM hallucinations. 
         
         We begin with an innovative taxonomy of LLM hallucinations, then delve into the 
         factors contributing to hallucinations. Subsequently, we present a comprehensive
         overview of hallucination detection methods and benchmarks. 
         Additionally, representative approaches designed to mitigate hallucinations 
         are introduced accordingly. 
         
         Finally, we analyze the challenges that highlight the current limitations and 
         formulate open questions, aiming to delineate pathways for future  research on 
         hallucinations in LLMs 
    """
class ClassifyEmotion(dspy.Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence = dspy.InputField()
    sentiment = dspy.OutputField()

class SummarizeText(dspy.Signature):
    """Summarize a given text into a concise summary, in no less
    than ten sentence simple and compound sentences."""

    text = dspy.InputField()
    summary = dspy.OutputField()

if __name__ == "__main__":

    # Setup Ollama environment
    ollama_mistral = dspy.OllamaLocal(model='mistral')
    dspy.settings.configure(lm=ollama_mistral)

    for sentence in SENTIMENTS:
        classify = dspy.Predict(ClassifyEmotion)
        print(f"sentiment: {sentence}")
        print(classify(sentence=sentence))
        print("-------------------")

    # Use class signatures for summarization
    summarize = dspy.Predict(SummarizeText)
    print("Summary:")
    print(summarize(text=SUMMARY))
