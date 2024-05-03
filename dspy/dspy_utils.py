import dspy

class QuestionAnswer(dspy.Signature):
    """Answer questions based on the input question."""

    question = dspy.InputField()
    answer   = dspy.OutputField()

class ClassifyEmotion(dspy.Signature):
    """Classify emotion among sadness, joy, love, anger, fear, surprise."""

    sentence  = dspy.InputField()
    sentiment = dspy.OutputField()

class SummarizeText(dspy.Signature):
    """Summarize a given text into a succinct summary, in no more
    than a combination of 10 or 15 simple, compound, complex, and 
    compound-complex complete sentences. Don't truncate the text.
    """

    text    = dspy.InputField()
    summary = dspy.OutputField()

class SummarizeTextAndExtractKeyTheme(dspy.Signature):
    """Summarize a given text succinctly, in no more
    than a combination of six simple, compound, complex, and 
    compound-complex sentences. Extract the key themes 
    from the text, label it as 'Key Subjects:', and 
    enumerate 'Takeaways:'.
    """

    text       = dspy.InputField()
    summary    = dspy.OutputField()
    key_themes = dspy.OutputField()
    takeaways  = dspy.OutputField()

class TranslateText(dspy.Signature):
    """Translate a given text in English language, translate to another language."""

    text      = dspy.InputField()
    language  = dspy.InputField()
    translated_text = dspy.OutputField()


class TextCompletion(dspy.Signature):
    """Complete a given text with more words to the best 
    of your acquired knowledge. Don't truncate the generated
    response.
    """
    
    in_text  = dspy.InputField()
    out_text = dspy.OutputField()

class TextTransformationAndCorrection(dspy.Signature):
    """Transform the given text on Pirate speak to a standard english text. 
    Correct any grammatical errors. Provide the corrected 
    text as the output.
    """

    text = dspy.InputField()
    corrected_text = dspy.OutputField()

class TextCorrection(dspy.Signature):
    """Correct the given text for any grammatical errors. 
    Provide the corrected text as the output.
    """

    text = dspy.InputField()
    corrected_text = dspy.OutputField()

class GenerateJSON(dspy.Signature):
    """Generate five distinct products on training shoes. 
       Generate products and format them all in a single JSON object.
       For each product, the JSON object should 
       contain items: Brand, Description, Size, Gender (Male, Female or Unisex), 
       Price, and Review (three customer reviews.
    """

    json_text = dspy.OutputField(desc='key-value pairs')