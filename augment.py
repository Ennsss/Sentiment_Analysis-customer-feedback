import pandas as pd
from transformers import pipeline
from nltk.corpus import wordnet
import random
import nltk
nltk.download('wordnet')

# Load a pre-trained paraphrasing model from Hugging Face
def get_paraphrase_pipeline():
    return pipeline("text2text-generation", model="t5-small")

paraphraser = get_paraphrase_pipeline()

def augment_data(input_file, output_file, num_augmented_samples):
    """
    Augments a dataset for sentiment analysis by adding noisy and paraphrased versions of the original samples.

    :param input_file: Path to the CSV input file containing 'review_text' and 'sentiment' columns.
    :param output_file: Path to save the augmented dataset as a CSV file.
    :param num_augmented_samples: Number of augmented samples to generate per original review.
    """
    # Load the original dataset
    df = pd.read_csv(input_file)

    augmented_data = []

    for _, row in df.iterrows():
        review = row['review_text']
        sentiment = row['sentiment']

        # Append the original review
        augmented_data.append({'review_text': review, 'sentiment': sentiment})

        # Generate augmented samples
        for _ in range(num_augmented_samples):
            augmented_review = augment_review(review)
            augmented_data.append({'review_text': augmented_review, 'sentiment': sentiment})

    # Save the augmented dataset
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_file, index=False)

def augment_review(review):
    """
    Augments a given text by paraphrasing and adding noise.

    :param review: Original text to augment.
    :return: Augmented version of the text.
    """
    # Use paraphrasing as the primary augmentation technique
    try:
        paraphrased = paraphraser(review, max_length=50, num_return_sequences=1)[0]['generated_text']
    except Exception:
        paraphrased = review  # Fallback to original if paraphrasing fails

    # Add noise by replacing some words with synonyms
    noisy_review = add_synonym_noise(paraphrased)

    return noisy_review

def add_synonym_noise(text):
    """
    Adds noise to a given text by replacing words with synonyms.

    :param text: Original text to add noise.
    :return: Noisy version of the text.
    """
    words = text.split()
    noisy_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.random() < 0.3:  # 30% chance to replace with a synonym
            noisy_words.append(random.choice(synonyms))
        else:
            noisy_words.append(word)

    return ' '.join(noisy_words)

def get_synonyms(word):
    """
    Fetches synonyms for a word using NLTK's WordNet interface.

    :param word: The word for which synonyms are required.
    :return: List of synonyms, or None if no synonyms are found.
    """
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            if lemma.name().lower() != word.lower():  # Avoid the word itself
                synonyms.append(lemma.name().replace('_', ' '))
    return list(set(synonyms)) if synonyms else None

# Example Usage
augment_data('training_split_1 - Copy.csv', 'augmented_training.csv', 3)
