
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
import nltk

def ExtractHeadline(df):
    # Extract headline column for analysis
    headlines = df['headline'].dropna().tolist()
    return headlines



def PreprocessText(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation except for letters and spaces

    # Handle contractions (optional)
    text = text.replace("won't", "will not")
    text = text.replace("can't", "cannot")

    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)

    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)

    return text

def ClassifySentiment(score):
    if score <= -0.5:
        return 'Very Negative'
    elif score <= -0.0001:
        return 'Negative'
    elif score < 0.5:
        return 'Neutral'
    elif score < 1:
        return 'Positive'
    else:
        return 'Very Positive'

def GetSentimentScore(text):
    return TextBlob(text).sentiment.polarity

def sentimentScore(df):
    # Plotting the distribution of sentiment scores
    plt.figure(figsize=(12, 6))
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show() 
def SentimentClass(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_class', data=df, palette='viridis')
    plt.title('Distribution of Sentiments in Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.show()  


def ComputeTfidf(preprocessed_texts, max_features=1000, stop_words='english'):
    """
    Compute the TF-IDF matrix for a list of preprocessed texts.
    
    Args:
        preprocessed_texts (list of str): List of preprocessed text data.
        max_features (int): Maximum number of features for TF-IDF.
        stop_words (str or list): Stop words to exclude from TF-IDF computation.

    Returns:
        tuple: TF-IDF matrix, feature names, and vocabulary.
    """
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    X = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()
    vocabulary = vectorizer.vocabulary_
    return X, feature_names, vocabulary

def ExtractTopKeywords(tfidf_matrix, vocabulary, top_n=10):
    """
    Extract the top N keywords based on their summed TF-IDF scores.
    
    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix.
        vocabulary (dict): Vocabulary mapping words to indices.
        top_n (int): Number of top keywords to extract.

    Returns:
        list of tuples: Top N keywords and their summed TF-IDF scores.
    """
    sum_words = tfidf_matrix.sum(axis=0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vocabulary.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    return word_freq[:top_n]

def DisplayKeywords(keywords):
    """
    Display keywords with their corresponding scores.

    Args:
        keywords (list of tuples): Keywords and their scores.
    """
    print("Top Keywords:")
    for word, freq in keywords:
        print(f"{word}: {freq}")



def ConvertTopKeywordsToDF(word_freq, top_n=20):
    """
    Visualize the top N keywords and their TF-IDF scores using a bar plot.

    Args:
        word_freq (list of tuples): List of tuples containing words and their TF-IDF scores.
        top_n (int): Number of top keywords to visualize.

    Returns:
        None: Displays the bar plot.
    """
    # Create a DataFrame for the top N keywords
    top_keywords_df = pd.DataFrame(word_freq[:top_n], columns=['Word', 'TF-IDF Score'])

    # Visualize the keywords using a bar plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='TF-IDF Score', y='Word', data=top_keywords_df, palette='viridis')
    plt.title(f'Top {top_n} Keywords by TF-IDF Score')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Keyword')
    plt.tight_layout()
    plt.show()





def VisualizeKeywords(word_freq, top_keywords, wordcloud_title='Word Cloud of Keywords', barplot_title='Top Keywords by TF-IDF Score'):
    """
    Generate a word cloud and a bar plot for the given keywords and their TF-IDF scores.

    Args:
        word_freq (list of tuples): List of tuples [(word, score), ...] for generating the word cloud.
        top_keywords (list of tuples): List of top N tuples [(word, score), ...] for the bar plot.
        wordcloud_title (str): Title for the word cloud visualization.
        barplot_title (str): Title for the bar plot visualization.

    Returns:
        None: Displays the word cloud and bar plot.
    """
    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_freq))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(wordcloud_title)
    plt.show()

    # Bar Plot for Top Keywords
    top_words, top_freqs = zip(*top_keywords)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_words, y=top_freqs, palette='viridis')
    plt.title(barplot_title)
    plt.xlabel('Keywords')
    plt.ylabel('TF-IDF Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def WordPerTopic(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))