import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, logging as transformers_logging
from wordcloud import WordCloud
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import numpy as np
import os
import time

# Run only once (Check if required NLTK data is already downloaded)
def download_nltk_data():
    try:
        stopwords.words('english')
        word_tokenize("Test")
        SentimentIntensityAnalyzer().polarity_scores("Test")
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)

# Download only if needed
download_nltk_data()

MAX_COMMENTS = 200

# ------------------ YOUTUBE SCRAPER ------------------
class YouTubeScraper:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def extract_video_id(self, url):
        match = re.search(r"(?:v=|youtu\.be/|embed/)([\w-]{11})", url)
        if not match:
            raise ValueError("Invalid YouTube video URL.")
        return match.group(1)

    def get_video_title(self, video_id):
        response = self.youtube.videos().list(part='snippet', id=video_id).execute()
        if not response.get('items'):
            raise ValueError("Video not found or is unavailable.")
        return response['items'][0]['snippet']['title']

    def fetch_comments(self, video_id, max_comments=MAX_COMMENTS):
        comments = []
        next_page_token = None
        retries = 0
        max_retries = 3

        try:
            while len(comments) < max_comments:
                try:
                    response = self.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=min(100, max_comments - len(comments)),
                        pageToken=next_page_token,
                        textFormat='plainText'
                    ).execute()

                    for item in response.get('items', []):
                        top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                        comments.append(top_comment)
                        if len(comments) >= max_comments:
                            break

                    next_page_token = response.get('nextPageToken')
                    if not next_page_token:
                        break

                except HttpError as e:
                    if e.resp.status == 403:
                        raise ValueError("The comments section is DISABLED for this YouTube video")
                    elif e.resp.status == 429:
                        if retries < max_retries:
                            retries += 1
                            wait_time = 2 ** retries
                            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise ValueError("YouTube API rate limit exceeded. Try again later.")
                    else:
                        raise e

            if not comments:
                print("Warning: No comments were found for this video.")

            return comments

        except HttpError as e:
            if e.resp.status == 403:
                raise ValueError("The comments section is DISABLED for this YouTube video")
            else:
                raise ValueError(f"YouTube API error: {str(e)}")

# ------------------ SENTIMENT ANALYZER ------------------
class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.negation_words = {"no", "not", "never", "none", "neither", "nor", "nobody", "nothing", "nowhere", "hardly", "scarcely", "barely", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"}
        self.positive_words = {"good", "great", "awesome", "amazing", "excellent", "fantastic", "wonderful", "love", "best", "perfect", "better", "beautiful", "helpful", "impressive", "like", "enjoy", "insane", "crazy"}
        self.negative_words = {"bad", "terrible", "awful", "horrible", "poor", "disappointing", "hate", "worst", "useless", "boring"}
        self.stop_words = self.stop_words - self.negation_words - self.positive_words - self.negative_words

        try:
            transformers_logging.set_verbosity_error()
            self.classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", batch_size=16)
        except Exception as e:
            print(f"Warning: Could not initialize transformer model: {str(e)}")
            print("Falling back to VADER sentiment analysis only.")
            self.classifier = None
        finally:
            transformers_logging.set_verbosity_warning()

        self.intensifiers = {"very", "extremely", "incredibly", "absolutely", "completely", "totally", "utterly", "highly", "especially", "particularly", "remarkably", "exceedingly", "tremendously", "immensely", "extraordinarily"}
        self.diminishers = {"slightly", "somewhat", "rather", "little", "barely", "hardly", "scarcely", "only", "just", "merely", "fairly", "sort of", "kind of", "quite"}

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text.lower())
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words or
                          word in self.negation_words or
                          word in self.positive_words or
                          word in self.negative_words]
        return ' '.join(filtered_tokens)

    def extract_ngrams(self, text, n=2):
        tokens = text.split()
        if len(tokens) < n:
            return []
        n_grams = list(ngrams(tokens, n))
        return [' '.join(gram) for gram in n_grams]

    def get_sentiment_context(self, text, window_size=4):
        tokens = text.split()
        sentiment_phrases = []

        if not tokens:
            return sentiment_phrases

        for i in range(len(tokens)):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            window = ' '.join(tokens[start:end])
            sentiment = self.vader.polarity_scores(window)
            if abs(sentiment['compound']) > 0.2:
                sentiment_phrases.append((window, sentiment['compound']))
        return sentiment_phrases

    def analyze_comment(self, comment):
        if not comment or not comment.strip():
            return {
                'adjusted_score': 0,
                'sentiment': "NEUTRAL",
                'positive_phrases': "",
                'negative_phrases': ""
            }

        processed_text = self.preprocess_text(comment)
        vader_scores = self.vader.polarity_scores(processed_text)
        compound_score = vader_scores['compound']
        transformer_normalized = 0

        if self.classifier and processed_text.strip():
            try:
                transformer_result = self.classifier(processed_text)[0]
                sentiment_label = transformer_result['label']
                confidence = transformer_result['score']
                transformer_normalized = (confidence * 2 - 1) * (1 if sentiment_label == "POSITIVE" else -1)
            except Exception as e:
                print(f"Warning: Transformer analysis failed for comment: {str(e)}")
                transformer_normalized = compound_score
        else:
            transformer_normalized = compound_score

        transformer_weight = 0.6 if self.classifier else 0
        vader_weight = 1.0 - transformer_weight
        adjusted_score = (transformer_normalized * transformer_weight) + (compound_score * vader_weight)

        if -0.05 <= adjusted_score <= 0.05:
            final_sentiment = "NEUTRAL"
        elif adjusted_score > 0:
            final_sentiment = "POSITIVE"
        else:
            final_sentiment = "NEGATIVE"

        sentiment_phrases = self.get_sentiment_context(processed_text)
        unigrams = processed_text.split()
        bigrams = self.extract_ngrams(processed_text, 2)
        trigrams = self.extract_ngrams(processed_text, 3)

        pos_phrases = []
        neg_phrases = []

        for phrase, score in sentiment_phrases:
            if score > 0.2:
                pos_phrases.append(phrase)
            elif score < -0.3:
                neg_phrases.append(phrase)

        for word in unigrams:
            if word in self.positive_words:
                pos_phrases.append(word)
            elif word in self.negative_words:
                neg_phrases.append(word)
            else:
                word_score = self.vader.polarity_scores(word)['compound']
                if word_score > 0.6 and word not in ' '.join(pos_phrases):
                    pos_phrases.append(word)
                elif word_score < -0.6 and word not in ' '.join(neg_phrases):
                    neg_phrases.append(word)

        # Add selective bigrams with strong sentiment
        for bigram in bigrams:
            # Only add if it contains intensifiers, negations, or has strong sentiment
            bigram_score = self.vader.polarity_scores(bigram)['compound']
            words = bigram.split()
            if (any(word in self.intensifiers for word in words) or
                any(word in self.negation_words for word in words) or
                abs(bigram_score) > 0.6):
                if bigram_score > 0 and bigram not in ' '.join(pos_phrases):
                    pos_phrases.append(bigram)
                elif bigram_score < 0 and bigram not in ' '.join(neg_phrases):
                    neg_phrases.append(bigram)

        return {
            'adjusted_score': adjusted_score,
            'sentiment': final_sentiment,
            'positive_phrases': ' '.join(pos_phrases),
            'negative_phrases': ' '.join(neg_phrases)
        }

    def analyze(self, comments):
        if not comments:
            raise ValueError("No comments provided.")

        all_results = []
        pos_phrases_all = []
        neg_phrases_all = []
        scores = []
        sentiments = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

        batch_size = 50
        for i in range(0, len(comments), batch_size):
            comment_batch = comments[i:i+batch_size]

            for comment in comment_batch:
                if not comment.strip() or '\n' in comment:
                    continue  # Skip empty and multi-line comments
                try:
                    result = self.analyze_comment(comment)
                    all_results.append(result)
                    sentiments[result['sentiment']] += 1

                    if result['sentiment'] != "NEUTRAL":
                        scores.append(result['adjusted_score'])
                        pos_phrases_all.append(result['positive_phrases'])
                        neg_phrases_all.append(result['negative_phrases'])
                except Exception as e:
                    print(f"Error analyzing comment: {str(e)}")

        if scores:
            significant_scores = [s for s in scores if abs(s) > 0.2]
            if significant_scores:
                weighted_scores = [self._sigmoid_weight(s) * s for s in significant_scores]
                aggregated_score = sum(weighted_scores) / sum(self._sigmoid_weight(s) for s in significant_scores)
            else:
                weighted_scores = [self._sigmoid_weight(s) * s for s in scores]
                aggregated_score = sum(weighted_scores) / sum(self._sigmoid_weight(s) for s in scores)
        else:
            aggregated_score = 0

        all_pos_phrases = ' '.join([p for p in pos_phrases_all if p])
        all_neg_phrases = ' '.join([p for p in neg_phrases_all if p])

        return {
            "positive_phrases": all_pos_phrases,
            "negative_phrases": all_neg_phrases,
            "total_score": aggregated_score,
            "sentiment_counts": sentiments,
            "comment_results": all_results
        }

    def _sigmoid_weight(self, x):
        scaled_x = 5 * abs(x)
        return 1 / (1 + np.exp(-scaled_x))

    def categorize_sentiment(self, score, counts=None):
        if counts and sum(counts.values()) == 0:
            return "No Comments Available 🤷‍♂️"

        if counts and sum(counts.values()) == counts["NEUTRAL"]:
            return "Entirely Neutral 🤷‍♂️"

        intensity = abs(score)

        if counts:
            total = sum(counts.values())
            if total == 0:
                return "No Valid Comments 🤷‍♂️"

            pos_ratio = counts["POSITIVE"] / total
            neg_ratio = counts["NEGATIVE"] / total

            if pos_ratio > 0.85:
                return "Overwhelmingly Positive 😍"
            elif neg_ratio > 0.85:
                return "Overwhelmingly Negative 🤬"

            if 0.85 > pos_ratio > 0.65:
                return "Strongly Positive 😍"
            elif 0.85 > neg_ratio > 0.65:
                return "Strongly Negative 😠"

            if pos_ratio > 0.5 and neg_ratio > 0.2:
                return "Slightly Positive 🙂"
            elif neg_ratio > 0.5 and pos_ratio > 0.2:
                return "Slightly Negative 🤨"
            elif counts["NEUTRAL"] > total * 0.75:
                return "Predominantly Neutral 😐"

            if pos_ratio > neg_ratio * 2 and pos_ratio > 0.4:
                return "Generally Positive 😃"
            elif neg_ratio > pos_ratio * 2 and neg_ratio > 0.4:
                return "Generally Negative 😠"
            elif max(pos_ratio, neg_ratio) < 0.4:
                return "Mixed Sentiments 🤔"

        if score >= 0.5:
            return "Strongly Positive 😍"
        elif 0.2 <= score < 0.5:
            return "Generally Positive 😃"
        elif 0.05 <= score < 0.2:
            return "Slightly Positive 🙂"
        elif -0.05 <= score < 0.05:
            return "Predominantly Neutral 😐"
        elif -0.2 <= score < -0.05:
            return "Slightly Negative 🤨"
        elif -0.5 <= score < -0.2:
            return "Generally Negative 😠"
        elif score < -0.5:
            return "Strongly Negative 🤬"
        else:
            return "Undetermined Sentiment"

# ------------------ VISUALIZER ------------------
class Visualizer:
    @staticmethod
    def wordcloud(phrases, title, max_words=100):
        if not phrases or phrases.strip() == "":
            print(f"No {title.lower()} to display.")
            return

        wc = WordCloud(
            width=1000,
            height=500,
            background_color='white',
            max_words=max_words,
            colormap='viridis' if 'Positive' in title else 'RdGy',
            contour_width=1,
            contour_color='steelblue' if 'Positive' in title else 'firebrick',
            collocations=True,
            normalize_plurals=True
        ).generate(phrases)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout(pad=1)
        plt.show()

    @staticmethod
    def sentiment_distribution(counts):
        labels = ['Positive', 'Neutral', 'Negative']
        values = [counts["POSITIVE"], counts["NEUTRAL"], counts["NEGATIVE"]]
        colors = ['#50C878', '#AFAFAF', '#FF6961']

        plt.figure(figsize=(9.25, 5))
        bars = plt.bar(labels, values, color=colors)

        total = sum(values)
        for bar, value in zip(bars, values):
            percentage = (value / total) * 100 if total > 0 else 0
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{percentage:.1f}%',
                ha='center',
                fontweight='bold'
            )

        plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Number of Comments')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def sentiment_score_chart(results):
        scores = [r['adjusted_score'] for r in results]
        if not scores:
            print("No sentiment scores to display.")
            return

        scores.sort()
        indices = range(len(scores))

        plt.figure(figsize=(9.25, 5))
        colors = []
        for s in scores:
            if s > 0.5:
                colors.append('#00A36C')  # Dark green for strongly positive
            elif s > 0.2:
                colors.append('#50C878')  # Medium green for moderately positive
            elif s > 0.05:
                colors.append('#98FB98')  # Light green for slightly positive
            elif s > -0.05:
                colors.append('#AFAFAF')  # Gray for neutral
            elif s > -0.2:
                colors.append('#FFA07A')  # Light red for slightly negative
            elif s > -0.5:
                colors.append('#FF6961')  # Medium red for moderately negative
            else:
                colors.append('#DC143C')  # Dark red for strongly negative

        plt.scatter(indices, scores, c=colors, s=50, alpha=0.7)
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strongly Positive')
        plt.axhline(y=0.2, color='lightgreen', linestyle='--', alpha=0.5, label='Moderately Positive')
        plt.axhline(y=0.05, color='palegreen', linestyle='--', alpha=0.5, label='Slightly Positive')
        plt.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
        plt.axhline(y=-0.05, color='lightsalmon', linestyle='--', alpha=0.5, label='Slightly Negative')
        plt.axhline(y=-0.2, color='salmon', linestyle='--', alpha=0.5, label='Moderately Negative')
        plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strongly Negative')

        plt.title('Sentiment Score Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Comments (sorted by sentiment)')
        plt.ylabel('Sentiment Score')
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

# ------------------ MAIN ------------------
def main():
    try:
        scraper = YouTubeScraper(YOUTUBE_API_KEY)
        analyzer = SentimentAnalyzer()
        visualizer = Visualizer()

        url = input("Enter YouTube video URL: ").strip()
        video_id = scraper.extract_video_id(url)
        title = scraper.get_video_title(video_id)
        comments = scraper.fetch_comments(video_id, MAX_COMMENTS)

        # Save comments to file
        clean_title = re.sub(r'[\\/*?:"<>|]', '', title)[:50]
        filename = f"Comments_{clean_title.replace(' ', '_')}.xlsx"
        pd.DataFrame(comments, columns=["Comments"]).to_excel(filename, index=False)

        # Analyze sentiment
        results = analyzer.analyze(comments)

        # Display results
        total_score = results['total_score']
        sentiment_counts = results['sentiment_counts']
        sentiment_summary = analyzer.categorize_sentiment(total_score, sentiment_counts)

        print(f"\n   Video Title: {title}\n")
        print(f"Public Opinion: {sentiment_summary}\n")

        # Generate visualizations
        visualizer.sentiment_distribution(sentiment_counts)
        print("")
        visualizer.sentiment_score_chart(results['comment_results'])
        print("")
        visualizer.wordcloud(results['positive_phrases'], "Positive Sentiment Phrases")
        print("")
        visualizer.wordcloud(results['negative_phrases'], "Negative Sentiment Phrases")

        # Delete the Excel file after analysis is complete (REMOVE this if the Excel file is wanted)
        if os.path.exists(filename):
            os.remove(filename)

    except HttpError as e:
        if 'API key not valid' in str(e):
            print("❌ Enter your YouTube Data API v3 key properly")
        else:
            print(f"\n❌ YouTube API error: {str(e)}")
    except ValueError as e:
        print(f"\n❌ {str(e)}")
    except Exception as e:
        print(f"\n❌ {str(e)}")

# Replace with your YouTube Data API v3 key
YOUTUBE_API_KEY = '<your YouTube API key>'

if __name__ == '__main__':
    main()
