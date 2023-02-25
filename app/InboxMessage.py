import nltk
from stop_words import get_stop_words
from nltk.corpus import stopwords
import re
from nltk.sentiment import SentimentIntensityAnalyzer

class InboxMessage():
    def __init__(self, content, sender) -> None:
        self.content = content
        self.sender = sender
        self.sentimentAnaliysis = None

    def analize(self):
        sia = SentimentIntensityAnalyzer()
        self.sentimentAnaliysis = sia.polarity_scores(self.content)

    def SetResponse(self, response):
        self.response = response