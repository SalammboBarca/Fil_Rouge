import configparser
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorTfidf:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('../config.cfg')
        self.train_df = pd.read_csv(config['FILES']['TRAIN'])
        self.test_df = pd.read_csv(config['FILES']['TEST'])

    @staticmethod
    def clean_text(text):
        text = text.lower()
        for change in [(r"what's", "what is "), (r"\'s", " "), (r"\'ve", " have "), (r"can't", "cannot "),
                       (r"n't", " not "), (r"i'm", "i am "), (r"\'re", " are "), (r"\'d", " would "), (r"\'ll", " will "), (r"\'scuse", " excuse ")]:
            text = re.sub(change[0], change[1], text)
        text = text.strip(' ')
        return text

    def vectorize(self):
        self.train_df['comment_text'] = self.train_df['comment_text'].map(lambda com: self.clean_text(com))
        self.test_df['comment_text'] = self.test_df['comment_text'].map(lambda com: self.clean_text(com))

        x_train = self.train_df.comment_text
        x_test = self.test_df.comment_text

        # vect = TfidfVectorizer(max_df=60000, min_df=2, stop_words='english')
        vect = TfidfVectorizer(max_features=5000, stop_words='english')
        # learn the vocabulary in the training data, then use it to create a document-term matrix
        x_train_tfid = vect.fit_transform(x_train)
        # transform the test data using the earlier fitted vocabulary, into a document-term matrix
        x_test_tfid = vect.transform(x_test)
        # return xa, xt, ya, yt
        return x_train_tfid, x_test_tfid
