import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora, models, similarities, matutils
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from textblob import TextBlob

import string
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def make_stopwords():
    sw = stopwords.words('english')
    sw = [w.translate(str.maketrans('', '', string.punctuation)) for w in sw]
    sw = list(set(sw))
    sw.extend(['said','did','like', 'woman', 'man'])

titles = ['The Eye of the World',
 'The Great Hunt',
 'The Dragon Reborn',
 'The Shadow Rising',
 'The Fires of Heaven',
 'Lord of Chaos',
 'A Crown of Swords',
 'The Path of Daggers',
 'Winter\'s Heart',
 'Crossroads of Twilight',
 'Knife of Dreams',
 'The Gathering Storm',
 'Towers of Midnight',
 'A Memory of Light']
