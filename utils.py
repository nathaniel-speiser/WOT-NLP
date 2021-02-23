
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords


import string


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def plot_topic_words(model, feature_names, feature_num, topics):

    topic_name = topics[feature_num]
    topic = model.components_[feature_num]
    top_indices = topic.argsort()[:-11:-1]
    features = [feature_names[i] for i in top_indices]
    feature_weights =  [topic[i] for i in top_indices]

    plt.figure(figsize = (12,8))
    sns.barplot(x=features, y = feature_weights, color = 'royalblue')
    plt.title(f'Top words in {topic_name} topic', fontsize =20)
    plt.xticks(fontsize = 14)
    plt.show()

def plot_chapter_topics_book(topics_by_chapter, book_num):
    i=book_num-1
    title = titles[i]
    start = starts[i]
    end = ends[i]
    book_df = topics_by_chapter.iloc[start:end]
    chapters = list(book_df['chapter_title'])

    plt.figure(figsize=(12,8))
    sns.heatmap(data = book_df[topics].T, cmap='Blues', cbar=False, robust=True, xticklabels=chapters)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize = 15, rotation = 90 )
    plt.title(title + ' topics by chapter', fontsize =24)
    plt.show()
topics = ['Rand', 'Perrin', 'Mat', 'Nynaeve', 'Egwene', 'Moiraine', 'Aiel', 'Gawyn', 'Elayne', 'Seanchan', 'Siuan',\
         'Cadsuane politics', 'Black Tower', 'White Tower Politics', 'Horn of Valere', 'Faile Kidnapping', 'Min',\
         'Last Battle', 'Emond\'s Field','Thom']
starts = [0,54, 105, 162, 220, 277, 334, 376, 408, 444, 476, 515, 567, 626]
ends = [54, 105, 162, 220, 277, 334, 376, 408, 444, 476, 515, 567, 626, 676]
tick_locations = np.array([ 27,  79, 133, 191, 248, 305, 355, 392, 426, 460, 495, 541, 596, 651])

def plot_topic_time(topics_by_chapter,topic):
    y = topics_by_chapter[topic]
    x = topics_by_chapter['cumulative_chapter_number']
    plt.figure(figsize=(12,6))
    sns.scatterplot(x=x, y=y)
    plt.xticks(ticks = tick_locations, labels = titles, fontsize = 12, rotation = 90 )
    plt.ylim(0,max(y)+.01)
    plt.vlines(ends, 0 ,max(y)+.01, color = 'black', linestyles='dotted', linewidths=2)
    plt.ylabel(f'{topic} topic presence', fontsize = 14)
    plt.xlabel('', fontsize = 1)
    plt.title(f'{topic} topic presence by chapter', fontsize = 20)
    plt.show()

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
