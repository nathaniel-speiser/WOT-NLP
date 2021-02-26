
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler

import string
from wordcloud import WordCloud

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def plot_topic_words(model, feature_names, topic_num, topics):

    topic_name = topics[topic_num]
    topic = model.components_[topic_num]
    top_indices = topic.argsort()[:-11:-1]
    features = [feature_names[i] for i in top_indices]
    feature_weights =  [topic[i] for i in top_indices]

    plt.figure(figsize = (12,8))
    sns.barplot(x=features, y = feature_weights, color = 'royalblue')
    plt.title(f'Top words in {topic_name} topic', fontsize =20)
    plt.xticks(fontsize = 14)
    plt.show()

def topic_wordcloud(model, feature_names, topic_num, topics, ret = False):
    topic_name = topics[topic_num]
    topic = model.components_[topic_num]
    top_indices = topic.argsort()[:-51:-1]
    features = [feature_names[i] for i in top_indices]
    feature_weights =  [topic[i] for i in top_indices]

    scaler = MinMaxScaler(feature_range=(1,50))
    scaled_weights = scaler.fit_transform(np.array(feature_weights).reshape(1,-1)).flatten()
    frequencies = {}
    for i in range(len(features)):
        frequencies[features[i]]= int(scaled_weights[i])

    wordcloud = WordCloud(width = 1200, height = 600,
                background_color ='white',
                min_font_size = 12).generate_from_frequencies(frequencies)

    f = plt.figure(figsize = (12, 6), facecolor = None)
    plt.imshow(wordcloud)
    plt.title(f'Words in {topics[topic_num]} topic', fontsize = 20)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    if ret:
        return f
    else: plt.show()

def plot_chapter_topics_book(topics_by_chapter, book_num, ret = False):
    i=book_num-1
    title = titles[i]
    start = starts[i]
    end = ends[i]
    book_df = topics_by_chapter.iloc[start:end]
    chapters = list(book_df['chapter_title'])

    f=plt.figure(figsize=(8,3),dpi=200)
    sns.heatmap(data = book_df[topics].T, cmap='Blues', cbar=False, robust=True, yticklabels=topics, xticklabels = np.arange(1,len(chapters)+1))
    plt.yticks(fontsize = 5)
    plt.xticks(fontsize = 5, rotation = 0)
    plt.ylabel('Topic', fontsize = 7)
    plt.xlabel('Chapter number (including pro/epilogue)', fontsize = 7)
    plt.tick_params(axis='both', length = 0, width= 0)
    plt.title('Topic presence in '+ title + ' by chapter', fontsize =8)

    if ret: return f
    else: plt.show()





topics = ['Rand', 'Perrin', 'Mat', 'Nynaeve', 'Egwene', 'Moiraine', 'Aiel', 'Gawyn', 'Elayne', 'Seanchan', 'Siuan',\
         'Cadsuane politics', 'Black Tower', 'White Tower Politics', 'Horn of Valere', 'Faile Kidnapping', 'Min',\
         'Last Battle', 'Emond\'s Field','Thom']
starts = [0,54, 105, 162, 220, 277, 334, 376, 408, 444, 476, 515, 567, 626]
ends = [54, 105, 162, 220, 277, 334, 376, 408, 444, 476, 515, 567, 626, 676]
tick_locations = np.array([ 27,  79, 133, 191, 248, 305, 355, 392, 426, 460, 495, 541, 596, 651])
num_chapters = [54, 51,57,58,57,57,42,32,36,32,39,52,59,51]
ratings = [4.17, 4.22, 4.25, 4.23,4.15,4.13, 4.03, 3.91, 3.94, 3.81, 4.14, 4.36, 4.41, 4.49]
#for plotting
rating_arr = []
for i,num in enumerate(num_chapters):
    for _ in range(num):
        rating_arr.append(ratings[i])
rating_arr = np.array(rating_arr)

def plot_topic_time(topics_by_chapter,topic, ret=False):
    """
    Plot the prevalence of a topic over time (all chapters)
    """
    y = topics_by_chapter[topic]
    x = topics_by_chapter['cumulative_chapter_number']
    f= plt.figure(figsize=(12,6))
    sns.set_context(rc = {'patch.linewidth': 0.0})
    sns.barplot(x=x, y=y, color='royalblue')
    plt.xticks(ticks = tick_locations, labels = titles, fontsize = 12, rotation = 90 )
    plt.ylim(0,max(y)+.01)
    plt.vlines(ends, 0 ,max(y)+.01, color = 'black', linestyles='dotted', linewidths=2)
    plt.ylabel(f'{topic} topic presence', fontsize = 14)
    plt.xlabel('', fontsize = 1)
    plt.title(f'{topic} topic presence by chapter', fontsize = 20)
    if ret: return f
    else: plt.show()

def plot_topic_and_rating(topics_by_chapter,topic, ret=False):
    """
    Plot the prevalence of a topic over time (all chapters)
    """
    y = topics_by_chapter[topic]
    x = topics_by_chapter['cumulative_chapter_number']
    f= plt.figure(figsize=(12,6))
    sns.set_context(rc = {'patch.linewidth': 0.0})
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.barplot(x=x, y=y, color='royalblue', label = 'Topic presence')
    plt.xticks(ticks = tick_locations, labels = titles, fontsize = 12, rotation = 90 )
    plt.ylim(0,max(y)+.01)
    plt.vlines(ends, 0 ,max(y)+.01, color = 'black', linestyles='dotted', linewidths=2)
    plt.ylabel(f'{topic} topic presence', fontsize = 14)
    plt.xlabel('', fontsize = 1)
    plt.title(f'{topic} topic presence by chapter', fontsize = 20)
    ax2 = plt.twinx()
    sns.lineplot(x=x, y=rating_arr, color="orange", label='Book rating', ax=ax2)
    plt.legend()
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
