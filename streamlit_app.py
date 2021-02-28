import streamlit as st
from network_plotting import plot_network
import pandas as pd
import pickle
from utils import topics, topic_wordcloud, plot_topic_time, plot_chapter_topics_book, titles
st.set_page_config(layout='wide')

##############################################################
# Importing data/models
with open('saved_models/nmf_model.pkl', 'rb') as f:
    nmf_model = pickle.load(f)

with open('saved_models/vec_feature_names.pkl', 'rb') as f:
    vec_feature_names = pickle.load(f)
graphs_df = pd.read_pickle('saved_models/graphs_df.pkl')

important_chars = ['rand', 'perrin', 'mat', 'egwene', 'elayne', 'nynaeve', 'moiraine', 'min', 'faile', 'aviendha',
                  'gawyn', 'lan', 'siuan', 'morgase', 'cadsuane', 'ituralde', 'galad', 'pevara', 'tuon', 'elaida',
                  'androl', 'taim', 'logain', 'gareth', 'rhuarc', 'graendal', 'moridin', 'moghedien', 'verin',
                  'birgitte', 'loial', 'tam', 'demandred', 'sammael', 'thom']
tbc = pd.read_pickle('saved_models/topics_by_chapter.pkl')

##################################################################




st.title('Wheel of Time NLP')
st.write('## Topics over time ##')
book = st.selectbox('Select a book', ['All books']+titles)

if book == 'All books':
    st.image('saved_models/topicsovertime2.png', use_container_width=True)
else:
    booknum = titles.index(book)+1
    f= plot_chapter_topics_book(tbc, booknum, ret=True )
    st.pyplot(f)





#Side by side of topic word cloud and occurrence of topic over time
topic = st.selectbox('Select a topic', topics)
topic_index = topics.index(topic)
col1, col2 = st.beta_columns(2)
with col1:
    wordcloud = topic_wordcloud(nmf_model, vec_feature_names, topic_index, topics, ret = True)
    st.pyplot(wordcloud)
with col2:
    topic_over_time = plot_topic_time(tbc,topic,ret=True)
    st.pyplot(topic_over_time)

#Character connection graph
st.write('## Character connection graph ##')
chap = st.slider('Select a chapter number', min_value = 1, max_value = 676, step = 1)

#Parameters for graphing
show_all = st.checkbox('Show all connections regardless of size (recommended for early chapters)')
set_width = None
if show_all: set_width = 2

chap_name = graphs_df['book_title'][chap] + ' chapter ' +str(graphs_df['chapter_nr'][chap]) + ': ' + graphs_df['chapter_title'][chap]
st.write('### Character connections as of {} ###'.format(chap_name))


graph = graphs_df['cumulative_networkx'][chap]
graph_fig = plot_network(graph, chars=important_chars,set_width = set_width, show_all=show_all, output='return')
st.plotly_chart(graph_fig, use_container_width=True)
