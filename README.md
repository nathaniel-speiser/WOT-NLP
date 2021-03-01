# Using NLP to analyze the Wheel of Time

## Description

This project investigated the *Wheel of Time* book series by Robert Jordan and Brandon Sanderson using natural language processing techniques. A variety of techniques were used in order to gain various insights into the series.

First, I wanted to investigate writing differences between the two authors of the series. I figured that writing differences will mostly show up in verbs, adverbs, and adjectives, so I used nltk's part of speech tagger to find the words that one author used and the other didnt. I found that Robert Jordan used many more adverbs, which fits with his reputation for flowery language, while Brandon Sanderson used more plain but action oriented language, also fitting with his reputation. Below are word clouds with the words each author used most frequently that the other did not.
![Writing differences](https://github.com/nathaniel-speiser/WOT-NLP/blob/main/pics/word_differences.png)

However, the bulk of this project was spent on topic modeling the main 14 *Wheel of Time* books. After some experimentation I found that a Tfidf vectorizer and NMF topic modeling gave the best results. I included both unigrams and bigrams. The inclusion of the latter did not drastically increase performance or interpretability, but I chose to include them because of the large number of important bigrams, such as Aes Sedai or Lews Therin. This topic modeling approach worked quite well, and with 20 topics produced clearly distinguishable topics. These topics largely fell into 3 groups: main characters, recurring groups, and distinct plotlines. The topics that came out were largely expected, including the 5 main characters, several groups of Aes Sedai, the Aiel, and the Last Battle. Interestingly, the plotline topics seem to occur when a large number of main characters are together, such as in the first two and final book. Further clustering with KMeans clustering resulted in clusters corresponding to the 5 main characters from Emond's field Below is a plot of all the topics over each chapter of the *Wheel of Time* series.
![Topics over time](https://github.com/nathaniel-speiser/WOT-NLP/blob/main/saved_models/topicsovertime.png)


In addition to topic modeling, I also created networks of character interactions in and up to each chapter. I defined a connection as the characters' names appearing withing a certain number of words of each other. The networks come out largely as expected, with main characters having a large number of connections to each other as well as a large number of side characters. However, it is still interesting to see the subgroups (speaking nontechnically) that emerge, such as the Forsaken, in the graph visualizations.
![Character network](https://github.com/nathaniel-speiser/WOT-NLP/blob/main/pics/network.png)

In order to visualize the topic modeling and character networks, I have created a streamlit app. When I recieve an invitation to host the app I will put it up there and update this readme, but until then you can use streamlit to run streamlit_app.py. To run the app you will need to have scikit-learn, nltk, networkx, seaborn and wordcloud installed.

Finally, I also trained a gensim word2vec model on the entirety of the *Wheel of Time* series text. Early results showed good similarity performance, such as finding that the most similar words to Bela, the name of a horse, are other horses/pets throughout the series. Analogy performance is more of a mixed bag, with some relationships showing up but many others not appearing. In the future I'd like to make a lightweight app to display these results.



## Data

The data used for this project was the text of the main 14 books of the Wheel of time series. If you would like to replicate this project you will need the text of the books as text files, which you can get from epubs of the books.

## Tools

* pandas
* numpy
* scikit-learn
* nltk
* gensim
* networkx
* matplotlib
* seaborn
* wordcloud
* streamlit


## Future directions

If I revisit this project I'd like to make better visualizations/interactive apps to demonstrate my results. I also think theres more work to be done with the word2vec model, including seeing how the usage of words (especially character names) evolves over time.
