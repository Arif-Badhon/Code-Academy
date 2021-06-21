#Topic modeling is an area of NLP dedicated to uncovering latent, or hidden, topics within a body of language. For example, one Codecademy curriculum developer 
#used topic modeling to discover patterns within Taylor Swift songs related to love and heartbreak over time.
#A common technique is to deprioritize the most common words and prioritize less frequently used terms as topics in a process known as 
#term frequency-inverse document frequency (tf-idf). Say what?! This may sound counter-intuitive at first. Why would you want to give more priority to less-used words? 
#Well, when you’re working with a lot of text, it makes a bit of sense if you don’t want your topics filled with words like “the” and “is.” The Python 
#libraries gensim and sklearn have modules to handle tf-idf.
#Whether you use your plain bag of words (which will give you term frequency) or run it through tf-idf, the next step in your topic modeling journey 
#is often latent Dirichlet allocation (LDA). LDA is a statistical model that takes your documents and determines which words keep popping up together 
#in the same contexts (i.e., documents). We’ll use sklearn to tackle this for us.
#If you have any interest in visualizing your newly minted topics, word2vec is a great technique to have up your sleeve. word2vec can map out your topic 
#model results spatially as vectors so that similarly used words are closer together. In the case of a language sample consisting of “The squids jumped out of the suitcases. 
#The squids were furious. Why are your suitcases full of jumping squids?”, we might see that “suitcase”, “jump”, and “squid” were words used within similar contexts. 
#This word-to-vector mapping is known as a word embedding.

import nltk, re
from sherlock_holmes import bohemia_ch1, bohemia_ch2, bohemia_ch3, boscombe_ch1, boscombe_ch2, boscombe_ch3
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# preparing the text
corpus = [bohemia_ch1, bohemia_ch2, bohemia_ch3, boscombe_ch1, boscombe_ch2, boscombe_ch3]
preprocessed_corpus = [preprocess_text(chapter) for chapter in corpus]

# Update stop_list:
stop_list = "say", "see", "holmes", "shall", "say", "man", "upon", "know", "quite", "one", 
"well", "could", "would", "take", "may", 
"think", "come", "go", "little", "must", 
"look"
# filtering topics for stop words
def filter_out_stop_words(corpus):
  no_stops_corpus = []
  for chapter in corpus:
    no_stops_chapter = " ".join([word for word in chapter.split(" ") if word not in stop_list])
    no_stops_corpus.append(no_stops_chapter)
  return no_stops_corpus
filtered_for_stops = filter_out_stop_words(preprocessed_corpus)

# creating the bag of words model
bag_of_words_creator = CountVectorizer()
bag_of_words = bag_of_words_creator.fit_transform(filtered_for_stops)

# creating the tf-idf model
tfidf_creator = TfidfVectorizer(min_df = 0.2)
tfidf = tfidf_creator.fit_transform(preprocessed_corpus)

# creating the bag of words LDA model
lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_bag_of_words = lda_bag_of_words_creator.fit_transform(bag_of_words)

# creating the tf-idf LDA model
lda_tfidf_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_tfidf = lda_tfidf_creator.fit_transform(tfidf)

print("~~~ Topics found by bag of words LDA ~~~")
for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([bag_of_words_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)

print("\n\n~~~ Topics found by tf-idf LDA ~~~")
for topic_id, topic in enumerate(lda_tfidf_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([tfidf_creator.get_feature_names()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)
