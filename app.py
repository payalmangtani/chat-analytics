import streamlit as st
from streamlit_option_menu import option_menu
import preprocessor, helper
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os

import seaborn as sns
import nltk.classify.util
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import ImageDraw
import spacy
spacy.load('en_core_web_sm')
from wordcloud import WordCloud, STOPWORDS

def clean(words):
    return dict([(word, True) for word in words])

negative_ids = movie_reviews.fileids('neg')
positive_ids = movie_reviews.fileids('pos')

negative_features = [(clean(movie_reviews.words(fileids=[f])), 'negative') for f in negative_ids]
positive_features = [(clean(movie_reviews.words(fileids=[f])), 'positive') for f in positive_ids]

negative_cutoff = int(len(negative_features) * 95/100)
positive_cutoff = int(len(positive_features) * 90/100)

train_features = negative_features[:negative_cutoff] + positive_features[:positive_cutoff]
test_features = negative_features[negative_cutoff:] + positive_features[positive_cutoff:]

classifier = NaiveBayesClassifier.train(train_features)


#Object
sentiments = SentimentIntensityAnalyzer()

neutral, negative, positive = 0, 0, 0
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.sidebar.title("Whatsapp Chat Analyzer")
nltk.download('vader_lexicon')
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        d = bytes_data.decode("utf-8")
        data = preprocessor.preprocess(d)

        data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data['message']]  # Positive
        data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data['message']]  # Negative
        data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data['message']]  # Neutral


        # To indentify true sentiment per row in message column
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                return 0


        data['value'] = data.apply(lambda row: sentiment(row), axis=1)

        polarity = [round(sentiments.polarity_scores(i)['compound'], 2) for i in data['message']]
        data['sentiment_score'] = polarity

        # Creating new column & Applying function


        data['value'] = data.apply(lambda row: sentiment(row), axis=1)


        for i in data['sentiment_score']:
            if i > 0:
                positive+=1
            elif i < 0:
                negative+=1
            elif i == 0:
                neutral+=1

        total = neutral + negative + positive
        label =[]
        data['cleaned_reviews'] = data['message'].apply(lambda x: helper.clean_text(x))
        for i in data['message']:
            if "Nivea" in i:
                x = 'Nivea'
            elif "KayBeauty" in i:
                x = 'Kay Beauty'
            elif "baby" in i:
                x = 'Himalaya baby kit'
            elif "Olay" in i:
                x = 'Olay'
            elif "LOreal" in i:
                x = 'LOreal Paris'
            elif "shampoo" in i:
                x = 'Herbal Essences shampoo'
            elif "Nykaa" in i:
                x = 'Nykaa'
            elif "Lakme" in i:
                x = 'Lakme'
            else:
                x = 'Himalaya Naturals'
            label.append(x)

        data['label']=label


                # fetch unique users
        user_list = data['user'].unique().tolist()
        time_list = data['year'].unique().tolist()
        time_list.sort()
        time_list.insert(0, "Overall")
        #user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        st.sidebar.write("Select year for analysis")
        selected_time = st.sidebar.selectbox("Show analysis wrt Time Period", time_list)
        st.sidebar.write("Select user for analysis")
        selected_user = st.sidebar.selectbox("Show analysis wrt User", user_list)
        selected = option_menu(
            menu_title="Analysis Results",
            options=["Time Series Analysis", "Participation Analysis", "Emoji Analysis", "Sentiment Analysis", "Insights"],
            icons=["clock", "profile", "emoji", "smile", "emoji"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        if selected == "Time Series Analysis":

            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, selected_time, data)

            col1, col2 = st.columns(2)
            with col1:
                # monthly timeline
                st.subheader("Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, selected_time, data)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # daily timeline
                #st.subheader("Daily Timeline")
                #daily_timeline = helper.daily_timeline(selected_user, data)
                #fig, ax = plt.subplots()
                #ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                #plt.xticks(rotation='vertical')
                #st.pyplot(fig)
                st.subheader("Weekly Activity Map")
                user_heatmap = helper.activity_heatmap(selected_user, selected_time, data)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)

        if selected == "Participation Analysis":
            # activity map
            st.subheader('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Most busy day")
                busy_day = helper.week_activity_map(selected_user, selected_time, data)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.caption("Most busy month")
                busy_month = helper.month_activity_map(selected_user, selected_time, data)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)


            # finding the busiest users in the group(Group level)
            if selected_time != 'Overall':
                st.subheader('Most Busy Users')
                x, new_df = helper.most_busy_users(selected_user, selected_time,data)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

        if selected == "Emoji Analysis":
            # emoji analysis
            emoji_df = helper.emoji_helper(selected_user,selected_time, data)
            st.subheader("Emoji Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                if emoji_df.empty:
                    st.write("No emojis used")
                else:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                    st.pyplot(fig)




        if selected == "Sentiment Analysis":



            # WORDCLOUD......

            labels = 'Neutral', 'Negative', 'Positive'
            sizes = [neutral, negative, positive]
            colors = ['#00bcd7', '#F57C00', '#CDDC39']

            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

            #st.markdown('### Metrics')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Messages: "
                            "</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: green;'> {positive} </h3>", unsafe_allow_html=True)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Nagative Messages: </h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: red;'> {negative} </h3>", unsafe_allow_html=True)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Messages: </h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: grey;'> {neutral} </h3>", unsafe_allow_html=True)


            data1=helper.TopN(data)
            data2=data1.head(5)
            toplist=[]
            for i in data2['message']:
                toplist.append(i)
            st.subheader('Top 5 Positive messages:')
            length1 = len(toplist)
            for i in range(length1):
                st.write(i+1,'.',toplist[i])
            data3=data1.tail()
            taillist=[]
            for i in data3['message']:
                taillist.append(i)
            length2 = len(taillist)
            st.subheader('Top 5 Negative Messages:')
            for i in range(length2):
                st.write(i+1, '.', taillist[i])



        if selected == "Insights":

            # use_container_width=True
            st.header("Topic Modelling")
            # fig.update_layout(width=1100,height=900)
            def tokenize(sentences):
                for sentence in sentences:
                    # deacc true removes punctuations
                    yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


            # Apply function on dataset
            cleaned_review_words = list(tokenize(data['cleaned_reviews']))
            data1=helper.TopN(data)
            data2 = data1.head(20)
            cleaned_review_words2 = list(tokenize(data2['cleaned_reviews']))

            bigram2 = gensim.models.Phrases(cleaned_review_words2)
            trigram2 = gensim.models.Phrases(bigram2[cleaned_review_words2])

            # Implement bigrams and trigrams with phraser function
            bigram_mod2 = gensim.models.phrases.Phraser(bigram2)
            trigram_mod2 = gensim.models.phrases.Phraser(trigram2)


            def make_bigrams2(texts):
                return [bigram_mod2[doc] for doc in texts]


            def make_trigrams2(texts):
                return [trigram_mod2[bigram_mod2[doc]] for doc in texts]


            # Run bigram function on dataset
            data_words_bigrams2 = make_bigrams2(cleaned_review_words2)
            # Run trigram function on dataset
            data_words_trigrams2 = make_trigrams2((data_words_bigrams2))


            def lemmatization(texts, allowed_postags=['ADJ']):
                texts_out = []
                for sent in texts:
                    doc = nlp(" ".join(sent))
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                return texts_out


            # ........ SOME CODE IS OMMITTED FROM HERE ........

            # # load spacy with english as the language
            nlp = spacy.load('en_core_web_sm')
            # # lemmatization
            data_lemmatized2 = lemmatization(data_words_trigrams2)

            # 1- create dictionary
            dictionary2 = corpora.Dictionary(data_lemmatized2)

            # 2- Create corpus
            # it's important we understand that this converts the list of words to matching integers for our LDA algorithm
            texts2 = data_lemmatized2
            corpus2 = [dictionary2.doc2bow(text) for text in texts2]

            lda_model2 = gensim.models.ldamodel.LdaModel(corpus=corpus2,
                                                        id2word=dictionary2,
                                                        num_topics=2,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=7,
                                                        passes=7,
                                                        alpha='symmetric',
                                                        iterations=100,
                                                        per_word_topics=True)

            # Print the Keyword in the 5 topics
            stop_words = stopwords.words('english')
            stop_words.extend(['from', 'nice', 'like', 'Like', 'brush','comfortable', 'tough',\
                               'stand', 'great', 'well', 'shave', 'good', 'product', 'put', \
                               'buy', 'ago', 're', 'edu', 'use', 'from', 'my', 'we', \
                               'i', 've', 'buy', 'set', 'lot', 'decide', 'give', 'add'])


            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
            cloud = WordCloud(stopwords= stop_words, background_color='white', width=2500, height=1800, max_words=10, colormap='table0', color_func=lambda *args, **kwargs:cols[i], prefer_horizontal=1.0)
            topics2 = lda_model2.show_topics(formatted=False)

            data1 = helper.TopN(data)
            data3 = data1.tail(10)
            cleaned_review_words3 = list(tokenize(data3['cleaned_reviews']))

            bigram3 = gensim.models.Phrases(cleaned_review_words3)
            trigram3 = gensim.models.Phrases(bigram3[cleaned_review_words3])

            # Implement bigrams and trigrams with phraser function
            bigram_mod3 = gensim.models.phrases.Phraser(bigram3)
            trigram_mod3 = gensim.models.phrases.Phraser(trigram3)


            def make_bigrams3(texts):
                return [bigram_mod3[doc] for doc in texts]


            def make_trigrams3(texts):
                return [trigram_mod2[bigram_mod3[doc]] for doc in texts]


            # Run bigram function on dataset
            data_words_bigrams3 = make_bigrams3(cleaned_review_words3)
            # Run trigram function on dataset
            data_words_trigrams3 = make_trigrams3((data_words_bigrams3))


            # ........ SOME CODE IS OMMITTED FROM HERE ........

            # # load spacy with english as the language
            nlp = spacy.load('en_core_web_sm')
            # # lemmatization
            data_lemmatized3 = lemmatization(data_words_trigrams3)

            # 1- create dictionary
            dictionary3 = corpora.Dictionary(data_lemmatized3)

            # 2- Create corpus
            # it's important we understand that this converts the list of words to matching integers for our LDA algorithm
            texts3 = data_lemmatized3
            corpus3 = [dictionary3.doc2bow(text) for text in texts3]

            lda_model3 = gensim.models.ldamodel.LdaModel(corpus=corpus3,
                                                         id2word=dictionary3,
                                                         num_topics=2,
                                                         random_state=100,
                                                         update_every=1,
                                                         chunksize=7,
                                                         passes=7,
                                                         alpha='symmetric',
                                                         iterations=100,
                                                         per_word_topics=True)

            # Print the Keyword in the 5 topics


            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
            cloud = WordCloud(stopwords=stop_words, background_color='white', width=2500, height=1800, max_words=10,
                              colormap='table0', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)
            topics3 = lda_model3.show_topics(formatted=False)


            # Generate a word cloud image for given topic



            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Positive Sentiments")
                df_wc = helper.LDA(1, topics2)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)


            with col2:
                st.subheader("Negative Sentiments: ")
                df_wc = helper.LDA(0, topics3)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)

            st.subheader("Predicted brands to boost sale:")
            toplist = []
            data4=data1.head(5)
            for i in data4['label']:
                toplist.append(i)
            topset = list(set(toplist))
            l1=len(topset)
            for i in range(l1):
                st.write(topset[i])

            st.subheader("Brands having poor reviews:")
            taillist = []
            data5 = data1.tail(5)
            for i in data5['label']:
                taillist.append(i)
            tailset = list(set(taillist))
            l2 = len(tailset)
            for i in range(l2):
                st.write(tailset[i])

