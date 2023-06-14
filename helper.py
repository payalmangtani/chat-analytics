from typing import Any, Union
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from PIL import ImageDraw
import spacy

spacy.load('en_core_web_sm')

extract = URLExtract()


def fetch_stats(selected_user, selected_time, df):
    if selected_time != 'Overall':
        df = df[df['year'] == selected_time]
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(selected_user, selected_time, df):
    if selected_time != 'Overall':
        df = df[df['year'] == selected_time]
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def create_wordcloud(selected_user, selected_time, df):
    f = open('stop_words.txt', 'r')
    stop_words = f.read()

    if selected_time != 'Overall':
        df = df[df['year'] == selected_time]
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['year'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc





def common_words(selected_user, selected_time, df):
    f = open('stop_words.txt', 'r')
    stop_words = f.read()

    if selected_time != 'Overall':
        df = df[df['year'] == selected_time]
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['year'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    common_df = pd.DataFrame(Counter(words).most_common(20))
    return common_df


def emoji_helper(selected_user, selected_time, df):
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        emojis = []
        for message in df1['message']:
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    if selected_user != 'Overall':
        df2=df
        df2 = df2[df2['user'] == selected_user]
        emojis = []
        for message in df2['message']:
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user, selected_time, df):
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        timeline = df1.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    if selected_user != 'Overall':
        df2=df
        df2 = df2[df2['user'] == selected_user]
        timeline = df2.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, selected_time, df):
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        daily_timeline = df1.groupby('only_date').count()['message'].reset_index()
    if selected_user != 'Overall':
        df2=df
        df2= df2[df2['user'] == selected_user]
        daily_timeline = df2.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, selected_time, df):
    n=df['day_name'].value_counts()
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        n = df1['day_name'].value_counts()
    if selected_user != 'Overall':
        df2=df
        df2 = df2[df2['user'] == selected_user]
        n = df2['day_name'].value_counts()

    return n


def month_activity_map(selected_user, selected_time, df):
    n=df['month'].value_counts()
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        n=df1['month'].value_counts()
    if selected_user != 'Overall':
        df2=df
        df2 = df2[df2['user'] == selected_user]
        n=df2['month'].value_counts()

    return n


def activity_heatmap(selected_user, selected_time, df):
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    if selected_time != 'Overall':
        df1=df
        df1 = df1[df1['year'] == selected_time]
        user_heatmap = df1.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    if selected_user != 'Overall':
        df2=df
        df2 = df2[df2['user'] == selected_user]
        user_heatmap = df2.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def keyword_helper(keyword, df):
    messages = []

    for message in df['message']:
        if (keyword in message):
            messages.append(message)

    return messages


def keyphrase(df):
    messages = []

    for message in df['message']:
        if ('⭐' in message):
            messages.append(message)

    return messages


# Return set of most common words having k(0/1/-1) sentiment
def most_common_words(selected_user, selected_time, df, k):
    f = open('stop_words.txt', 'r')
    stop_words = f.read()

    if selected_time != 'Overall':
        df = df[df['year'] == selected_time]
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['year'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(5))
    return most_common_df


def clean_text(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)
    text_cleaned = "".join([x for x in text if x not in string.punctuation])
    text_cleaned = re.sub(' +', ' ', text_cleaned)
    text_cleaned = text_cleaned.lower()
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'nice', 'like', 'Like', 'brush', \
                       'stand', 'great', 'well', 'shave', 'good', 'product', 'put', \
                       'buy', 'ago', 're', 'edu', 'use', 'from', 'my', 'we', \
                       'i', 've', 'buy', 'set', 'lot', 'decide', 'give', 'add'])

    # ........

    # Taking only those words which are not stopwords
    tokens = [token for token in tokens if token not in stop_words]
    ps = PorterStemmer()
    # .......... THERE'RE SOME OMITTED CODE BEFORE THIS .........
    text_cleaned = " ".join([ps.stem(token) for token in tokens])
    return text_cleaned


def LDA(index,topics):
    imp_topic_words = ""
    topic_words = dict(topics[index][1])
    for key in topic_words:
        imp_topic_words = imp_topic_words + key + ", "
    wc = WordCloud(min_font_size=10, background_color='white')
    df_wc = wc.generate(imp_topic_words)
    return df_wc
#topics = lda_model.show_topics(formatted=False)

def TopN(df):
    mask = df['sentiment_score'].gt(0)
    df1 = pd.concat([df[mask].sort_values('sentiment_score'),
                    df[~mask].sort_values('sentiment_score', ascending=False)], ignore_index=True)

    return df1