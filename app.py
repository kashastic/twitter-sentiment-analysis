import json
from dotenv import load_dotenv

from flask import Flask, request, render_template
from textblob import TextBlob
import tweepy
import pandas as pd
import numpy as np
import os
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

load_dotenv()
@app.route('/')
def hello_world():
    return render_template("index_tsa.html")


@app.route('/predict', methods=['POST'])
def predict():
    def create_wordcloud(text):
        mask = np.array(Image.open("cloud_like.png"))
        stopwords = set(STOPWORDS)
        wc = WordCloud(background_color="white",
                       mask=mask,
                       max_words=700,
                       stopwords=stopwords,
                       repeat=True)
        wc.generate(str(text))
        wc.to_file("static/wc.png")

    input_string = [str(x) for x in request.form.values()]
    str_keyword = input_string[0]
    numTweetsInp = int(input_string[1])

    consumerKey = os.getenv("consumerKey")
    consumerSecret = os.getenv("consumerSecret")
    accessToken = os.getenv("accessToken")
    accessTokenSecret = os.getenv("accessTokenSecret")
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    def percentage(part, whole):
        return 100 * float(part) / float(whole)

    tweets = tweepy.Cursor(api.search, q=str_keyword,lang="en").items(numTweetsInp)
    tweet_list = []

    for tweet in tweets:
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            tweet_list.append(tweet.text)

    tweet_list = pd.DataFrame(tweet_list)
    tweet_list.drop_duplicates(inplace=True)

    # Cleaning Text (RT, Punctuation etc)

    # Creating new dataframe and new features
    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Removing RT, Punctuation etc
    remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
    tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
    tw_list["text"] = tw_list.text.str.lower()
    # print(tw_list.head(10))

    #noOfTweets = len(tw_list)

    # Calculating Negative, Positive, Neutral and Compound values

    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tw_list['text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = "positive"
        else:
            tw_list.loc[index, 'sentiment'] = "neutral"
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp

    # Creating new data frames for all sentiments (positive, negative and neutral)

    tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
    tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]

    # Function for count_values_in single columns

    def count_values_in_column(data, feature):
        total = data.loc[:, feature].value_counts(dropna=False)
        percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
        return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])

    # Count_values for sentiment
    count_values_in_column(tw_list, "sentiment")

    create_wordcloud(tw_list["text"].values)

    # print(noOfTweets)
    positive = percentage(len(tw_list_positive), noOfTweets)
    negative = percentage(len(tw_list_negative), noOfTweets)
    neutral = percentage(len(tw_list_neutral), noOfTweets)

    result = str(positive) + ' ' + str(negative) + ' ' + str(neutral)
    response = json.dumps(result)

    return response


if __name__ == '__main__':
    app.run(debug=True)
