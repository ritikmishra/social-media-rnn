import tensorflow
import rnn_for_server as rnn
import tweepy
import argparse
import os
import nltk
import time

parser = argparse.ArgumentParser(description='Run an RNN on the last 20 tweets of a twitter account.')

parser.add_argument("--twitter-account", help="What is the name of the twitter user you want to do an RNN to?", default="ritmish")
parser.add_argument("--train-again", help="Train an entire model again", action="store_true")
args = parser.parse_args()

auth = tweepy.OAuthHandler(os.environ["TWITTER_CONSUMER_KEY"], os.environ["TWITTER_CONSUMER_SECRET"])
auth.set_access_token(os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_TOKEN_SECRET"])

api = tweepy.API(auth)

tweets = []
good_words = []
for tweet in api.user_timeline(args.twitter_account):
    tweets.extend(nltk.word_tokenize(tweet.text))
for word in tweets:
    if word != "https" and word != ":" and word[0:2] != "//":
        good_words.append(word)

compiled_tweets = " ".join(good_words)

bob_the_rnn = rnn.RecurrentNeuralNetwork(text=compiled_tweets, train_new_model=args.train_again, model_name=args.twitter_account)


bob_the_rnn.generate_weights(20000)

while True:
    bob_the_rnn.predict()
    time.sleep(0.5)