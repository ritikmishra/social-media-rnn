import rnn_for_server as rnn
import argparse
import os
import nltk
import time
import subprocess
import json
import collections

parser = argparse.ArgumentParser(description='Run an RNN on the last 20 tweets of a twitter account.')

parser.add_argument("--twitter-account", help="What is the name of the twitter user you want to do an RNN to?", default="ritmish")
parser.add_argument("--train-again", help="Train an entire model again", action="store_true")
parser.add_argument("--training-iters", help="Number of training iterations", default="50000")
args = parser.parse_args()

tweets_json = subprocess.check_output(['node', 'get_tweets.js', args.twitter_account])
tweets = tweets_json.split("\n")
tweets_text = " ".join(tweets)

print(tweets_text)
bob_the_rnn = rnn.RecurrentNeuralNetwork(text=tweets_text, train_new_model=args.train_again, model_name=args.twitter_account)


bob_the_rnn.generate_weights(int(args.training_iters))

while True:
    bob_the_rnn.predict()
    time.sleep(0.5)