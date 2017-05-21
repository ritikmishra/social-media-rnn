"""Run the server for the slack stuff."""
import tornado.ioloop
import tornado.web
import tornado.httpclient
import tornado.testing
from tornado import gen
import tweepy
import os
import subprocess
import nltk
import rnn_for_server as rnn
import sys

try:
    PORT = os.environ['PORT']
except KeyError:
    PORT = 8888

auth = tweepy.OAuthHandler(os.environ["TWITTER_CONSUMER_KEY"], os.environ["TWITTER_CONSUMER_SECRET"])
auth.set_access_token(os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_TOKEN_SECRET"])

api = tweepy.API(auth)


def params_from_request(request):
    """Change the format of the HTTP request parameters so that they may be more easily used."""
    params_dict = {}
    for key, value in request.arguments.items():
        if len(value) == 1:
            params_dict[key] = value[0].decode('UTF-8')
        else:
            params_dict[key] = []
            for part in value:
                params_dict[key].append(part.decode('UTF-8'))
    return params_dict


class HomePageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class TrainOnTwitterHandler(tornado.web.RequestHandler):
    """Handle requests to find out the bitcoin exchange rate."""

    def prepare(self):
        """Prepare for handling the request."""
        self.params = params_from_request(self.request)
        self.existing_models = rnn.RecurrentNeuralNetwork().get_existing_models()
        self.model_name = self.params['model_name']
        if self.model_name not in self.existing_models:
            self.train = True
        else:
            self.train = False

        tweets = []
        for tweet in api.user_timeline(self.model_name):
            tweets.extend(nltk.word_tokenize(tweet.text))
        compiled_tweets = " ".join(tweets)

        self.rn_network = rnn.RecurrentNeuralNetwork(compiled_tweets, self.train, self.model_name)


    def get(self):
        self.write("done")
        self.rn_network.generate_weights(100)
        self.rn_network.predict()



def make_app():
    """Assign handlers."""
    return tornado.web.Application([
        (r"/", TrainOnTwitterHandler),
        (r"/home", HomePageHandler)
    ])


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", TrainOnTwitterHandler),
            (r"/home", HomePageHandler)
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "template"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
        )
        super(Application, self).__init__(handlers, **settings)


if __name__ == "__main__":
    app = Application()
    app.listen(PORT)
    print("Listening on port " + str(PORT))
    tornado.ioloop.IOLoop.current().start()
