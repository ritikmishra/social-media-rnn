import argparse

default_training_data = "long ago , the mice had a general council to consider what measures they could take to outwit their " \
                        "common enemy , the cat . some said this , and some said that but at last a young mouse got up and " \
                        "said he had a proposal to make , which he thought would meet the case . you will all agree , " \
                        "said he , that our chief danger consists in the sly and treacherous manner in which the enemy " \
                        "approaches us . now , if we could receive some signal of her approach , we could easily escape from " \
                        "her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon " \
                        "round the neck of the cat . by this means we should always know when she was about , " \
                        "and could easily retire while she was in the neighbourhood . this proposal met with general applause " \
                        ", until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice " \
                        "looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible " \
                        "remedies . "

parser = argparse.ArgumentParser(description='Run an RNN on some text.')

parser.add_argument("--text", type=str, metavar="text", nargs="+", help="Some text to train the RNN on",
                    default=default_training_data)
parser.add_argument("--train-new-model", action="store_true",
                    help="Include this flag if you want to train a brand new model")
parser.add_argument("--model-name", help="What is the name of the model that you want to load or save?", default="default")
args = parser.parse_args()

if type(args.text) == list:
    args.text = " ".join(args.text)

