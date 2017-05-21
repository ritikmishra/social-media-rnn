import random
import tensorflow as tf
import nltk
import numpy as np
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

# Values from the argparser
# args.model_name (str)
# args.text (str)
# args.train_new_model (bool)


vocab_size = None
training_data = args.text

training_data = np.array(nltk.word_tokenize(training_data))
training_data = np.reshape(training_data, [-1, ])

n_input = 3  # how many words will be read at a time
n_layers = 2 # layers in our RNN
n_hidden = 512  # how many hidden units there will be


def pred_to_word(pred_data, num_to_char):
    """
    :param pred_data:  Data as outputed by the predictor function 
    :return: 
    """
    pred_data = pred_data[0]
    largest_value = {"index": 0, "val": pred_data[0]}
    for index, val in enumerate(pred_data):
        if val > largest_value["val"]:
            largest_value["val"] = val
            largest_value["index"] = index

    return num_to_char[largest_value["index"]]


def build_dataset_from_words(tokens):
    """
    :param tokens: A string of words to turn into a suitable dataset 
    :return: (dictionary, reverse_dictionary) where the keys of dictionary are the words and the values are the word numbers. reverse_dictionary is the opposite.
    """
    global vocab_size
    # filter out duplicates
    unique_tokens = []

    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    dictionary = dict(zip(list(range(len(unique_tokens))), unique_tokens))
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    vocab_size = len(unique_tokens)
    return dictionary, reverse_dictionary


reverse_dictionary, dictionary = build_dataset_from_words(training_data)


def RNN(x, weights, biases):
    """
    Generate some words based on an rnn
    :param x: The input
    :param weights: Weights
    :param biases: Biases
    :return: Words
    """
    # Reshape the input vector to be appropriate
    tf.reshape(x, [-1, n_input])

    x = tf.split(x, n_input, 1)

    rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

outer_x = tf.placeholder("float", [None, n_input])  # placeholder for a 1d array that is n_input wide
outer_y = tf.placeholder("float", [None, vocab_size])  # placeholder fr a 1d array that is vocab size wide

predictor = RNN(outer_x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=outer_y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(predictor, 1), tf.argmax(outer_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print(dictionary)
print(reverse_dictionary)

print(dictionary[training_data[3]])

training_iters = 50000
acc_total = 0
loss_total = 0
saver = tf.train.Saver()
should_save = True
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    end_offset = n_input + 1
    try:
        if args.train_new_model:
            for step in range(training_iters):
                offset = random.randint(0, vocab_size - end_offset)

                symbols_in_keys = [[dictionary[str(training_data[i])] for i in range(offset, offset + n_input)]]
                human_readable_sik = [reverse_dictionary[word] for word in symbols_in_keys[0]]
                symbols_out_onehot = np.zeros([1, vocab_size], dtype=float)  # don't worry if pycharm complains
                symbols_out_onehot[0, dictionary[str(training_data[offset + n_input])]] = 1.0
                _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, predictor],
                                                     feed_dict={outer_x: symbols_in_keys, outer_y: symbols_out_onehot})

                loss_total += loss
                acc_total += acc

                if step % 100 == 0:
                    print("Step", step)
                    print("Average Loss:", loss_total / 100)
                    print("Average Accuracy:", acc_total / 100)
                    print("Input: ", human_readable_sik, "Output: ", pred_to_word(onehot_pred, reverse_dictionary)
                          , "vs Label:", str(training_data[offset + n_input]))
                    loss_total = 0
                    acc_total = 0
        else:
            saver.restore(sess, "./saved_model/" + args.model_name + ".ckpt")
            print("Model restored.")

    except (EOFError, KeyboardInterrupt):
        print("Skipping further iterations. . .")
        _should_save = input("Do you want to save? [Y/n]").strip()
        if _should_save[0].lower() == "y":
            print("Understood. Saving.")
        elif _should_save[0].lower() == "n":
            print("Not saving.")
            should_save = False
        else:
            print("Going with default action of saving.")

    finally:
        if(should_save):
            save_path = saver.save(sess, "./saved_model/" + args.model_name + ".ckpt")
            print("Model saved in file: %s" % save_path)
        else:
            print("Didn't save")

    while True:
        seed = str(input("Type 3 words:"))
        seed = nltk.word_tokenize(seed.strip())
        # Correct amount of words?
        if len(seed) != 3:
            continue
        # Good words?
        for word in seed:
            if word not in training_data:
                print("The RNN does not know the words you have entered")
                continue

        symbols_in_keys = [[dictionary[i] for i in seed]]
        output = seed
        for j in range(32):
            pred = sess.run(predictor, feed_dict={outer_x: symbols_in_keys})
            output.append(pred_to_word(pred, reverse_dictionary))
            symbols_in_keys = [[dictionary[output[-1]], dictionary[output[-2]], dictionary[output[-3]]]]
        print(" ".join(output))
