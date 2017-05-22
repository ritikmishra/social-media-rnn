import random
import tensorflow as tf
import nltk
import numpy as np
from os import listdir
import collections

default_training_data = "long ago , the mice had a general council to consider what measures they could take to " \
                        "outwit their common enemy , the cat . some said this , and some said that but at last a " \
                        "young mouse got up and said he had a proposal to make , which he thought would meet the case " \
                        ". you will all agree , said he , that our chief danger consists in the sly and treacherous " \
                        "manner in which the enemy approaches us . now , if we could receive some signal of her " \
                        "approach , we could easily escape from her . i venture , therefore , to propose that a small " \
                        "bell be procured , and attached by a ribbon round the neck of the cat . by this means we " \
                        "should always know when she was about , and could easily retire while she was in the " \
                        "neighbourhood . this proposal met with general applause , until an old mouse got up and said " \
                        "that is all very well , but who is to bell the cat ? the mice looked at one another and " \
                        "nobody spoke . then the old mouse said it is easy to propose impossible remedies . "


class RecurrentNeuralNetwork:
    def __init__(self, text=default_training_data, train_new_model=False, model_name="model"):
        """
        :param text: The text you want to train your model on, if you're training a new model. Defaults to Aesop's fable 
                        about belling a cat  
        :param train_new_model: A boolean indicating whether or not you want to train a new model. Defaults to false.
        :param model_name: The name of your model. This will be used when loading and saving your model. If we cannot 
                            find this model, we will train a new model anyways.
        """
        self.train_new_model = train_new_model
        self.model_name = model_name

        self.vocab_size = None

        self.training_data = text

        self.n_input = 3  # how many words will be read at a time
        self.n_layers = 5  # layers in our RNN
        self.n_hidden = 1024  # how many hidden units there will be

        model_names = self.get_existing_models()
        if model_name not in model_names:
            self.train_new_model = True
            print("WARNING! Training new model anyway because " + model_name + " is not saved yet!")

        self.acc_total = 0
        self.loss_total = 0
        self.saver = None
        self.should_save = True

        self.reverse_dictionary = {}
        self.dictionary = {}

        self.weights = {}
        self.biases = {}

        self.predictor = None
        self.sess = None
        self.outer_x = None
        self.outer_y = None

        self.generate_weights_calls = 0


        self.build_dataset_from_words()

    @staticmethod
    def get_existing_models():
        temporary_unprocessed_filename_spot = []
        model_names = []
        for file in listdir("./saved_model"):
            if file.split(".") != file:
                file = file.split(".")
            temporary_unprocessed_filename_spot.extend(file)
        for term in temporary_unprocessed_filename_spot:
            if term not in model_names and term != "index" and term != "checkpoint-" and term != "ckpt" and term != "meta" and term[0:4] != "data":
                model_names.append(term)
        return model_names

    @staticmethod
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

    def build_dataset_from_words(self):
        """
        Generate the dataset out of the words we're training on
        """
        # filter out duplicates

        self.training_data = nltk.word_tokenize(self.training_data)

        unique_tokens = collections.Counter(self.training_data).most_common()

        for word, _ in unique_tokens:
            self.dictionary[word] = len(self.dictionary)

        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.vocab_size = len(unique_tokens)
        print("Dictionary")
        print(self.dictionary) # word to number
        print("Reverse Dictionary")
        print(self.reverse_dictionary) # number to word


    def __rnn(self, x, weights, biases):
        """
        Generate some words based on an rnn
        :param x: The input
        :param weights: Weights
        :param biases: Biases
        :return: Words
        """
        # Reshape the input vector to be appropriate
        tf.reshape(x, [-1, self.n_input])

        x = tf.split(x, self.n_input, 1)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self.n_hidden) for _ in range(self.n_layers)])

        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def generate_weights(self, training_iters):
        """
        Train the RNN. Automatically saves based on parameters in the constructor.
        In the event that you 
        """

        try:

            self.outer_x = tf.placeholder("float", [None, self.n_input])  # placeholder for a 1d array that is n_input wide
            self.outer_y = tf.placeholder("float", [None, self.vocab_size])  # placeholder fr a 1d array that is vocab size wide

            # RNN output node weights and biases
            self.weights = {
                'out': tf.Variable(tf.random_normal([int(self.n_hidden), int(self.vocab_size)]))
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([self.vocab_size]))
            }

            if(self.generate_weights_calls == 0):
                self.predictor = self.__rnn(self.outer_x, self.weights, self.biases)
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictor, labels=self.outer_y))
                optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

            self.saver = tf.train.Saver()

            correct_pred = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(self.outer_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            loss_total = 0
            acc_total = 0
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            end_offset = self.n_input + 1

            if self.train_new_model:
                for step in range(training_iters):
                    offset = random.randint(0, self.vocab_size - end_offset)

                    training_symbols_in_keys = [
                        [self.dictionary[str(self.training_data[i])] for i in range(offset, offset + self.n_input)]]
                    human_readable_sik = [self.reverse_dictionary[word] for word in training_symbols_in_keys[0]]
                    symbols_out_onehot = np.zeros([1, self.vocab_size],
                                                  dtype=float)  # don't worry if pycharm complains
                    symbols_out_onehot[0, self.dictionary[str(self.training_data[offset + self.n_input])]] = 1.0
                    _, acc, loss, onehot_pred = self.sess.run([optimizer, accuracy, cost, self.predictor],
                                                         feed_dict={self.outer_x: training_symbols_in_keys,
                                                                    self.outer_y: symbols_out_onehot})

                    loss_total += loss
                    acc_total += acc

                    if step % 100 == 0:
                        print("Step", step)
                        print("Average Loss:", loss_total / 100)
                        print("Average Accuracy:", acc_total / 100)
                        print("Input: ", human_readable_sik, "Output: ",
                              self.pred_to_word(onehot_pred, self.reverse_dictionary), "vs Label:",
                              str(self.training_data[offset + self.n_input]))
                        loss_total = 0
                        acc_total = 0
            else:
                self.saver.restore(self.sess, "./saved_model/" + self.model_name + ".ckpt")
                print("Model restored.")
                self.should_save = False  # since we have loaded what is already saved

            if self.should_save:
                save_path = self.saver.save(self.sess, "./saved_model/" + self.model_name + ".ckpt")
                print("Model saved in file: %s" % save_path)
        except KeyboardInterrupt, EOFError:
            print("Iter", training_iters)
            print("Saving. . .")
            save_path = self.saver.save(self.sess, "./saved_model/" + self.model_name + ".ckpt")

    def predict(self, seed=None):
        if seed is None:
            seed = [random.choice(list(self.dictionary.keys())) for _ in range(3)]
        else:
            seed = nltk.word_tokenize(seed.strip())

        # Correct amount of words?
        if len(seed) != self.n_input:
            raise ValueError("There needs to be exactly " + str(self.n_input) + " words in the parameter")

        try:

            prediction_symbols_in_keys = [[self.dictionary[i] for i in seed]]
            output = seed
            for j in range(32):
                pred = self.sess.run(self.predictor, feed_dict={self.outer_x: prediction_symbols_in_keys})
                output.append(self.pred_to_word(pred, self.reverse_dictionary))
                prediction_symbols_in_keys = [[self.dictionary[output[-3]], self.dictionary[output[-2]], self.dictionary[output[-1]]]]
                print(prediction_symbols_in_keys)
            print(" ".join(output))
        except KeyError:
            raise KeyError("You have entered words that were not in the source text.")

