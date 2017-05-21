import rnn_for_server
import unittest

bob_the_nn = rnn_for_server.RecurrentNeuralNetwork(model_name="model")

bob_the_nn.generate_weights(100)

bob_the_nn.predict("mouse the,")