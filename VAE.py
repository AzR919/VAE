# import the necessary packages
import numpy as np
from utils import *

class VAE(object):
    #This is VAE model that you should implement
    def __init__(self, hidden_units=128, z_units=20, input_dim=784, batch_size=64):
        """
        initialize all parameters in the model.
        Encoding part:
        1. W_input_hidden, b_input_hidden: convert input to hidden
        2. W_hidden_mu, b_hidden_mu:
        3. W_hidden_logvar, b_hidden_logvar
        Sampling:
        1. random_sample
        Decoding part:
        1. W_z_hidden, b_z_hidden
        2. W_hidden_out, b_hidden_out
        """


    def encode(self, x):
        """
        input: x is input image with size (batch, indim)
        return: hidden_mu, hidden_logvar, both sizes should be (batch, z_units)
        """


    def decode(self, z):
        """
        input: z is the result from sampling with size (batch, z_unit)
        return: out, the generated images from decoder with size (batch, indim)
        """


    def forward(self, x, unittest=False):
        """
        combining encode, sampling and decode.
        input: x is input image with size (batch, indim)
        return: out, the generated images from decoder with size (batch, indim)
        """
        # DO NOT modify or delete this line, it is for testing 
        # just ignore it, and write you implementation below
        if (unittest): np.random.seed(1433125)




    def loss(self, x, out):
        """
        Given the input x (also the ground truth) and out, computing the loss (CrossEntropy + KL).
        input: x is the input of the model with size (batch, indim)
               out is the predicted output of the model with size (batch, indim)
        IMPORTANT: the loss computed should be divided by batch size.
        """


    def backward(self, x, pred):
        """
        Given the input x (also the ground truth) and out, computing the gradient of parameters.
        input: x is the input of the model with size (batch, indim)
               pred is the predicted output of the model with size (batch, indim)
        return: grad_list = [dW_input_hidden, db_input_hidden, dW_hidden_mu, db_hidden_mu, dW_hidden_logvar, db_hidden_logvar,
                            dW_z_hidden, db_z_hidden, dW_hidden_out, db_hidden_out]
        IMPORTANT: make sure the gradients follows the exact same order in grad_list.
        """


    def set_params(self, parameter_list):
        """
        IMPORTANT: used by autograder and unit-test
        TO set parameters with parameter_list
        input: parameter_list = [W_input_hidden, b_input_hidden, W_hidden_mu, b_hidden_mu, W_hidden_logvar, b_hidden_logvar,
                            W_z_hidden, b_z_hidden, W_hidden_out, b_hidden_out]
        """
 


if __name__ == "__main__":
    # x_train is of shape (5000 * 784)
    # We've done necessary preprocessing for you so just feed it into your model.
    x_train = np.load('data.npy')

