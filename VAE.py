# import the necessary packages
import numpy as np
from utils import *
import progressbar

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

        self.hidden_units = hidden_units
        self.z_units = z_units
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.W_input_hidden = np.random.normal(0, 1, (input_dim,hidden_units) ) * 0.001
        self.b_input_hidden = np.zeros((hidden_units,))
        self.W_hidden_mu = np.random.normal(0, 1, (hidden_units, z_units) ) * 0.001
        self.b_hidden_mu = np.zeros((z_units,))
        self.W_hidden_logvar = np.random.normal(0, 1, (hidden_units, z_units) ) * 0.001
        self.b_hidden_logvar = np.zeros((z_units,))
        self.W_z_hidden = np.random.normal(0, 1, (z_units, hidden_units) ) * 0.001
        self.b_z_hidden = np.zeros((hidden_units,))
        self.W_hidden_out = np.random.normal(0, 1, (hidden_units, input_dim) ) * 0.001
        self.b_hidden_out = np.zeros((input_dim,))

        self.A_hidden = LRelu("VAE")
        
        self.A_z_hidden = Relu()
        self.A_hidden_out = Sigmoid()

    def encode(self, x):
        """
        input: x is input image with size (batch, indim)
        return: hidden_mu, hidden_logvar, both sizes should be (batch, z_units)
        """
        self.hidden_e = x @ self.W_input_hidden + self.b_input_hidden

        self.hidden_e_act = self.A_hidden.forward(self.hidden_e)

        hidden_mu = self.hidden_e_act @ self.W_hidden_mu + self.b_hidden_mu

        hidden_logvar = self.hidden_e_act @ self.W_hidden_logvar + self.b_hidden_logvar

        return hidden_mu, hidden_logvar


    def decode(self, z):
        """
        input: z is the result from sampling with size (batch, z_unit)
        return: out, the generated images from decoder with size (batch, indim)
        """
        self.hidden_d = z @ self.W_z_hidden + self.b_z_hidden

        self.hidden_a = self.A_z_hidden.forward(self.hidden_d)

        out_pre_a = self.hidden_a @ self.W_hidden_out + self.b_hidden_out

        out = self.A_hidden_out.forward(out_pre_a)

        return out


    def forward(self, x, unittest=False):
        """
        combining encode, sampling and decode.
        input: x is input image with size (batch, indim)
        return: out, the generated images from decoder with size (batch, indim)
        """
        # DO NOT modify or delete this line, it is for testing 
        # just ignore it, and write you implementation below
        if (unittest): np.random.seed(1433125)

        self.hidden_mu, self.hidden_logvar = self.encode(x)

        self.eps = np.random.normal(0, 1, self.hidden_mu.shape)

        self.z = self.hidden_mu + np.exp(0.5 * (self.hidden_logvar)) * self.eps

        self.out = self.decode(self.z)

        return self.out




    def loss(self, x, out):
        """
        Given the input x (also the ground truth) and out, computing the loss (CrossEntropy + KL).
        input: x is the input of the model with size (batch, indim)
               out is the predicted output of the model with size (batch, indim)
        IMPORTANT: the loss computed should be divided by batch size.
        """

        CE = np.sum(BCE_loss(out, x))

        KL = np.sum(-0.5 * (1 + self.hidden_logvar - np.square(self.hidden_mu) - np.exp(self.hidden_logvar)))

        return (CE + KL)/x.shape[0]


    def backward(self, x, pred):
        """
        Given the input x (also the ground truth) and out, computing the gradient of parameters.
        input: x is the input of the model with size (batch, indim)
               pred is the predicted output of the model with size (batch, indim)
        return: grad_list = [dW_input_hidden, db_input_hidden, dW_hidden_mu, db_hidden_mu, dW_hidden_logvar, db_hidden_logvar,
                            dW_z_hidden, db_z_hidden, dW_hidden_out, db_hidden_out]
        IMPORTANT: make sure the gradients follows the exact same order in grad_list.
        """
        dout = -1 * x / pred + (1-x)/(1-pred)

        da = dout * (self.out * (1-self.out))
        
        db_hidden_out = np.sum(da, axis=0)
        dW_hidden_out = self.hidden_a.T @ da

        db = self.A_z_hidden.backward(da @ self.W_hidden_out.T)

        db_z_hidden = np.sum(db, axis=0)
        dW_z_hidden = self.z.T @ db

        dz = db @ self.W_z_hidden.T

        dmu = dz + self.hidden_mu
        dlogvar = dz * (0.5 * np.exp(0.5 * self.hidden_logvar) * self.eps) + 0.5 * (np.exp(self.hidden_logvar) - 1)

        db_hidden_logvar = np.sum(dlogvar, axis=0)
        dW_hidden_logvar = (dlogvar.T @ self.hidden_e_act).T

        db_hidden_mu = np.sum(dmu, axis=0)
        dW_hidden_mu = (dmu.T @ self.hidden_e_act).T

        dmu_b = self.A_hidden.backward(dmu @ self.W_hidden_mu.T)
        dlogvar_b = self.A_hidden.backward(dlogvar @ self.W_hidden_logvar.T)

        dx = dmu_b + dlogvar_b

        db_input_hidden = np.sum(dx, axis=0)
        dW_input_hidden = (dx.T @ x).T

        return [dW_input_hidden, db_input_hidden, dW_hidden_mu, db_hidden_mu, dW_hidden_logvar, db_hidden_logvar,
                            dW_z_hidden, db_z_hidden, dW_hidden_out, db_hidden_out]


    def set_params(self, parameter_list):
        """
        IMPORTANT: used by autograder and unit-test
        TO set parameters with parameter_list
        input: parameter_list = [W_input_hidden, b_input_hidden, W_hidden_mu, b_hidden_mu, W_hidden_logvar, b_hidden_logvar,
                            W_z_hidden, b_z_hidden, W_hidden_out, b_hidden_out]
        """

        self.W_input_hidden = parameter_list[0]
        self.b_input_hidden = parameter_list[1]
        self.W_hidden_mu = parameter_list[2]
        self.b_hidden_mu = parameter_list[3]
        self.W_hidden_logvar = parameter_list[4]
        self.b_hidden_logvar = parameter_list[5]
        self.W_z_hidden = parameter_list[6]
        self.b_z_hidden = parameter_list[7]
        self.W_hidden_out = parameter_list[8]
        self.b_hidden_out = parameter_list[9]

    def step(self, grads, lr):

        self.W_input_hidden -= lr * grads[0]
        self.b_input_hidden -= lr * grads[1]
        self.W_hidden_mu -= lr * grads[2]
        self.b_hidden_mu -= lr * grads[3]
        self.W_hidden_logvar -= lr * grads[4]
        self.b_hidden_logvar -= lr * grads[5]
        self.W_z_hidden -= lr * grads[6]
        self.b_z_hidden -= lr * grads[7]
        self.W_hidden_out -= lr * grads[8]
        self.b_hidden_out -= lr * grads[9]

 


if __name__ == "__main__":
    # x_train is of shape (5000 * 784)
    # We've done necessary preprocessing for you so just feed it into your model.
    x_train = np.load('data.npy')

    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64

    vae = VAE(batch_size=batch_size)

    train_losses = []

    for epoch in progressbar.progressbar(range(num_epochs)):

        np.random.shuffle(x_train)

        for x in range(0, x_train.shape[0], batch_size):

            x_batch = x_train[x:x+batch_size]
            
            out = vae.forward(x_batch)
            grads = vae.backward(x_batch, out)
            vae.step(grads, learning_rate)

        out = vae.forward(x_train)
        train_loss = vae.loss(x_train, out)
        train_losses.append(train_loss)

        img_save(out.reshape(5000, 28, 28), "./gen_i", epoch)

    plt.plot(train_losses, label="Train_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss plot")
    plt.show()