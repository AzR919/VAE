import unittest
import numpy as np
import VAE
from numpy.testing import assert_allclose
import pickle as pk

with open("test.pkl","rb") as f: tests = pk.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.TestEncode
# to run all tests: python -m unittest tests

class TestEncode(unittest.TestCase):
    def test(self):
        # Set model parameters and batch sample
        for i in range(len(tests[0])):
            sample = tests[0][i]
            params = tests[1][i]
            model = VAE.VAE()
            model.set_params(params)
            res = tests[2][i]
            mu, var = model.encode(sample)
            assert_allclose(mu, res[0], atol=TOLERANCE)
            assert_allclose(var, res[1], atol=TOLERANCE)

class TestForward(unittest.TestCase):
    # set unittest mode for assigning a random seed
    def test(self):
        for i in range(len(tests[0])):
            sample = tests[0][i]
            params = tests[1][i]
            model = VAE.VAE()
            model.set_params(params)
            res = tests[3][i][1]
            ans = model.forward(sample, unittest=True)
            assert_allclose(ans, res, atol=TOLERANCE)


class TestLoss(unittest.TestCase):
    def test(self):
        for i in range(len(tests[0])):
            sample = tests[0][i]
            params = tests[1][i]
            model = VAE.VAE()
            model.set_params(params)
            res = tests[4][i]
            pred = model.forward(sample, unittest=True)
            ans1 = model.loss(sample, pred)
            assert_allclose(ans1, res, atol=TOLERANCE)



class TestBackward(unittest.TestCase):
    def test(self):
        for i in range(len(tests[0])):
            sample = tests[0][i]
            params = tests[1][i]
            grads = tests[5][i]
            model = VAE.VAE()
            model.set_params(params)
            pred = model.forward(sample, unittest=True)
            ans = model.backward(sample, pred)
            for j in range(10):
                assert_allclose(ans[j], grads[3][j], atol=TOLERANCE)
