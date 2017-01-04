from math import exp, log
from nltk.corpus import treebank
from scipy.optimize import fmin_bfgs
from features import make_feature_function
import numpy
import random

NUM_FEATURES = 100
DEBUG = True


def debug_print(s, nl=True):
    if DEBUG:
        print s,
    if nl:
        print


sents = treebank.tagged_sents()
with open("features.txt") as features_file:
    f = make_feature_function(features_file.read())

TAGS = set([])
for sent in sents:
    for word,tag in sent:
        TAGS.add(tag)


def Z(words, prev_tag, index, w):
    return sum([score(words, prev_tag, other_tag, index, w) for other_tag in TAGS])


def score(words, prev_tag, tag, index, w):
    return exp(numpy.dot(f(words, prev_tag, tag, index), w))


# def f(words, prev_tag, tag, index):

def gradient(w):
    """
    compute the gradient of the log likelihood as a function of w
    """
    grad = [0] * NUM_FEATURES

    debug_print("Preprocessing for gradient...")
    # preprocess Z,f values
    z_values = {}
    f_values = {}
    for i, sent in enumerate(sents):
        if i % 10 == 0:
            debug_print(i)
        words = [word for word, tag in sent]
        for j in xrange(len(sent)):
            z_all = 0
            for tag in TAGS:
                z_values[(i,j,tag)] = score(words, None, tag, j, w)
                z_all += z_values[(i,j,tag)]
                f_values[(i,j,tag)] = f(words, None, tag, j)
            z_values[(i,j,"ALL_TAGS_SUM")] = z_all
    debug_print("Done pre-processing, compute partial derivatives")

    for k in xrange(NUM_FEATURES):
        debug_print(k)
        grad_k = 0
        for i,sent in enumerate(sents):
            for j in xrange(len(sent)):
                true_tag = sent[j][1]
                f_k = f_values[(i,j,true_tag)][k]
                z_grad = 0
                for tag in TAGS:
                    z_grad += z_values[(i,j,tag)] * f_values[(i,j,tag)][k]
                z_grad /= z_values[(i,j,"ALL_TAGS_SUM")]
                grad_k += f_k - z_grad
        grad[k] = grad_k
    debug_print("Done computing gradient")
    return grad


def minus_log_likelihood(w):
    debug_print("Compute likelihood...")
    ll = 0
    for sent in sents:
        words = [word for word, tag in sent]
        for j in xrange(len(sent)):
            true_tag = sent[j][1]
            ll += numpy.dot(f(words, None, true_tag, j), w)
            ll -= log(Z(words, None, j, w))
    debug_print("Done")
    return -1 * ll


def minus_gradient(w):
    return -1 * numpy.array(gradient(w))


def optimize(x0=None):
    if x0 is None:
        x0 = [random.random() - 0.5 for i in xrange(NUM_FEATURES)]
        print "Initial guess is", x0
    best_w = fmin_bfgs(minus_log_likelihood, x0, fprime=minus_gradient)
    with open("results.txt", "w") as res_file:
        res_file.write(str(best_w))
    return best_w
