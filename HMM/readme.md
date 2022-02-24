
Hidden Markov Models
================================

General instructions
--------------------

-   In this task you will implement various inference algorithms for
    **HMM** and also apply them to sentence tagging.

Q1 Implement the inference algorithms
-------------------------------------

In `hmm.py`, you will find a class called HMM whose attributes specify
the model parameters of a Hidden Markov Model (including its initial
state probability, transition probability, and emission probability).
You need to implement the following six functions

-   `forward`: compute the forward messages
-   `backward`: compute the backward messages
-   `sequence_prob`: compute the probability of observing a particular
    sequence
-   `posterior_prob`: compute the probability of the state at a
    particular time step given the observation sequence
-   `likelihood_prob`: compute the probability of state transition at a
    particular time step given the observation sequence
-   `viterbi`: compute the most likely hidden state path using the
    Viterbi algorithm.

We have discussed how to compute all these via dynamic programming in
the lecture. Here, the only thing you need to pay extra attention to is
that the indexing system is slightly different between the python code
and the formulas we discussed (the former starts from 0 and the latter
starts from 1). Read the comments in the code carefully to get a better
sense of this discrepancy.

Q2 Application to speech tagging
--------------------------------

Part-of-Speech (POS) is a category of words (or, more generally, of
lexical items) which have similar grammatical properties. (Example:
noun, verb, adjective, adverb, pronoun, preposition, conjunction,
interjection, and sometimes numeral, article, or determiner.)
Part-of-Speech Tagging (POST) is the process of marking up a word in a
text (corpus) as corresponding to a particular part of speech, based on
both its definition and its context.

Here you will use HMM to perform POST, where the tags are states and the
words are observations. We collect our dataset and tags with the Dataset
class. Dataset class includes tags, train\_data and test\_data. Both
datasets include a list of sentences, and each sentence is an object of
the Line class. You only need to implement the `model_training` function
and the `speech_tagging` function in `tagger.py`.

-   `model_training`: in this function, you need to build an instance of
    the HMM class by setting its five parameters
    `(pi, A, B, obs_dict, state_dict)`. The way you estimate the
    parameter `pi, A, B` is simply by counting the corresponding
    frequency from the given training set, as we discussed in the class.
    Read the comments in the code for more instructions.
-   `speech_tagging`: given the HMM built from model\_training, now your
    task is to run the Viterbi algorithm to find the most likely tagging
    of a given sentence. One particular detail you need to take care of
    is when you meet a new word which was unseen in the training
    dataset. In this case, you need to update the dictionary `obs_dict`
    accordingly, and also expand the emission matrix by assuming that
    the probability of seeing this new word under any state is 1e-6.
    Again, read the comments in the code for more instructions.

Q3 Testing
----------

Once you finish these two parts, run `hmm_test_script.py`. You will see the function output,
accuracy, and time consuming.
