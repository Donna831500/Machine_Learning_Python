
import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    unique_words_list = list(unique_words.keys())
    value_list = list(range(0,len(unique_words_list)))
    word2idx = dict(zip(unique_words_list, value_list))

    value_list_2 = list(range(0,S))
    tag2idx = dict(zip(tags, value_list_2))
    

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    ## Pi
    pi_dict = {}
    for line in train_data:
        freq = pi_dict.get(line.tags[0], 0)
        freq += 1
        pi_dict[line.tags[0]] = freq
    #print(pi_dict)

    for tag,freq in pi_dict.items():
        idx = tag2idx.get(tag)
        pi[idx]=freq
    pi = pi/np.sum(pi)
    #print(type(pi))

    ## A
    transition_dict = {}
    state_dict = {}
    for line in train_data:
        for i in range(0,len(line.tags)-1):
            freq = transition_dict.get((line.tags[i],line.tags[i+1]),0)
            freq = freq+1
            transition_dict[(line.tags[i],line.tags[i+1])]=freq
            freq_s = state_dict.get(line.tags[i], 0)
            freq_s = freq_s+1
            state_dict[line.tags[i]] = freq_s

    for transition,freq_t in transition_dict.items():
        denom = state_dict.get(transition[0],0)
        if denom!=0:
            start_idx = tag2idx.get(transition[0])
            end_idx = tag2idx.get(transition[1])
            A[start_idx][end_idx] = freq_t/denom


    ## B
    emission_dict = {}
    tag_dict = {}
    for line in train_data:
        current_tags = line.tags
        current_words = line.words
        for i in range(0,len(current_tags)):
            freq_t = tag_dict.get(current_tags[i],0)
            freq_t = freq_t+1
            tag_dict[current_tags[i]] = freq_t
            freq = emission_dict.get((current_tags[i], current_words[i]),0)
            freq = freq+1
            emission_dict[(current_tags[i], current_words[i])] = freq

    for emission,freq_e in emission_dict.items():
        denom = tag_dict.get(emission[0],0)
        if denom!=0:
            start_idx = tag2idx.get(emission[0])
            end_idx = word2idx.get(emission[1])
            B[start_idx][end_idx] = freq_e/denom
    

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    for line in test_data:
        for each_word in line.words:
            obs_idx_dict = model.obs_dict
            if obs_idx_dict.get(each_word,-1)==-1:
                new_idx = len(obs_idx_dict)
                obs_idx_dict[each_word]=new_idx
                model.obs_dict = obs_idx_dict

                current_B = model.B
                b = np.array([0.000001]*len(tags)).reshape((len(tags), 1))
                current_B = np.hstack((current_B,b))
                model.B = current_B

        sentence = np.array(line.words)
        tagging.append(model.viterbi(sentence))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
