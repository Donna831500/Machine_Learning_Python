




from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for state, idx in self.state_dict.items():
            temp_alpha = self.pi[idx]*self.B[idx][O[0]]
            alpha[idx][0]=temp_alpha

        state_dict_p = self.state_dict.copy()
        for t in range(1, L):
            for state, idx in self.state_dict.items():
                total = 0.0
                for state_p, idx_p in state_dict_p.items():
                    total = total+(self.A[idx_p][idx]*alpha[idx_p][t-1])
                temp_alpha = self.B[idx][O[t]]*total
                alpha[idx][t]=temp_alpha

        return alpha


        

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for state, idx in self.state_dict.items():
            beta[idx][L-1]=1.0

        state_dict_p = self.state_dict.copy()
        for t in range(L-2, -1, -1):
            for state, idx in self.state_dict.items():
                total = 0.0
                for state_p, idx_p in state_dict_p.items():
                    total = total+(self.A[idx][idx_p]*self.B[idx_p][O[t+1]]*beta[idx_p][t+1])
                beta[idx][t]=total

        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        L = len(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prob = 0.0
        for state, idx in self.state_dict.items():
            prob = prob+(alpha[idx][L-1]*beta[idx][L-1])
        return prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        gamma = np.multiply(alpha,beta)/seq_prob
        return gamma


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        O = self.find_item(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        state_dict_p = self.state_dict.copy()
        for t in range(0, L-1):
            for state, idx in self.state_dict.items():
                for state_p, idx_p in state_dict_p.items():
                    prob[idx][idx_p][t] = alpha[idx][t]*self.A[idx][idx_p]*self.B[idx_p][O[t+1]]*beta[idx_p][t+1]
        prob= prob/seq_prob
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)

        delta = np.zeros([S, L])
        tri = np.zeros([S, L])

        # base case
        for state, idx in self.state_dict.items():
            temp_delta = self.pi[idx]*self.B[idx][O[0]]
            delta[idx][0]=temp_delta

        state_dict_p = self.state_dict.copy()
        for t in range(1, L):
            for state, idx in self.state_dict.items():
                temp_list = []
                for state_p, idx_p in state_dict_p.items():
                    temp_tuple = (self.A[idx_p][idx]*delta[idx_p][t-1], idx, idx_p)
                    temp_list.append(temp_tuple)
                max_tuple = sorted(temp_list, key=lambda x: x[0], reverse=True)[0]
                max_value = max_tuple[0]
                delta[idx][t]=self.B[idx][O[t]]*max_value
                tri[idx][t] = max_tuple[2]

        ### backtracking
        # base
        temp_path = [-1]*L
        max_delta = 0.0
        for state, idx in state_dict_p.items():
            current_delta = delta[idx][L-1]
            if current_delta>max_delta:
                max_delta = current_delta
                temp_path[L-1] = idx

        for t in range(L-2, -1, -1):
            idx = int(temp_path[t+1])
            temp_path[t] = tri[idx][t+1]

        for i in range(0,L):
            path.append(self.find_key(self.state_dict, temp_path[i]))

        return path



    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
