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
        #base-case
        alpha[:, 0] = self.pi*self.B[:, O[0]]

        #the rest -- dynamic programming
        # O provides the index of observation symbol
        for t in range(1, L):
            for s in range(S):
                alpha[s][t] = self.B[s][O[t]] * np.dot(self.A[:, s], alpha[:, t-1])
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
        #base-case
        beta[:, L-1] = np.ones(S)

        #the rest... dynamic programming
        for t in range(L-2, -1, -1):
            for s in range(S):
                beta[s][t] = np.dot((beta[:, t+1] * self.B[:, O[t+1]]), self.A[s, :] )

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
        forward_message = self.forward(Osequence)
        seq_probs = forward_message[:, -1] #we just want at time T for all states
        return np.sum(seq_probs)

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
        normalize = self.sequence_prob(Osequence)
        gamma = alpha * beta / normalize
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
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        normalize = self.sequence_prob(Osequence)
        O = self.find_item(Osequence)
        for t in range(L-1):
            for i in range(S):
                for j in range(S):
                    prob[i][j][t] = alpha[i][t] * self.A[i][j] * beta[j][t+1] *self.B[j] [O[t+1]] /normalize
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
        O = self.find_item(Osequence)  # gets the index
        delta = np.zeros((S,L)) #calculate most likely paths
        choice = np.zeros((S,L), dtype=int)
        backwards = np.zeros(L, dtype=int)
        tracker = 0
        #base-case:
        delta[:,0] = self.pi * self.B[:, O[0]]
        #main algo
        for t in range(1,L):
            for s in range(S):
                margin = self.A[:,s] * delta[:, t-1]
                delta[s][t] = self.B[s][O[t]] * np.max(margin)
                choice[s][t] = np.argmax(margin)
        #back-track
        end = delta[:,L-1]
        s_t = np.argmax(end)
        backwards[tracker]= s_t
        tracker += 1
        for t in range(L-1, 0, -1):
            s_t = choice[s_t][t]
            backwards[tracker] = s_t
            tracker += 1
        backwards = np.flip(backwards)
        for idx in backwards:
            path.append(self.find_key(self.state_dict, idx))

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
