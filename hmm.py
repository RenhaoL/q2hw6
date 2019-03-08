from hmmlearn import hmm
import numpy as np
import math

states = ['Sunny', 'Rain']
observations = ['Happy','Sad']
start_probability = {'Sunny': 2/3, 'Rain': 1/3}
transition_probability = {'Sunny':{'Sunny':0.8, 'Rain': 0.2}, 'Rain': {'Sunny': 0.4, 'Rain': 0.6}}

emission_probability = {'Sunny':{'Happy': 0.8, 'Sad':0.2}, 'Rain': {'Happy': 0.4, 'Sad':0.6}}


model = hmm.MultinomialHMM(n_components = len(states))
model.startprob_ = np.array([2/3, 1/3])
model.transmat_ = np.array([[0.8, 0.2],[0.4, 0.6]])
model.emissionprob_ = np.array([[0.8, 0.2],[0.4, 0.6]])

# 0 = happy, 1 = sad
given_obs = np.array([[0,1]]).transpose() # input observation
logprob, seq = model.decode(given_obs, algorithm='viterbi')
print('Given observations are as following:')
print(' -> '.join(observations[i] for item in given_obs for i in item ))
print('The highest probability of given observations is: ',math.exp(logprob)) # print out the highest probability
print('The HMM chain of hidden states is as following:')
print(' -> '.join(states[i] for i in seq)) # print out the hidden states.
print('\n')

given_obs = np.array([[0,1,0]]).transpose() # input observation
logprob, seq = model.decode(given_obs, algorithm='viterbi')
print('Given observations are as following:')
print(' -> '.join(observations[i] for item in given_obs for i in item ))
print('The highest probability of given observations is: ',math.exp(logprob)) # print out the highest probability
print('The HMM chain of hidden states is as following:')
print(' -> '.join(states[i] for i in seq)) # print out the hidden states.
print('\n')

given_obs = np.array([[0,0,1,1,1,0]]).transpose() # input observation
logprob, seq = model.decode(given_obs, algorithm='viterbi')
print('Given observations are as following:')
print(' -> '.join(observations[i] for item in given_obs for i in item ))

print('The highest probability of given observations is: ',math.exp(logprob)) # print out the highest probability
print('The HMM chain of hidden states is as following:')
print(' -> '.join(states[i] for i in seq)) # print out the hidden states.
print('\n')


given_obs = np.array([[0,1,0,1,0,1,0]]).transpose() # input observation
logprob, seq = model.decode(given_obs, algorithm='viterbi')

print('Given observations are as following:')
print(' -> '.join(observations[i] for item in given_obs for i in item ))

print('The highest probability of given observations is: ',math.exp(logprob)) # print out the highest probability
print('The HMM chain of hidden states is as following:')
print(' -> '.join(states[i] for i in seq)) # print out the hidden states.
print('\n')
