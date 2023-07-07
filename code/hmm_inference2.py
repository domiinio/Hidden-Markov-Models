"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""
import numpy as np
from collections import Counter
from utils import normalized
import hmm_inference 
import hmm
import time

def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B = prior  # This shall be iteratively updated
    Bs = []    # This shall be a collection of Bs over time steps
    A = hmm.get_transition_matrix()
    for _ in range(n_steps):
        b = np.array([B[state] for state in hmm.get_states()])
        update = A @ b               # solve the matrix equation
        # store the results
        B2 = Counter()
        for key, val in zip(hmm.get_states(), update):
            B2[key] = val             
        B = B2
        Bs.append(B)
    return Bs

def forward1(prev_f, cur_e, hmm, A=None, B=None):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    if A is None or B is None:
        A = hmm.get_transition_matrix()
        B = hmm.get_observation_matrix()
    i = list(hmm.get_observations()).index(cur_e) # find index of current observation
    O = np.diag(B[:,i]) # diagonal matrix giving probs of a state leading to an observation
    b = np.array([prev_f[state] for state in hmm.get_states()]) # create vector of probabilites
    update = O @ (A @ b)    # solve the matrix equation
    # store the results
    cur_f = Counter()
    for state in hmm.get_states():
        cur_f[state] = update[list(hmm.get_states()).index(state)]
    return cur_f


def forward(init_f, e_seq, hmm, A=None, B=None):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, i.e., estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice
    if  A is None  or B is None:
        A = hmm.get_transition_matrix()
        B = hmm.get_observation_matrix()
    
    for e in e_seq:
        f= forward1(f, e, hmm, A, B)
        fs.append(f)  # scaling with forward algorithm
    return fs


def likelihood(prior, e_seq, hmm, A=None, B=None):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    l = prior 
    for e in e_seq:
        l = forward1(l,e,hmm, A, B)
    return sum(l.values())


def backward1(next_b, next_e, hmm, A=None, B=None):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    if A is None or B is None:
        A = hmm.get_transition_matrix()
        B = hmm.get_observation_matrix()
    i = list(hmm.get_observations()).index(next_e) # find index of next observation
    O = np.diag(B[:,i]) # diagonal matrix giving probs of a state leading to an observation
    b = np.array([next_b[state] for state in hmm.get_states()]) # create vector of probabilites
    # solve matrix equation
    update = A.T @ (O @ b)    
    cur_b = Counter()
    for key, val in zip(hmm.get_states(), update):
        cur_b[key] = val             
    return cur_b


def backward(e_seq, hmm, A=None, B=None, norm_factors=None):
    if A is None or B is None:
        A = hmm.get_transition_matrix()
        B = hmm.get_observation_matrix()

    b = Counter({state: 1 for state in hmm.get_states()})
    bs = [b]
    if norm_factors is not None:
        norm_factors.reverse()
        i = 0
        for e in reversed(e_seq):
            b = normalized(backward1(b,e,hmm, A, B), factor=norm_factors[i])
            i += 1
            bs.append(b)
    else:
        for e in reversed(e_seq):
            b = backward1(b,e,hmm, A, B)
            bs.append(b)
    return list(reversed(bs[:-1]))


def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []   # Smoothed belief distributions
    fs = forward(priors, e_seq, hmm)
    bs = backward(e_seq, hmm)
    for f,b in zip(fs, bs):
        se.append(normalized(Counter({state: f[state]*b[state] for state in hmm.get_states()})))
    return se


def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()   # Current (updated) max message
    predecessors = {}   # The best of previous states for each current state

    start_time1 = time.time()

    A = hmm.get_transition_matrix()   # matrix of transition probabilities
    B = hmm.get_observation_matrix()  # matrix of emission probabilities
    i = list(hmm.get_observations()).index(cur_e) # find index of current observation
    O = np.diag(B[:,i]) # diagonal matrix giving probs of a state leading to an observation
    b = np.array([prev_m[state] for state in hmm.get_states()]) # create vector of probabilites

    p = np.multiply(A,b)                    # do the forward step
    max_p = np.amax(p, axis=1)              # find probability of best predecessor
    best_predecessors = np.argmax(p, axis=1) # find best predecessor
    p = O @ max_p                           # compute final probability
    
    # fill the data and return new counter
    all_states = list(hmm.get_states())
    #all_observations = list(hmm.get_observations())
    for state in all_states:
        cur_m[state] = p[all_states.index(state)]
        predecessors[state] = all_states[best_predecessors[all_states.index(state)]]    
    return cur_m, predecessors


def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief dstribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []      # Sequence of max messages

    m = forward1(priors, e_seq[0],hmm)
    ms.append(m)
    predecessors = []
    for e in e_seq[1:]:
        m, pred = viterbi1(m, e, hmm)
        ms.append(m)
        predecessors.append(pred)
    # construct ML Sequence
    final_state, _ = max(m.items(),key=lambda x:x[1])
    ml_seq = construct_ml_seq(predecessors, final_state)
    return ml_seq, ms


def construct_ml_seq(predecessors, state):
    seq = [state]
    for pred in reversed(predecessors):
        state = pred[state]
        seq.append(state)
    return list(reversed(seq))


if __name__=='__main__':
    #from weather import WeatherHMM 
    #wtr = WeatherHMM()
    B = Counter({'+rain': 0.1,'-rain': 0.9})
    #B_new = update_belief_by_time_step(B, wtr)
    #print(B_new)
    #Counter({'+rain': 0.16,'-rain': 0.84})
    #wtr = WeatherHMM()
    #B = Counter({'+rain': 0.5,'-rain': 0.5})
    #B_new = update_belief_by_evidence(B,'-umb', wtr)
    #print(B_new)
    #print(normalized(B_new))
    #Counter({'+rain': 0.11111111111111112,'-rain': 0.888888888888889})
    #wtr = WeatherHMM()
    #b = Counter({'+rain': 0.6,'-rain': 0.7})
    #e ='+umb'
    #n_b = backward1(b, e, wtr)
    #print(n_b)

