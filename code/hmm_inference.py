"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized
import numpy as np
import hmm_inference2
from robot import *
from hmm import *


def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    # Your code here
    for state in hmm.get_states():
        cur_B[state] = sum(prev_B[prev_state] * hmm.pt(prev_state, state) for prev_state in hmm.get_sources(state))
    return cur_B


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
    for _ in range(n_steps):
        B = update_belief_by_time_step(B, hmm)
        Bs.append(B)
    return Bs


def update_belief_by_evidence(prev_B, e, hmm):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :return: Counter, current (updated) belief distribution over states
    """
    # Create a new copy of the current belief state
    cur_B = Counter(prev_B)
    # Your code here
    for state in hmm.get_states():
        cur_B[state] = prev_B[state] * hmm.pe(state, e)
    return cur_B


def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    # Your code here
    cur_f = update_belief_by_time_step(prev_f, hmm)
    cur_f = update_belief_by_evidence(cur_f, cur_e, hmm)
    return cur_f


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, i.e., estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice
    norm_factors = []
    # Your code here
    for e in e_seq:
        f, norm_factor = normalized(forward1(f, e, hmm), factor=None, return_normalization_factor=True)
        fs.append(f)
        norm_factors.append(norm_factor)
    return fs, norm_factors


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    # Your code here
    l = prior
    lhood = None
    for e in e_seq:
        # compute likelihood messages
        l = forward1(l, e, hmm)
    # sum up dict values of final iteration of likelihood message
    lhood = sum(l.values())
    return lhood


def backward1(next_b, next_e, hmm):
    """Propagate the backward message
    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    # Your code here
    for state in hmm.get_states():
        cur_b[state] = sum(next_b[next_state]*(hmm.pt(state, next_state)*hmm.pe(next_state, next_e)) for next_state in hmm.get_states())
    return cur_b


def backward_scaling(e_seq, hmm, norm_factors):
    b = Counter({state: 1 for state in hmm.get_states()})
    bs = [b]
    i = 0
    norm_factors.reverse()
    for e in reversed(e_seq):
        b = normalized(backward1(b, e, hmm), factor=norm_factors[i])
        i += 1
        bs.append(b)
    return list(reversed(bs[:-1]))


def backward(e_seq, hmm):
    b = Counter({state: 1 for state in hmm.get_states()})
    bs = [b]
    for e in reversed(e_seq):
        b = backward1(b, e, hmm)
        bs.append(b)
    return list(reversed(bs[:-1]))


def forwardbackward_scaling(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []   # Smoothed belief distributions
    # Your code here
    fs, norm_factors = forward(priors, e_seq, hmm)
    bs = backward_scaling(e_seq, hmm, norm_factors)
    for f, b in zip(fs, bs):
        se.append(normalized(Counter({state:  f[state] * b[state] for state in hmm.get_states()})))
    return se


def forwardbackward(priors, e_seq, hmm):
    se = []  # Smoothed belief distributions
    # Your code here
    fs, _ = forward(priors, e_seq, hmm)
    bs = backward(e_seq, hmm)
    for f, b in zip(fs, bs):
        # se.append(normalized(Counter({state: f[state] * b[state] for state in hmm.get_states()})))
        se.append(Counter({state: f[state] * b[state] for state in hmm.get_states()}))
    return se


def viterbi1_log(prev_m, cur_e, hmm):
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
    # Your code here
    for x_cur in hmm.get_states():
        sources = []
        for x_prev in hmm.get_states():
            if hmm.pt(x_prev, x_cur) == 0:
                p = -np.inf
            else:
                p = np.log(hmm.pt(x_prev, x_cur)) + prev_m[x_prev]
            sources.append((p, x_prev))
        max_p, best_prev = max(sources)
        if hmm.pe(x_cur, cur_e) == 0:
            cur_m[x_cur] = -np.inf
        else:
            cur_m[x_cur] = max_p + np.log(hmm.pe(x_cur, cur_e))
        predecessors[x_cur] = best_prev
    return cur_m, predecessors


def viterbi1(prev_m, cur_e, hmm):
    cur_m = Counter()  # Current (updated) max message
    predecessors = {}  # The best of previous states for each current state
    # Your code here
    for x_cur in hmm.get_states():
        sources = []
        for x_prev in hmm.get_states():
            p = hmm.pt(x_prev, x_cur) * prev_m[x_prev]
            sources.append((p, x_prev))
        max_p, best_prev = max(sources)
        cur_m[x_cur] = max_p * hmm.pe(x_cur, cur_e)
        predecessors[x_cur] = best_prev
    return cur_m, predecessors


def viterbi_log(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []      # Sequence of max messages
    best_pred = []   # top prev states
    # Your code here
    m = forward1(priors, e_seq[0], hmm)
    m_new = Counter()
    # initial probabilities need to be log'ged as well
    for k, v in m.items():
        if v > 0:
            m_new[k] = np.log(v)
        else:
            m_new[k] = -np.inf
    m = m_new
    ms.append(m)
    for e in e_seq[1:]:
        m, pred = viterbi1_log(m, e, hmm)
        ms.append(m)
        best_pred.append(pred)
    # construct ML sequence
    final_state, final_prob = max(m.items(), key=lambda x: x[1])
    ml_seq = construct_ml_seq(best_pred, final_state)
    return ml_seq, ms


def viterbi(priors, e_seq, hmm):
    ml_seq = []  # Most likely sequence of states
    ms = []  # Sequence of max messages
    best_pred = []  # top prev states
    # Your code here
    m = forward1(priors, e_seq[0], hmm)
    ms.append(m)
    for e in e_seq[1:]:
        m, pred = viterbi1(m, e, hmm)
        ms.append(m)
        best_pred.append(pred)
    # construct ML sequence
    final_state, final_prob = max(m.items(), key=lambda x: x[1])
    ml_seq = construct_ml_seq(best_pred, final_state)
    return ml_seq, ms


def baumwelch(obs, initial_distribution, A, B, robot, n_iters=100):
    """ Baum-Welch algorithm for estimating values of HMM parameters

    :param obs: sequence of observations
    :param initial_distribution: prior belief
    :param A: initial transition matrix
    :param B: initial observation matrix
    :param robot: structure containing information about the HMM
    :param n_iters: number of iterations of the algorithm

    :return A: updated transition matrix
    :return B: updated observation matrix
    :return distribution: updated beliefs
    """

    # initialize values
    m = A.shape[0]  # number of hidden states
    t = len(obs)  # number of available observations
    all_obsvs = list(map(list, robot.get_observations()))
    for n in range(n_iters):
        # estimation step
        # get forward and backward variables
        alpha = to_array(hmm_inference2.forward(initial_distribution, obs, robot, A, B))
        beta = to_array(hmm_inference2.backward(obs, robot, A, B))
        # compute xi
        xi = np.zeros((m, m, t - 1))
        for t_i in range(t - 1):
            obs_i = list(robot.get_observations()).index(obs[t_i + 1])
            for s_i in range(m):
                for s_j in range(m):
                    xi[s_i, s_j, t_i] = alpha[t_i, s_i] * A[s_i, s_j] * B[s_j, obs_i] * beta[t_i + 1, s_j]
            xi[:, :, t_i] /= np.sum(xi[:, :, t_i])
        # compute gamma
        gamma = np.sum(xi, axis=1)
        # maximization step

        # update the transition matrix
        A = np.sum(xi, axis=2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, t - 2], axis=0).reshape((-1, 1))))

        # update the observation matrix
        for s_i in range(m):
            den = np.sum(gamma[s_i, :])
            for o_j in range(len(all_obsvs)):
                num = np.sum([gamma[s_i, t_i] for t_i in range(t - 1) if tuple(obs[t_i + 1]) == tuple(all_obsvs[o_j])])
                B[s_i, o_j] = num / den

                # update the uptimal distribution
        distribution = {pos: gamma_i for pos, gamma_i in zip(robot.get_states(), gamma[:, 0])}
    return A, B, distribution  # return the estimated transition and emission probabilities and the updated beliefs


def to_array(cntr):
    """ Convrets counter into np.array """
    out_arr = []
    for c in cntr:
        out_arr.append(list(c.values()))
    return np.array(out_arr)


def construct_ml_seq(predecessors, state):
    seq = [state]
    for pred in reversed(predecessors):
        state = pred[state]
        seq.append(state)
    return list(reversed(seq))

def init_maze():
    """Create and initialize robot instance for subsequent test"""
    m = Maze('mazes/rect_6x10_obstacles.map')
    print(m)
    robot = Robot()
    robot.maze = m
    robot.position = (1,1)
    print('Robot at ', robot.position)
    return robot


if __name__ == '__main__':
    """ Baum welch testing """
    np.set_printoptions(precision=3, suppress=True)
    import time

    robot = init_maze()
    states, obs = robot.simulate(init_state=(1, 1), n_steps=50)  # real states

    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    print(initial_belief)

    A = np.random.rand(len(robot.get_states()), len(robot.get_states()))
    A /= np.sum(A, axis=0)
    B = np.random.rand(len(robot.get_states()), len(robot.get_observations()))
    B /= np.sum(B, axis=0)

    distribution = initial_belief
    for i in range(10):
        print(
            f"Observation error: {np.linalg.norm(robot.get_observation_matrix() - B, 'fro')}, transition error: {np.linalg.norm(robot.get_transition_matrix() - A, 'fro')}")
        print(
            f"Likelihood true model: {hmm_inference2.likelihood(distribution, obs[1:], robot, robot.get_transition_matrix(), robot.get_observation_matrix())}, WB likelihood: {hmm_inference2.likelihood(distribution, obs[1:], robot, A, B)}")
        ti = time.time()
        A, B, distribution = baumwelch(obs, distribution, A, B, robot, 1)
        print(f"took {time.time() - ti:.2f}s")
