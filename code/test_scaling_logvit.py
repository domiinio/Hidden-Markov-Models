from hmm_inference import *
from robot import *
from test_robot import *
from utils import normalized
import numpy as np
import pandas as pd


def select_nbr_of_beliefs(beliefs, nbr_of_beliefs=1):
    states = []
    for belief in beliefs:
        i = 0
        xx = {}
        for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True):
            i += 1
            if i > nbr_of_beliefs:
                break
            xx[k] = v
        states.append(xx)
    return states


def test_all():
    robot_pos_init = (1, 1)
    robot = init_maze('mazes/square.map', robot_pos_init=robot_pos_init)
    states, obs = robot.simulate(init_state=robot_pos_init, n_steps=10)
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs_smoothing = forwardbackward_scaling(initial_belief, obs, robot)
    beliefs_filtering, _ = forward(initial_belief, obs, robot)
    viterbi_states, max_msgs = viterbi_log(initial_belief, obs, robot)

    nbr_of_beliefs = 1
    smoothing_states = select_nbr_of_beliefs(beliefs_smoothing, nbr_of_beliefs)
    filter_states = select_nbr_of_beliefs(beliefs_filtering, nbr_of_beliefs)

    states = list(map(list, states))
    viterbi_states = list(map(list, viterbi_states))
    '''print('True states ', states)
    print('Viterby states ', viterbi_states)
    print('Smooth states ', smoothing_states)
    print('Filter states ', filter_states)'''

    viterbi_dist = 0
    smooth_dist = 0
    filter_dist = 0
    viterbi_hit = 0
    smooth_hit = 0
    filter_hit = 0
    for i in range(len(states)):
        viterbi_dist += np.linalg.norm(np.array(states[i])-np.array(viterbi_states[i]), ord=1)
        if np.linalg.norm(np.array(states[i])-np.array(viterbi_states[i]), ord=1) == 0:
            viterbi_hit += 1
        smoothing_vals = list(smoothing_states[i].values())
        smoothing_keys = list(smoothing_states[i].keys())
        smoothing_vals_norm = np.array(smoothing_vals)/sum(smoothing_vals)
        for key, val_norm in zip(smoothing_keys, smoothing_vals_norm):
            d = np.linalg.norm(np.array([key[0], key[1]])-np.array(states[i]), ord=1)
            d = d*val_norm
            smooth_dist += d
            if d == 0:
                smooth_hit += 1

        filter_vals = list(filter_states[i].values())
        filter_keys = list(filter_states[i].keys())
        filter_vals_norm = np.array(filter_vals)/sum(filter_vals)
        for key, val_norm in zip(filter_keys, filter_vals_norm):
            d = np.linalg.norm(np.array([key[0], key[1]])-np.array(states[i]), ord=1)
            d = d*val_norm
            filter_dist += d
            if d == 0:
                filter_hit += 1

    print('Viterb err ', viterbi_dist, ',hits ', viterbi_hit)
    print('Smooth err ', smooth_dist, ',hits ', smooth_hit)
    print('Filter err ', filter_dist, ',hits ', filter_hit)
    return viterbi_dist, smooth_dist, filter_dist, viterbi_hit, smooth_hit, filter_hit


def calculate_err_and_hits():
    viterbi_dists = []
    smooth_dists = []
    filter_dists = []
    viterbi_hits = 0
    smooth_hits = 0
    filter_hits = 0
    for i in range(1):
        print('i: ', i)
        viterbi_dist, smooth_dist, filter_dist, viterbi_hit, smooth_hit, filter_hit = test_all()
        viterbi_dists.append(viterbi_dist)
        smooth_dists.append(smooth_dist)
        filter_dists.append(filter_dist)
        viterbi_hits += viterbi_hit
        smooth_hits += smooth_hit
        filter_hits += filter_hit
    a = np.asarray([viterbi_hits, smooth_hits, filter_hits])
    np.savetxt("temp_result.csv", a.T, delimiter=";")
    print(a)


def compare_FB_FBscaled():
    robot_pos_init = (1, 1)
    robot = init_maze('mazes/square.map', robot_pos_init=robot_pos_init)
    states, obs = robot.simulate(init_state=robot_pos_init, n_steps=100)
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs_FB = forwardbackward(initial_belief, obs, robot)
    beliefs_FB_scaled = forwardbackward_scaling(initial_belief, obs, robot)
    nbr_of_beliefs = 1
    FB_states = select_nbr_of_beliefs(beliefs_FB, nbr_of_beliefs)
    FB_scaled_states = select_nbr_of_beliefs(beliefs_FB_scaled, nbr_of_beliefs)

    guesses_match = []
    FB_hits = []
    FB_scaled_hits = []
    for i in range(len(states)):
        good_state = states[i]
        FB_state = list((FB_states[i]).keys())[0]
        FB_scaled_state = list((FB_scaled_states[i]).keys())[0]
        if FB_state == FB_scaled_state:
            guesses_match.append(1)
        else:
            guesses_match.append(0)
        if FB_state == good_state:
            FB_hits.append(1)
        else:
            FB_hits.append(0)
        if FB_scaled_state == good_state:
            FB_scaled_hits.append(1)
        else:
            FB_scaled_hits.append(0)
    print(sum(guesses_match))
    print(sum(FB_hits), sum(FB_scaled_hits))
    a = np.asarray([guesses_match, FB_hits, FB_scaled_hits])
    np.savetxt("temp_result.csv", a.T, delimiter=";")


def compare_vit_logvit():
    robot_pos_init = (1, 1)
    robot = init_maze('mazes/square.map', robot_pos_init=robot_pos_init)
    states, obs = robot.simulate(init_state=robot_pos_init, n_steps=5)
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    viterbi_states_log, max_msgs_log = viterbi_log(initial_belief, obs, robot)
    viterbi_states, max_msgs = viterbi(initial_belief, obs, robot)
    guesses_match = []
    vit_hits = []
    vit_log_hits = []
    for i in range(len(states)):
        good_state = states[i]
        vit_state = viterbi_states[i]
        vit_log_state = viterbi_states_log[i]
        if vit_state == vit_log_state:
            guesses_match.append(1)
        else:
            guesses_match.append(0)
        if vit_state == good_state:
            vit_hits.append(1)
        else:
            vit_hits.append(0)
        if vit_log_state == good_state:
            vit_log_hits.append(1)
        else:
            vit_log_hits.append(0)
    a = np.asarray([guesses_match, vit_hits, vit_log_hits])
    np.savetxt("temp_result.csv", a.T, delimiter=";")


if __name__ == '__main__':
    compare_vit_logvit()
    calculate_err_and_hits()
    compare_FB_FBscaled()