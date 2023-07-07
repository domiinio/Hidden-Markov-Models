"""
Script for testing the matrix implementation of HMM prediction algorithms

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from hmm_inference import *
import hmm_inference2
from robot import *
from utils import normalized, return_nonzero_elements
import time

direction_probabilities = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}


def init_maze():
    """Create and initialize robot instance for subsequent test"""
    m = Maze('mazes/rect_6x10_obstacles.map')
    print(m)
    robot = Robot()
    robot.maze = m
    robot.position = (1,1)
    print('Robot at ', robot.position)
    return robot


def test_filtering(n_steps):
    """Try to run filtering for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(n_steps=n_steps)
    print('Running filtering...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    start_time1 = time.time()
    beliefs, _ = forward(initial_belief, obs, robot)
    timef1 = time.time() - start_time1 
    start_time2 = time.time()
    beliefs2 = hmm_inference2.forward(initial_belief, obs, robot)
    timef2 = time.time() - start_time2 
    for i in range(len(beliefs)):
        print('another belief')
        print('max_diff:', max([beliefs[i][state] - beliefs2[i][state] for state in robot.get_states()]))
    
    print('naive:', timef1)
    print('matrix:', timef2)


def test_smoothing(n_steps):
    """Try to run smoothing for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(init_state=(1,10), n_steps=n_steps)
    print('Running smoothing...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    start_time1 = time.time()
    beliefs = forwardbackward(initial_belief, obs, robot)
    timef1 = time.time() - start_time1 
    start_time2 = time.time()
    beliefs2 = hmm_inference2.forwardbackward(initial_belief, obs, robot) 
    timef2 = time.time() - start_time2 
    for i in range(len(beliefs)):
        print('another belief')
        print('max_diff:', max([beliefs[i][state] - beliefs2[i][state] for state in robot.get_states()]))
    print('naive:', timef1)
    print('matrix:', timef2)



def test_viterbi(n_steps):
    """Try to run Viterbi alg. for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(init_state=(3,3), n_steps=n_steps)
    print('Running Viterbi...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    start_time1 = time.time()
    ml_states, max_msgs = viterbi(initial_belief, obs, robot)
    timef1 = time.time() - start_time1 
    start_time2 = time.time()
    ml_states2, max_msgs2 = hmm_inference2.viterbi(initial_belief, obs, robot)
    timef2 = time.time() - start_time2 
    misses = 0
    for real, est in zip(states, ml_states):
        i = ml_states.index(est)
        misses += 1 if est != ml_states2[i] else 0
    print('misses:', misses)
    print('naive:', timef1)
    print('matrix:', timef2)


if __name__=='__main__':
    print('Matrix implementation tests..')
    print('\nTesting the filtering algorithm')
    test_filtering(5)
    print('\nTesting the smoothing algorithm')
    test_smoothing(5)
    print('\nTesting the Viterbi algorithm')
    test_viterbi(5)

