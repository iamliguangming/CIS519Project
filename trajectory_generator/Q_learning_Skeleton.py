#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:45:59 2020

@author: yupengli
"""

import numpy as np 
from scipy.spatial.transform import Rotation
from flightsim.crazyflie_params import quad_params
from generator.code.occupancy_map import OccupancyMap
from generator.code.se3_control import SE3Control
from generator.code.world_traj import WorldTraj
from flightsim.simulate import simulate
from flightsim.simulate import Quadrotor
from flightsim.world import World


world = World.random_block(lower_bounds=(-2, -2, 0), upper_bounds=(3, 2, 2), 
                           block_width=0.5, block_height=1.5,
                           num_blocks=4, robot_radii=0.25, margin=0.2) 
resolution=(.1, .1, .1)
margin=.2
occ_map = OccupancyMap(world,resolution,margin)
my_se3_control = SE3Control(quad_params)
start = world.world['start']  # Start point, shape=(3,)
goal = world.world['goal']  # Goal point, shape=(3,)
my_world_traj = WorldTraj(world, start, goal)
t_final = 60
quadrotor = Quadrotor(quad_params)
initial_state = {'x': start,
                    'v': (0, 0, 0),
                    'q': (0, 0, 0, 1), # [i,j,k,w]
                    'w': (0, 0, 0)}
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                                  quadrotor,
                                                  my_se3_control,
                                                  my_world_traj,
                                                  t_final)

action_List = control['cmd_motor_speeds']
t_step = 2e-3

def search_direction(position_index,direction,steps,occ_map):
    """
    This subrountine takes in current position and direction searched within
    a specific map and returns the steps to the closest obstacle in the 
    specified direction.
    If there is no obstacle within the steps, return 0
    
    """
    for i in range(1,steps+1):
        current_index = position_index + i*np.array(direction)
        if occ_map.is_occupied_index(
                current_index)or not occ_map.is_valid_index(current_index):
            return i
    return 0
        
def get_extended_state(state, occ_map):
    """
    Input: state as a dictionary
            x, position, m, shape=(3,)
            v, linear velocity, m/s, shape=(3,)
            q, quaternion [i,j,k,w], shape=(4,)
            w, angular velocity, rad/s, shape=(3,)
    Return: Updated  state as a dictionary
            x, position, m, shape=(3,)
            v, linear velocity, m/s, shape=(3,)
            theta, Euler Angle, rad, shape = (3,)
            w, angular velocity, rad/s, shape=(3,)
            d, closest neighbor distance, # of blocks, shape(14,)
    """
    position = state['x']
    velocity = state['v']
    orientation = Rotation.from_quat(state['q'])
    theta = orientation.as_euler('XYZ')
    w = state['w']
    position_index = occ_map.metric_to_index(position)
    d = np.array([search_direction(position_index, [1,0,0], 3, occ_map),
                  search_direction(position_index, [0,1,0], 3, occ_map),
                  search_direction(position_index, [0,0,1], 3, occ_map),
                  search_direction(position_index, [1,1,1], 3, occ_map),
                  search_direction(position_index, [-1,0,0], 3, occ_map),
                  search_direction(position_index, [0,-1,0], 3, occ_map),
                  search_direction(position_index, [0,0,-1], 3, occ_map),
                  search_direction(position_index, [-1,-1,-1], 3, occ_map),
                  search_direction(position_index, [1,1,-1], 3, occ_map),
                  search_direction(position_index, [1,-1,1], 3, occ_map),
                  search_direction(position_index, [1,-1,-1], 3, occ_map),
                  search_direction(position_index, [-1,1,1], 3, occ_map),
                  search_direction(position_index, [-1,-1,1], 3, occ_map),
                  search_direction(position_index, [-1,1,-1], 3, occ_map)])
    
    extended_state = {'x':position,
                      'v':velocity,
                      'theta':theta,
                      'w':w,
                      'd':d}
    
    return extended_state

vectorized_extended_state_List = np.zeros((state['x'].shape[0],30))
for i in range(state['x'].shape[0]):
    current_state = {'x':state['x'][i],
                     'v':state['v'][i],
                     'q':state['q'][i],
                     'w':state['w'][i],
                     }
    extended_state = get_extended_state(current_state, occ_map)
    vectorized_extended_state = np.concatenate((extended_state['x'],
                                                extended_state['v'],
                                                extended_state['theta'],
                                                extended_state['w'],
                                                extended_state['d'],
                                                action_List[i,:]),axis=0)
    vectorized_extended_state_List[i] = vectorized_extended_state
    
# The following section is for Q learning

def discretize(state, discretization, env):
    """
    Need to work on this
    """
    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state


def choose_action(epsilon, Q, state, env):
    """    
    Choose an action according to an epsilon greedy strategy.
    Args:
        epsilon (float): the probability of choosing a random action
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state (Box(2,)): the observation state, here it is [position, velocity]
        env: the RL environment 
        
    Returns:
        action (int): the chosen action
    """
    action = 0
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]]) #Need to change the dim
    else:
        action = np.random.randint(0, env.action_space.n)
  
    return action


def update_epsilon(epsilon, decay_rate):
    """
    Decay epsilon by the specified rate.
    
    Args:
        epsilon (float): the probability of choosing a random action
        decay_rate (float): the decay rate (between 0 and 1) to scale epsilon by
        
    Returns:
        updated epsilon
    """
  
    epsilon *= decay_rate

    return epsilon


def update_Q(Q, state_disc, next_state_disc, action, discount, learning_rate, reward, terminal):
    """
    
    Update Q values following the Q-learning update rule. 
    
    Be sure to handle the terminal state case.
    
    Args:
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state_disc (np.array): the discretized version of the current observation state [position, velocity]
        next_state_disc (np.array): the discretized version of the next observation state [position, velocity]
        action (int): the chosen action
        discount (float): the discount factor, may be referred to as gamma
        learning_rate (float): the learning rate, may be referred to as alpha
        reward (float): the current (immediate) reward
        terminal (bool): flag for whether the state is terminal
        
    Returns:
        Q, with the [state_disc[0], state_disc[1], action] entry updated.
    """    
    if terminal:        
        Q[state_disc[0], state_disc[1], action] = reward

    # Adjust Q value for current state
    else:
        delta = learning_rate*(reward + discount*np.max(Q[next_state_disc[0], next_state_disc[1]]) - Q[state_disc[0], state_disc[1],action])
        Q[state_disc[0], state_disc[1],action] += delta
  
    return Q


def Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes=5000):
    """
    
    The main Q-learning function, utilizing the functions implemented above.
    Need to change to choose actions of discretized action space
          
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached 
    frames = []
  
    for i in range(max_episodes):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = env.reset() # initial environment state
        state_disc = discretize(state,discretization,env)

        while done != True:                 
            # Determine next action 
            action = choose_action(epsilon, Q, state_disc, env)                                      
            # Get next_state, reward, and done using env.step(), see http://gym.openai.com/docs/#environments for reference
            if i==1 or i==(max_episodes-1):
              frames.append(env.render())
            next_state, reward, done, _ = env.step(action) 
            # Discretize next state 
            next_state_disc = discretize(next_state,discretization,env)
            # Update terminal
            terminal = done and next_state[0]>=0.5
            # Update Q
            Q = update_Q(Q,state_disc,next_state_disc,action,discount,learning_rate, reward, terminal)  
            # Update tot_reward, state_disc, and success (if applicable)
            tot_reward += reward
            state_disc = next_state_disc
            if terminal: success +=1 
            
        epsilon = update_epsilon(epsilon, decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state[0])
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',np.mean(reward_list))
            reward_list = []
                
    env.close()
    
    return Q, position_list, success_list, frames

# num_states = (env.observation_space.high - env.observation_space.low)*discretization
# #Size of discretized state space 
# num_states = np.round(num_states, 0).astype(int) + 1
# # Initialize Q table
# Q = np.random.uniform(low = -1, 
#                       high = 1, 
#                       size = (num_states[0], num_states[1], env.action_space.n))

# # Run Q Learning by calling your Qlearning() function
# Q, position, successes, frames = Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes)

class StatesNetwork(nn.Module):
  '''
  This NN should take state action pairs and return a Q value
  '''
    def __init__(self, env):
        """
        Your code here
        """
    
    def forward(self, x):    
        """
        Your code here
        @x: torch.Tensor((B,dim_of_observation))
        @return: torch.Tensor((B,dim_of_actions))
        """
    
    return forward_pass

# def train():
