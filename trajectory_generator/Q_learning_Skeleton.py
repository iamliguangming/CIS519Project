#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:45:59 2020

@author: yupengli
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
from flightsim.crazyflie_params import quad_params
from generator.code.occupancy_map import OccupancyMap
from generator.code.se3_control import SE3Control
from generator.code.world_traj import WorldTraj
from generator.code.graph_search import graph_search
from flightsim.simulate import simulate
from flightsim.simulate import Quadrotor
from flightsim.world import World
from tqdm import tqdm
from flightsim.world import ExpectTimeout

def search_direction(args, position_index, direction, steps):
    """
    This subrountine takes in current position and direction searched within
    a specific map and returns the steps to the closest obstacle in the
    specified direction.
    If there is no obstacle within the steps, return 0

    """
    occ_map = args.occ_map
    for i in range(1,steps+1):
        current_index = position_index + i*np.array(direction)
        if not occ_map.is_valid_index(current_index) or occ_map.is_occupied_index(
                current_index):
            return i
    return 0

def get_warmup_data(args, state, action):
    """
    Input: state a vector (p)   p = (3,)
           action a vector (a)  a = (3,)

    Output: train_set = nparray(27,23(20+3))
            label = nparray((27,))

    """
    chosen_action = action.tolist()
    extended_state = get_extended_state(args,state)   ##(20,)
    extended_state = np.tile(extended_state, (27,1))   ##(27,20)
    all_action =[
     [-1, -1, -1],[-1, -1, 0],[-1, -1, 1],
     [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
     [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
     [0, -1, -1], [0, -1, 0], [0, -1, 1],
     [0, 0, -1],  [0, 0, 0],  [0, 0, 1],
     [0, 1, -1],  [0, 1, 0],  [0, 1, 1],
     [1, -1, -1], [1, -1, 0], [1, -1, 1],
     [1, 0, -1],  [1, 0, 0],  [1, 0, 1],
     [1, 1, -1],  [1, 1, 0],  [1, 1, 1]]     ##(27,3)
    nonchosen_action = all_action.copy()
    nonchosen_action.remove(chosen_action)  ##list (26,3)
    nonchosen_action = np.asarray(nonchosen_action)
    next_state_nonchosen = nonchosen_action + np.tile(state,(26,1)) ##array(26,3)
    Q_list=[]
    for i in next_state_nonchosen:
        if args.occ_map.is_valid_index(i) and not args.occ_map.is_occupied_index(i):
            Q_list.append(-1)   #non_chosen action in open space
        else:
            Q_list.append(-10)   #non_chosen action with collision

    all_action = np.concatenate((np.array(nonchosen_action),
                                 np.array(chosen_action).reshape(1,3)),axis = 0) ##(27,3)
    Q_list.append(1)    #chosen action

    train_set = np.hstack((extended_state,all_action))  ##(27,20+3)
    train_labels = np.asarray(Q_list).reshape(len(Q_list),1)
    return train_set, train_labels

def get_all_warm_up(args,discretized_path,action_List):
    train_set,train_labels = get_warmup_data(args,
                                             discretized_path[0],
                                             action_List[0])
    for i in range(1,discretized_path.shape[0]):
        print(f'{i} out of {discretized_path.shape}')
        new_train_set, new_train_labels = get_warmup_data(args,
                                                          discretized_path[i],
                                                          action_List[i])
        train_set = np.concatenate((train_set, new_train_set),axis = 0)
        train_labels = np.concatenate((train_labels, new_train_labels),axis =0)
    return train_set, train_labels

def get_extended_state(args, state):

    """
    Input: state as a vector (p)
            p, discretized_Position shape = (3,)
    Return: Updated  state as a vector (p,v,d)
            p, discretized_Position shape = (3,)
            v, discretized directions shape = (3,)
            d, closest neighbor distance, # of blocks, shape(14,)
    """
    position_index = state
    d = np.array([search_direction(args, position_index, [1,0,0], args.search_range),
                  search_direction(args, position_index, [0,1,0], args.search_range),
                  search_direction(args, position_index, [0,0,1], args.search_range),
                  search_direction(args, position_index, [1,1,1], args.search_range),
                  search_direction(args, position_index, [-1,0,0], args.search_range),
                  search_direction(args, position_index, [0,-1,0], args.search_range),
                  search_direction(args, position_index, [0,0,-1], args.search_range),
                  search_direction(args, position_index, [-1,-1,-1], args.search_range),
                  search_direction(args, position_index, [1,1,-1], args.search_range),
                  search_direction(args, position_index, [1,-1,1], args.search_range),
                  search_direction(args, position_index, [1,-1,-1], args.search_range),
                  search_direction(args, position_index, [-1,1,1], args.search_range),
                  search_direction(args, position_index, [-1,-1,1], args.search_range),
                  search_direction(args, position_index, [-1,1,-1], args.search_range)])

    extended_state = np.append(state,d)
    extended_state = np.append(extended_state,args.goal)

    return extended_state

# The following section is for Q learning
def step(args, state, action):
    """
    Inputs:
    state: Original state with only position indices: np array (3,)
    action: Action array (3,)
    args, an object with set of parameters and objects
    Output: Updated State (3,)
            Reward of the action
            Done: Boolean, True if reaches the goal or hit the wall

    """
    done = False
    distance_before_action = np.linalg.norm(args.goal - state)
    state = state + action
    distance_after_action =  np.linalg.norm(args.goal - state)
    distance_traveled = np.linalg.norm(action)
    reward = (distance_before_action - distance_after_action
              - distance_traveled) / distance_before_action
    if args.occ_map.is_occupied_index(state) or not args.occ_map.is_valid_index(state):
        reward = -10
        done = True
    elif (state == args.goal).all():
        reward = 10
        done = True

    return state, reward, done

def get_pair(args, state, action):
    """
    Get the extended state + action vector

    Input: args: The argument dictionary
           state : The raw state (3,)
           action : T
    """
    extended_state = get_extended_state(args, state)
    extended_state_action = np.append(extended_state,action)
    return extended_state_action

def get_all_pairs(args, state):
    """
    Combine states and actions to pairs
    Args:
        args, an object with set of parameters and objects
        state: Original state with only position indices: np array (3,)
    """
    all_pairs = np.zeros((27,23))
    action_array = np.zeros((27,3))
    count = 0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                action = np.array([i,j,k])
                action_array[count] = action
                all_pairs[count] = get_pair(args,state,action)
                count += 1
    return all_pairs, action_array.astype(int)

def choose_action(args,state,epsilon):
    """
    Choose an action according to an epsilon greedy strategy.
    Args:
        args, an object with set of parameters and objects
        state : raw state of position : nparray(3,)
        epsilon (float): the probability of choosing a random action

    Returns:
        chosen_action (3,) np_array: the chosen action
        Q value of chosen_action (int)
    """
    all_pairs, action_array = get_all_pairs(args, state)
    Q_array = np.zeros(27)
    for i in range(27):
        Q_array[i] = args.model.predict(torch.tensor(all_pairs[i]).float())

    chosen_action = np.zeros((3,))
    if np.random.random() < 1 - epsilon:
        chosen_action = action_array[np.argmax(Q_array)]
    else:
        chosen_action = action_array[np.random.randint(0, 27)]

    return chosen_action, np.max(Q_array)


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


def get_target_Q(args, state, next_state, action, reward, terminal):
    """

    Update Q values following the Q-learning update rule.

    Be sure to handle the terminal state case.

    Args:
        args, an object with set of parameters and objects
        state (np.array): current state with only position indices: np array (3,)
        next_state (np.array): next state with only position indices: np array (3,)
        action (int): the chosen action
        reward (float): the current (immediate) reward
        terminal (bool): flag for whether the state is terminal

    Returns:
        Q, with its value updated.
    """
    Q = args.model.predict(torch.tensor(get_pair(args,state,action)).float())
    _, next_Q = choose_action(args,next_state,0)

    if terminal:
        Q = reward

    # Adjust Q value for current state
    else:
        delta = args.lr*(reward + args.discount*next_Q - Q)
        Q += delta

    return Q

def aggregate_dataset(extended_state_list, Q_array, new_extended_state, new_Q):
    training_states = np.concatenate((extended_state_list,new_extended_state.reshape(1,-1)),axis=0)
    training_Q = np.concatenate((Q_array,new_Q.detach().numpy()),axis = 0)

    return training_states, training_Q

def Qlearning(args):
    """
    The main Q-learning function, utilizing the functions implemented above.
    Need to change to choose actions of discretized action space
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached

    for i in tqdm(range(args.max_episodes), position = 0):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = args.start

        while done != True:
            # Determine next action
            action,_ = choose_action(args,state,args.epsilon)
            next_state, reward, done = step(args,state,action)
            # Update terminal
            terminal = done and np.linalg.norm(state - args.goal) <= args.tol
            # Update Q
            Q = get_target_Q(args,state,next_state,action,reward,terminal)
            # Update tot_reward, state_disc, and success (if applicable)
            state_action_pair = get_pair(args,state,action)
            args.train_set,args.train_labels = aggregate_dataset(
                args.train_set,args.train_labels,state_action_pair,Q)

            tot_reward += reward
            state = next_state
            if terminal: success +=1


        args.dataloader = load_dataset(args.train_set,args.train_labels)
        train(args)
        args.epsilon = update_epsilon(args.epsilon, args.decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state.tolist())
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',np.mean(reward_list))
            reward_list = []

    return reward_list, position_list, success_list

class QNetwork(nn.Module):
    """
    This NN should take state action pairs and return a Q value
    """
    def __init__(self):
        super(QNetwork,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(23, 50),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0),-1)
        forward_pass = self.fc(x)

        return forward_pass

    def predict(self,x):

        return self.forward(x.reshape(-1,23).float())

def load_dataset(x, y, batch_size=64):
    """
    load data for neural network

    Args:
        x: state action pairs
        y: Q value labels
    """
    x = torch.tensor(x).float()
    y = torch.tensor(y.flatten()).float()
    dataset = torch.utils.data.TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


def train(args):
    '''
    Train the model

    Args:
        args, an object with set of parameters and objects

    Returns:
        The trained model
    '''
    model = args.model
    optimizer = args.optimizer
    criterion = args.criterion
    dataloader = args.dataloader

    running_loss = 0.0
    running_corrects = 0
    model.train()
    for e in range(args.num_epochs):
        for inputs, labels in (dataloader):
            labels = labels
            optimizer.zero_grad()
            # outputs = model(inputs)
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, axis=1)
            loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()* inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # print(f'epoch {e}')

        # epoch_loss = running_loss / dataset_size
        # epoch_acc = running_corrects.double() / dataset_size
        # print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model

def get_args():
    class Args(object):
        pass
    args = Args()
    args.occ_map = occ_map
    args.model = QNetwork()
    args.lr = 0.2
    args.discount = 0.9
    args.epsilon = 0.8
    args.decay_rate = 0.95
    args.max_episodes = 100
    args.tol = 1e-4

    args.train_set = None
    args.train_labels = None
    args.dataloader = None
    # args.dataset_size=0
    args.batch_size = 120
    args.num_epochs = 20
    args.optimizer = torch.optim.Adam(args.model.parameters(),args.lr)
    args.criterion = torch.nn.MSELoss()
    return args

if __name__ == '__main__':
    for _ in range(20):
        try:
            with ExpectTimeout(3):
                world = World.random_block(lower_bounds=(-2, -2, 0), upper_bounds=(3, 2, 2),
                               block_width=0.5, block_height=1.5,
                               num_blocks=4, robot_radii=0.25, margin=0.2)
                break
        except TimeoutError:
            pass

    resolution=(.1, .1, .1)
    margin=.2
    occ_map = OccupancyMap(world,resolution,margin)
    # my_se3_control = SE3Control(quad_params)
    start = world.world['start']  # Start point, shape=(3,)
    goal = world.world['goal']  # Goal point, shape=(3,)
    # my_world_traj = WorldTraj(world, start, goal)
    my_path = graph_search(world, resolution, margin, start, goal, False)[1:-1]
    start = my_path[0]
    goal = my_path[-1]
    # t_final = 60
    quadrotor = Quadrotor(quad_params)
    # initial_state = {'x': start,
    #                     'v': (0, 0, 0),
    #                     'q': (0, 0, 0, 1), # [i,j,k,w]
    #                     'w': (0, 0, 0)}
    # (sim_time, state, control, flat, exit) = simulate(initial_state,
    #                                                   quadrotor,
    #                                                   my_se3_control,
    #                                                   my_world_traj,
    #                                                   t_final)
    args = get_args()



    action_List = np.zeros((my_path.shape))
    discretized_path = np.zeros((my_path.shape))

    for i in range(discretized_path.shape[0]):
        discretized_path[i,:] = args.occ_map.metric_to_index(my_path[i,:])
    for i in range(discretized_path.shape[0]):
        try:
            action_List[i,:] = discretized_path[i+1]-discretized_path[i]
        except:
            action_List[i,:] = np.zeros(3)

    discretized_path = discretized_path.astype(int)
    action_List = action_List.astype(int)
    start_index = discretized_path[0]
    goal_index = discretized_path[-1]
    args.start = start_index
    args.goal = goal_index
    args.search_range = 20
    # t_step = 2e-3

    # ext = get_extended_state(discretized_path[0], occ_map,args)

    # extended_state_List = np.zeros((discretized_path.shape[0],20))
    # for i in range(discretized_path.shape[0]):
    #     current_state = discretized_path[i]
    #     extended_state = get_extended_state(args, current_state)


    #     extended_state_List[i] = extended_state
    warm_up_set, warm_up_labels = get_all_warm_up(args,
                                                  discretized_path,
                                                  action_List)
    args.dataloader = load_dataset(warm_up_set,warm_up_labels)
    args.train_set = warm_up_set
    args.train_labels = warm_up_labels
    args.model = train(args)

    state = discretized_path[0]

    all_pairs, action_array = get_all_pairs(args, state)

    print(args.model.forward(torch.tensor(all_pairs[0].reshape(1,-1)).float()))

    reward_list, position_list, success_list = Qlearning(args)
    print(reward_list)
    print(position_list)
    print(success_list)
    # num_states = (env.observation_space.high - env.observation_space.low)*discretization
    # #Size of discretized state space
    # num_states = np.round(num_states, 0).astype(int) + 1
    # # Initialize Q table
    # Q = np.random.uniform(low = -1,
    #                       high = 1,
    #                       size = (num_states[0], num_states[1], env.action_space.n))

    # # Run Q Learning by calling your Qlearning() function
    # Q, position, successes, frames = Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes)
