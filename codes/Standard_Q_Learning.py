#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:45:59 2020

@author: yupengli
"""

import numpy as np
import torch
import torch.nn as nn
from torch import save, load
from torch.utils.data import DataLoader
from generator.code.occupancy_map import OccupancyMap
from generator.code.graph_search import graph_search
from flightsim.world import World
from tqdm import tqdm
from flightsim.world import ExpectTimeout
from flightsim.axes3ds import Axes3Ds
import matplotlib.pyplot as plt
from flightsim.animate import animate

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
    extended_state = np.tile(extended_state, (53,1))   ##(27,20)
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
            Q_list.append(0)   #non_chosen action in open space
        else:
            Q_list.append(-100)   #non_chosen action with collision

    all_action = np.concatenate((np.array(nonchosen_action),
                                 np.array(chosen_action).reshape(1,3)),axis = 0) ##(27,3)
    Q_list.append((10))   #chosen action
    for i in range(26):
        all_action = np.concatenate((np.array(all_action),
                                     np.array(chosen_action).reshape(1,3)),axis = 0)
        Q_list.append((100))
    train_set = np.hstack((extended_state,all_action))  ##(27,20+3)
    train_labels = np.asarray(Q_list).reshape(len(Q_list),1)
    return train_set, train_labels

def get_all_warm_up(args,discretized_path,action_List):
    train_set,train_labels = get_warmup_data(args,
                                             discretized_path[0],
                                             action_List[0])
    for i in range(1,discretized_path.shape[0]):
        # print(f'{i} out of {discretized_path.shape}')
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
    reward = (distance_before_action - distance_after_action
              ) / distance_before_action
    if not args.occ_map.is_valid_index(state) or args.occ_map.is_occupied_index(state):
        reward = -1
        done = True
    elif (state == args.goal).all():
        reward = 1
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
        print('I am searching\n')

    # print (chosen_action)
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


def get_target_Q(args, state, next_state, action, reward, terminal,done):
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
    if not args.occ_map.is_valid_index(
            next_state) or args.occ_map.is_occupied_index(next_state):
        next_Q = torch.tensor([[-100]])

    if terminal:
        print('\nReach goal!')
        print(f'Q before update:{Q.detach().numpy().item()}')
        Q = np.array([[reward]])*100
        print(f'Q after update:{Q.item()}')

    else:
        delta = args.Qlr*(100*reward + args.discount*next_Q - Q)

        print(f'\nState: {state}, Next state: {next_state}')
        print(f'Reward: {reward}, next_Q: {next_Q}')
        print(f'Delta:{delta.detach().numpy().item()}')
        print(f'Q before update:{Q.detach().numpy().item()}')

        Q += delta
        Q = Q.detach().numpy()
        print(f'Q after update:{Q.item()}')

    return Q

def aggregate_dataset(extended_state_list, Q_array, new_extended_state, new_Q):
    training_states = np.concatenate((extended_state_list,new_extended_state.reshape(1,-1)),axis=0)
    training_Q = np.concatenate((Q_array,new_Q),axis = 0)

    return training_states, training_Q

def add_replace_element(args,new_pair,new_Q):
    inFlag = False
    for i in range(args.train_set.shape[0]):
        if (args.train_set[i] == new_pair).all():
            args.train_labels[i] = new_Q
            inFlag = True
            break
    if not inFlag:
        args.train_set,args.train_labels = aggregate_dataset(
             args.train_set,args.train_labels,new_pair,new_Q)

    return None

def Qlearning(args):
    """
    The main Q-learning function, utilizing the functions implemented above.
    Need to change to choose actions of discretized action space
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached

    success_array_5= 0

    args.log_permit = False
    
    for i in tqdm(range(args.max_episodes), position = 0):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = args.start
        num_steps = 0
        path_length = 0
        path_list = []
        flag = 0
        delete_action_list=[]
        print(f'\n Searching Likelihood: {args.epsilon}')

        while done != True and num_steps <= args.max_steps:
            # Determine next action
            path_list.append((args.occ_map.index_to_metric_center(state)).tolist())
            action,_ = choose_action(args,state,args.epsilon)
            next_state, reward, done = step(args,state,action)
            # Update terminal
            terminal = (done and (np.linalg.norm(next_state - args.goal) <= args.tol))
            # Update Q
            Q = get_target_Q(args,state,next_state,action,reward,terminal,done)
            # Update tot_reward, state_disc, and success (if applicable)
            state_action_pair = get_pair(args,state,action)
            add_replace_element(args, state_action_pair, Q)

            tot_reward += reward
            path_length += np.linalg.norm(next_state - state)
            state = next_state
            if terminal: success += 1
            num_steps += 1

        time = np.zeros((len(path_list),))
        for j in range(1,len(time)):
            time[j] = time[j-1] + args.max_time/len(time)
            
        if args.animate_permit is True:
            position = np.asarray(path_list)
            rotation = np.full((len(time),3,3),np.identity(3))
            animate(args.st,args.go,time, position, rotation, args.world,
                    filename = 'episode_'+str(i)+'.mp4',show_axes = True)
            
        if terminal and path_length < args.best_path_length:
            args.best_path_length = path_length
            args.final_path = path_list

        args.dataloader = load_dataset(args.train_set, args.train_labels, batch_size = 20)
        train(args)
        args.epsilon = update_epsilon(args.epsilon, args.decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state.tolist())
        success_array_5 += success
        if i==0 or i % 5==4:
            success_list.append(success_array_5/5)
            success_array_5 = 0
        success = 0
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
            nn.Linear(50,50),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0),-1)
        forward_pass = self.fc(x)

        return forward_pass

    def predict(self,x):

        return self.forward(x.reshape(-1,23).float())

def load_dataset(x, y, batch_size=5):
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

def save_model(model):
    return save(model.state_dict(), 'QNet.th')

def load_model():
    q = QNetwork()
    q.load_state_dict(load('QNet.th', map_location='cpu'))
    return q

def train(args):
    '''
    Train the model

    Args:
        args, an object with set of parameters and objects

    Returns:
        The trained model
    '''
    print('\nTraining...')
    
    model = args.model
    optimizer = args.optimizer
    criterion = args.criterion
    dataloader = args.dataloader

    best_epoch = 0
    best_loss = 1E8

    train_flag = True

    running_loss = 0.0
    running_corrects = 0

    model.train()
    for e in range(args.num_epochs):
        if train_flag is False:
            break
        running_loss = 0
        running_corrects = 0
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
            running_corrects += torch.sum(abs(outputs.reshape(-1).detach()-labels.data)<0.01)

        epoch_loss = running_loss / len(args.train_set)
        epoch_acc = running_corrects.double() / len(args.train_set)

        args.loss_list.append(epoch_loss)

        if args.log_permit is True and e % 100 == 0:
            print('Epoch: {} Loss: {:.4f}'.format(e, epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = e

        if e-best_epoch > 30:
            if args.log_permit is True: print('Early Stopping!')
            train_flag = False

    return model

def get_args():
    class Args(object):
        pass
    args = Args()

    args.model = QNetwork()
    args.Qlr = 0.3
    args.lr = 0.001
    args.discount = 0.9
    args.epsilon = 0.8
    args.decay_rate = 0.95
    args.max_episodes = 100
    args.tol = 1e-4
    args.max_steps = 500
    args.final_path  = []
    args.train_set = None
    args.train_labels = None
    args.dataloader = None
    args.batch_size = 120
    args.num_epochs = 2000
    args.best_path_length = 1E8
    args.optimal_path_length = 0
    args.max_time = 10
    args.optimizer = torch.optim.Adam(args.model.parameters(),args.lr)
    args.criterion = torch.nn.MSELoss()
    args.log_permit = True
    args.animate_permit = False
    args.warm_up_permit = False
    args.persistent_flag = False

    args.loss_list = []
    return args

def get_samples_from_new_map():
    for _ in range(20):
        try:
            with ExpectTimeout(3):
                world = World.fixed_block(lower_bounds=(-2, -2, 0), upper_bounds=(3, 2, 2),
                               block_width=0.5, block_height=1.5,
                               num_blocks=4, robot_radii=0.25, margin=0.2)
                break
        except TimeoutError:
            pass

    resolution=(.25, .25, .25)
    margin=.2
    occ_map = OccupancyMap(world,resolution,margin)

    start = world.world['start']  # Start point, shape=(3,)
    goal = world.world['goal']  # Goal point, shape=(3,)
    my_path = graph_search(world, resolution, margin, start, goal, False)[1:-1]
    start = my_path[0]
    goal = my_path[-1]

    args = get_args()
    args.go = goal
    args.st = start
    args.world = world
    args.occ_map = occ_map
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
    for i in range(len(action_List)):
        args.optimal_path_length += np.linalg.norm(action_List[i])
    start_index = discretized_path[0]
    goal_index = discretized_path[-1]
    args.start = start_index
    args.goal = goal_index
    args.search_range = 20

    warm_up_set, warm_up_labels = get_all_warm_up(args,
                                                  discretized_path,
                                                  action_List)

    return args, warm_up_set, warm_up_labels

if __name__ == '__main__':
    args, warm_up_set, warm_up_labels = get_samples_from_new_map()
    args.dataloader = load_dataset(warm_up_set,warm_up_labels,batch_size=20)
    args.train_set = warm_up_set
    args.train_labels = warm_up_labels
    if args.warm_up_permit is True:
        train(args)
        
    reward_list, position_list, success_list = Qlearning(args)
    
    fig = plt.figure()
    ax = Axes3Ds(fig)
    ax.plot([args.st[0]], [args.st[1]], [args.st[2]], 'go', markersize=16,
        markeredgewidth=3, markerfacecolor='none')
    ax.plot([args.go[0]],  [args.go[1]],  [args.go[2]], 'r*', markersize=16,
        markeredgewidth=3, markerfacecolor='none')
    args.world.draw_points(ax, args.final_path, color='purple', markersize=8)
    args.world.draw(ax)
    args.occ_map.draw(ax)

    plt.show()

    x= np.arange(0,(args.max_episodes/5)+1)
    plt.plot(5*x,success_list)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Success Rate')
    plt.show()
    plt.close()

    plt.plot(args.loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.show()
    plt.close()
