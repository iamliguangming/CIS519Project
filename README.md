# CIS519Project
CIS519 Spring 2020 Project
Group Member: Yupeng Li, Jiatong Sun, Junfan Pan

As MEAM students, we have been working on Advanced Robotics with topics related to quadrotor flight planning. In a typical situation, we would be given the entire map of the environment with positions of obstacles, a starting point and an end point. Based on the information given, we need to generate a path which avoids all the obstacles with the shortest path length. This is usually done by applying Dijkstra, A* and RRT algorithms. Although these path finding techniques are fast and optimal, they only apply to situations where the map is already given.
We want to incorporate our knowledge in applied machine learning to help generalize this path planning idea. This means, we want to put our quadrotor into any random new and unknown environment and ask it to generate a considerably optimal path to the destination based on its sensors with limited perception of surrounding. Combining this unknown environment navigation technique and good attitude & position control on the quadrotor, we would be able to navigate our quadrotor through extreme environments including explorations and rescues.

What data to use?
This project will be based on sandbox simulation data. We already have python code to generate random 3D maps with obstacles. We want to first use A* algorithm to generate the optimal path in space based on the map generated. Then we record the flat outputs of each point including the position, velocity, higher derivatives of velocity, heading and surrounding obstacles in sensor range. We will use these flat outputs as our features to train our model.

What classifier to use?
Inspired by the work done by Vikranth Dwaracherla’s team from Stanford University , we will be using Q-learning approach which approximate the optimal decision function using neural networks. For each point, we will have the sensor data from simulation and other measurements as input features and the action taken under that circumstance as our labels. The reward mechanism for each step taken will be determined later in this project.

Minimum Goal
The minimum expectation for this project is to develop a model that is able to produce a feasible and comparably optimal path with only limited knowledge of surroundings under any random circumstances. By saying path planning, it does not take kinematics and kinetics into account, it is just the optimal path we can achieve supposing we have an absolutely perfect controller.

Higher Expectation
If we have achieved the minimum goal earlier than expected, we are going to pick up the kinematics part of the quadrotor. The path will then evolve to a fully constrained trajectory which requires the quadrotor to fly according to physics law. By doing that, we need to take in some extra kinematics features into our consideration.
