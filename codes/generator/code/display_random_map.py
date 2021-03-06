import inspect
import matplotlib.pyplot as plt
from pathlib import Path
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World 
from flightsim.world import ExpectTimeout 
import numpy as np


quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

fig = plt.figure('A* Path, Waypoints, and Trajectory')
ax = Axes3Ds(fig)
for _ in range(20):
    try:
        with ExpectTimeout(3):
            world = World.fixed_block(lower_bounds=(-2,-2,0),upper_bounds=(3,2,2),block_width=0.5,block_height=1.5,num_blocks=4,robot_radii=robot_radius,margin=0.2)
            break
    except TimeoutError:
        pass
    
world.draw(ax)
start=world.world['start']
goal=world.world['goal']
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=5, markeredgewidth=3, markerfacecolor='none')
ax.plot([goal[0]], [goal[1]], [goal[2]], 'ro', markersize=5, markeredgewidth=3, markerfacecolor='none')
plt.show()





