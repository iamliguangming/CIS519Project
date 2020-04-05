This random_map_generator is based on the existing world.py file. 

I created a new class function called random_block(), and it takes in parameters (lower_bounds,upper_bounds,block_width, block_height, num_blocks,robot_radius,margin) to create a random world map.

To test this function and visualize the random map, just update the world.py using this world.py and add display_random_map.py in the same directory as the sandbox.py£¬ then run display_random_map.py. 

This different world generator has applied into the sandbox.py. I haven't yet tested it on the actual flightsim but I will test this new sandbox very soon.