import numpy as np
import math
def random_block(lower_bounds,upper_bounds,block_width, block_height, num_trees,robot_radius):
    """
    Return World object describing a random forest block parameterized by
    arguments.

    Parameters:
        world_dims, a tuple of (xmax, ymax, zmax). xmin,ymin, and zmin are set to 0.
        tree_width, weight of square cross section trees
        tree_height, height of trees
        num_trees, number of trees

    Returns:
        world, World object
    """

    # Bounds are outer boundary for world, which are implicit obstacles.
    bounds = {'extents': [lower_bounds[0], upper_bounds[0], lower_bounds[1], upper_bounds[1], lower_bounds[2], upper_bounds[2]]}

    # Blocks are obstacles in the environment.
    w, h = block_width, block_height
    xs = np.random.uniform(lower_bounds[0]+w/2, upper_bounds[0]-w/2, num_trees)
    ys = np.random.uniform(lower_bounds[1]+w/2, upper_bounds[1]-w/2,num_trees)
    zs = np.random.uniform(lower_bounds[2]+h/2, upper_bounds[2]-h/2,num_trees)

    pts = np.stack((xs, ys,zs), axis=-1)  # min corner location of trees

    blocks = []
    i=0
    for pt in pts:
        if i==0:
            extents = list(np.round([pt[0]-w/2, pt[0] + w/2, pt[1]-w/2, pt[1] + w/2, pt[2]-h/2, pt[2]+h/2], 2))
            blocks.append({'extents': extents, 'color': [1, 0, 0]})
            i+=1
        else:   
            if np.linalg.norm(
                    np.array([blocks[-1]['extents'][0]+w/2,blocks[-1]['extents'][2]+w/2,blocks[-1]['extents'][4]+h/2])
                    -np.array([pt[0],pt[1],pt[2]])
            )>2*math.sqrt(h**2/4+2*w**2)+1.5*robot_radius:                   
                extents = list(np.round([pt[0]-w/2, pt[0] + w/2, pt[1]-w/2, pt[1] + w/2, pt[2]-h/2, pt[2]+h/2], 2))
                blocks.append({'extents': extents, 'color': [1, 0, 0]})
    #Start and goal position
    start= {'extents': [-1.5, -1.5, 0.9]}
    goal={'extents':[2.5, 1.5, 0.9]}
    world_data = {'bounds': bounds, 'blocks': blocks,'start':start,'goal':goal}
    return world_data
    
    
    
if __name__ == '__main__':
    random_block((-2,-2,0),(3,2,2),0.5,0.5,4,0.25)
    