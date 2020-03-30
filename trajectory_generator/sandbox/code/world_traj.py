import numpy as np

from proj1_3.code.graph_search import graph_search
from proj1_3.code.occupancy_map import OccupancyMap # Recommended.

class WorldTraj(object):
    # max_Velocity = 0.5
    # max_Acceleration = 3
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.3
        self.beta = 0.8

        self.alpha = 0.3
        max_Dist_Point = 3.8
        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=False)
        self.occ_map = OccupancyMap(world, self.resolution, self.margin)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points[0,:] = self.path[0] 
        current_point = self.points[0,:]
        # last_direction = np.zeros(3) 
        for i in range(len(self.path)-1): 
            if self.pathBlocked(current_point, self.path[i+1]) or np.linalg.norm(self.path[i+1]-current_point)>max_Dist_Point:
                self.points = np.append(self.points,[self.path[i]],axis=0)
                current_point = self.path[i]
        self.points = np.append(self.points,[self.path[-1]],axis=0)

        print(self.points)
        number_Unknowns = (self.points.shape[0]-1)*6
        boundry_Conditions = np.zeros((number_Unknowns,number_Unknowns))
        self.time_List = np.array([0])
        self.time_Interval = np.array([0])
        for i in range(1,len(self.points)):
            self.time_List = np.append(self.time_List,self.beta*np.exp(self.alpha*np.linalg.norm(self.points[i]-self.points[i-1]))+self.time_List[i-1])
            self.time_Interval = np.append(self.time_Interval,self.beta*np.exp(self.alpha*np.linalg.norm(self.points[i]-self.points[i-1])))
        print(self.time_Interval)
        print(len(self.time_Interval))
        print(len(self.points))
        boundry_Conditions = np.zeros((number_Unknowns,number_Unknowns))
        state = np.zeros((number_Unknowns,3))
        boundry_Conditions[0:3,0:6] = np.array([[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,2,0,0]])
        boundry_Conditions[-3:,-6:] = np.array([[self.time_Interval[-1]**5,self.time_Interval[-1]**4,self.time_Interval[-1]**3,self.time_Interval[-1]**2,self.time_Interval[-1],1],
                                                [5*self.time_Interval[-1]**4,4*self.time_Interval[-1]**3,3*self.time_Interval[-1]**2,2*self.time_Interval[-1],1,0],
                                                [20*self.time_Interval[-1]**3,12*self.time_Interval[-1]**2,6*self.time_Interval[-1],2,0,0]
                                                ])
        state[0:3,:] = np.array([self.points[0],[0,0,0],[0,0,0]])
        state[-3:,:] = np.array([self.points[-1],[0,0,0],[0,0,0]])
        

        for i in range(1, len(self.points)-1):
            boundry_Conditions[-3+6*i:3+6*i,-6+6*i:6*i+6] = np.array([[self.time_Interval[i]**5,self.time_Interval[i]**4,self.time_Interval[i]**3,self.time_Interval[i]**2,self.time_Interval[i],1,0,0,0,0,0,0],
                                                                      [0,0,0,0,0,0,0,0,0,0,0,1],
                                                                      [5*self.time_Interval[i]**4,4*self.time_Interval[i]**3,3*self.time_Interval[i]**2,2*self.time_Interval[i],1,0,0,0,0,0,-1,0],
                                                                      [20*self.time_Interval[i]**3,12*self.time_Interval[i]**2,6*self.time_Interval[i],2,0,0,0,0,0,-2,0,0],
                                                                      [60*self.time_Interval[i]**2,24*self.time_Interval[i],6,0,0,0,0,0,-6,0,0,0],
                                                                      [120*self.time_Interval[i],24,0,0,0,0,0,-24,0,0,0,0]
                                                                    ])
            state[-3+6*i:3+6*i,:] = np.array([self.points[i],self.points[i],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
        self.polynomials = np.linalg.inv(boundry_Conditions) @ state


        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0
        
        if t > self.time_List[-1]:
            x = self.path[-1]
            x_dot = np.zeros((3,))
            x_ddot = np.zeros((3,))
            yaw = 0
        elif t <= self.time_List[-1]:
            for i in range(len(self.time_List)-1):
                if t>= self.time_List[i] and t <= self.time_List[i+1]:
                    delT = t - self.time_List[i]
                    [x,x_dot,x_ddot,x_dddot,x_ddddot]= np.array([[delT**5,delT**4,delT**3,delT**2,delT**1,1],
                                                                [5*delT**4,4*delT**3,3*delT**2,2*delT,1,0],
                                                                [20*delT**3,12*delT**2,6*delT,2,0,0],
                                                                [60*delT**2,24*delT,6,0,0,0],
                                                                [120*delT,24,0,0,0,0]
                                                                ]) @ self.polynomials[6*i:6*i+6,:]
        # elif t <= self.time_List[-1]:
        #     for i in range(len(self.time_List)-1):
        #         if t>=self.time_List[i] and t<self.time_List[i+1]:
        #             if t > self.time_List[i+1]-self.hover_interval and t < self.time_List[i+1]:
        #                 x = self.point_List[i+1]
        #                 x_ddot = np.zeros((3,))
        #             elif t-self.time_List[i] <= self.time_List[i+1]-self.hover_interval-t:
        #                 x_ddot = self.direction_List[i] * self.max_Acceleration
        #                 x_dot = x_ddot * (t-self.time_List[i])
        #                 x = self.point_List[i] + 1/2* x_ddot*(t-self.time_List[i])**2
        #             elif t-self.time_List[i] > self.time_List[i+1]-t-self.hover_interval:
        #                 x_ddot = -self.direction_List[i] * self.max_Acceleration
        #                 x_dot = -x_ddot*(self.time_List[i+1]-self.hover_interval-t)
        #                 x = self.point_List[i+1] + 1/2* x_ddot*(self.time_List[i+1]-self.hover_interval-t)**2
        # STUDENT CODE HERE

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    
    def pathBlocked(self, start_Point, end_Point):
        diff = end_Point - start_Point
        largest_Dis = max(abs(diff))
        points = np.linspace(start_Point,end_Point,num = int(largest_Dis/min(self.resolution)))
        for point in points :
            if self.occ_map.is_occupied_metric(point):
                return True
        return False
