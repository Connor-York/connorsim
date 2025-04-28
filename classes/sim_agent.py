import numpy as np
from classes import spatial_anomaly
from classes.algorithms import ordinal_walls


class Agent:
    """
    :class: 'Agent' represents the agent that will explore the world.
    """

    def __init__(self, ID, init_pos, world, anomaly_class):
        
        # agent parameters
        self.ID = ID
        self.position = init_pos
        self.top_speed = 5
        self.search_speed = self.top_speed
        
        # world variables
        self.world = world
        self.anomaly_class = anomaly_class
        
        # search variables
        self.current_reading = 0
        self.previous_reading = 0  
        self.directions = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)] # (x,y)
        """
        directions order, relative to centre (np array indexing)
        [[7. 0. 1.]
         [6. X  2.]
         [5. 4. 3.]]
        """
        self.previous_direction = np.random.randint(0,8)
        
    def e_coli_search(self):
        """
        E-coli inspired chemotaxis, take step, try step speed distance, but stop for walls
        """
        
        # if current reading is worse than previous, turn around, with 25% chance to go 45deg either way
        if self.current_reading < self.previous_reading:
            new_direction = self.previous_direction + 4
        else:
            new_direction = self.previous_direction
        
        r = np.random.rand()
        if 0.5 <= r < 0.75:
            new_direction = new_direction + 1
        elif 0.75 <= r < 1.0:
            new_direction = new_direction - 1
            
        new_direction = new_direction % 8
        new_coord = (self.position[0] + (self.search_speed * self.directions[new_direction][0]), self.position[1] + (self.search_speed * self.directions[new_direction][1]))

        wall_coord = ordinal_walls(self.world.gridworld, self.position, new_coord)
        if wall_coord is not False:
            new_coord = wall_coord
            self.previous_direction = new_direction
            self.position = new_coord # take step
        else:
            self.previous_direction = np.random.randint(0,8)
            # stay still i guess
            
        
        
    def agent_step(self, step_counter=None):
        
        self.current_reading = self.anomaly_class.take_reading(self.position)

        self.e_coli_search()
        self.previous_reading = self.current_reading
        
    