import numpy as np


class Agent:
    """
    :class: 'Agent' represents the agent that will explore the world.
    """

    def __init__(self, ID, init_pos, world):
        
        self.ID = ID
        self.position = init_pos
        self.world = world
        self.speed = 5
        self.neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
    def randomWalk(self):
        """
        :method: 'randomWalk' moves the agent in a random direction.
        """
        
        # get random direction
        direction = np.random.choice(range(8))
        
        # get new position
        # new_pos = (self.position[0] + (self.speed*self.neighbours[direction][0]), self.position[1] + (self.speed*self.neighbours[direction][1]))
        new_x = self.position[0] + (self.speed*self.neighbours[direction][0])
        new_y = self.position[1] + (self.speed*self.neighbours[direction][1])
        
        # check if new position is valid
        if 0 <= new_x <= self.world.gridworld.shape[1] and 0 <= new_y <= self.world.gridworld.shape[0] and self.world.gridworld[new_y, new_x]:
            self.position = [new_x, new_y]
        # else dont move
        
    def agent_step(self, step_counter=None):
        
        self.randomWalk()
        #pass
        
        
        