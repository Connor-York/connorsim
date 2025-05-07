import numpy as np
from classes import spatial_anomaly
from classes.algorithms import ordinal_walls, bresenhams_line, slice_from_coords


class Agent:
    """
    :class: 'Agent' represents the agent that will explore the world.
    """

    def __init__(self, ID, init_pos, world, anomaly_class):
        
        # agent parameters
        self.ID = ID
        self.position = np.array(init_pos)
        self.top_speed = 5
        self.search_speed = self.top_speed
        
        # world variables
        self.world = world
        self.anomaly_class = anomaly_class
        

        # search variables
        self.current_reading = anomaly_class.take_reading(self.position)
        
        self.previous_reading = 0  
        self.directions = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)] # (x,y)
        """
        directions order, relative to centre (np array indexing)
        [[7. 0. 1.]
         [6. X  2.]
         [5. 4. 3.]]
        """
        self.previous_direction = np.random.randint(0,8)
        
        self.personal_best_position = self.position
        self.personal_best_value = self.current_reading
        self.global_best_position = self.position
        self.global_best_value = 0
        self.isbest = False
        
        #PSO variables
        self.agent_positions = None #Agent positions from communication 
        self.repulsion_strength = 10.0
        self.velocity = np.random.randn(2) * 1.0 
        self.c1 = 0.1 # weighting for pbest
        self.c2 = 0.1 # weighting for gbest
        self.c3 = 1.0 # weighting for repulsion from other agents
        self.w = 1.0 # inertia (< 1.0 slows down over time )
        
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
            
        self.previous_reading = self.current_reading
    
    def PSO_step(self):
        # TODO some reason either new position or best position or smth idfk is getting flipped and not plotting properly?
        r1, r2 = np.random.rand(2)
        
        repulsion_vector = self.calculate_repulsion()
        self.velocity = self.w * self.velocity + (self.c1 * r1 * (self.personal_best_position-self.position)) + (self.c2 * r2 * (self.global_best_position-self.position)) + repulsion_vector
        #self.velocity = self.velocity.clip(-self.top_speed,self.top_speed).astype(int)
        dist = np.linalg.norm(self.velocity)
        if dist > self.top_speed:
            self.velocity = self.velocity * (self.top_speed / dist)
        self.velocity = self.velocity.astype(int)
        new_position = self.position + self.velocity
        
        # wall_coords = bresenhams_line(self.position, new_position)
        # print(f"pos: {self.position}, goal: {new_position}, coords: {wall_coords}")
        # for count,coord in enumerate(wall_coords):
        #     if self.world.gridworld(coord[1],coord[0]) == 0:
        #         new_position == wall_coords[count-1]
        #         break
        
        
        
        self.position = new_position
        
    def calculate_repulsion(self):
        forces = np.zeros_like(self.agent_positions, dtype=float)

        for count,agent_pos in enumerate(self.agent_positions):
            if (agent_pos == self.position).all():
                continue
            
            direction = agent_pos - self.position
            distance = np.linalg.norm(direction)
            unit_direction = direction / distance
            if distance == 0:
                magnitude = self.repulsion_strength 
            else:
                magnitude = self.repulsion_strength / distance**2
            
            agent_repulsion = unit_direction * magnitude
            forces[count] = agent_repulsion
        
        repulsion_vector = np.sum(forces, axis=0)
        return repulsion_vector
            
            
    
    
    def read_messages(self, messages):
        # assuming message is: [ [[x,y],z], [[x,y],z], ...] where x,y is position and z is value, also assuming your own message is in here
        
        val_pos, value, best_pos, best_ID = messages[np.argmax(np.array([val[1] for val in messages]))]
        self.agent_positions = np.array([val[2] for val in messages])
        
        if value > self.global_best_value:
            self.global_best_value = value
            self.global_best_position = val_pos
        
        # print(f"A{self.ID} | B_Pos{val_pos} | Pos{self.position}")
        # if (val_pos == self.position).all():
        #     self.isbest = True
        # else:
        #     self.isbest = False
        
        
        
    def agent_step(self, messages=None, step_counter=None):
        
        #print(f"A{self.ID} - Vel: {self.velocity} - Pos:{self.position} - PB:{self.personal_best_position} - GB:{self.global_best_position} - IsBest?:{self.isbest}")
        
        self.current_reading = self.anomaly_class.take_reading(self.position)
        
        if self.current_reading > self.personal_best_value: #update personal best value
            self.personal_best_value = self.current_reading
            self.personal_best_position = self.position
            
        if messages is not None:                            #update global best value
            self.read_messages(messages)
            
        
        self.PSO_step()
        
        
    