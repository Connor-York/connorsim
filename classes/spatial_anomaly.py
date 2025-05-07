import numpy as np
from classes.algorithms import bresenhams_line, slice_from_coords

class RadioAnomaly: 
    def __init__(self, gridworld):
        generate_new = False
        self.gridworld = gridworld
        self.map_name = "cumberland"
        
        
        sy, sx = gridworld.shape
        self.source = (int(sx/2), int(sy/2)) #(600,100) # #middle of the map
        #print(f"SOR = {self.source}")
        self.anomaly_status = 1
        self.anomaly_strength = 1000
        self.wall_strength = 0.7 # how much to decrease signal by for each wall
        
        if generate_new:
            self.anomaly_world = self.generate_anomaly_world_walled()
            np.save(f"maps/{self.map_name}/central_multi.npy", self.anomaly_world)
        else:
            try:
                self.anomaly_world = np.load(f"maps/{self.map_name}/central_multi.npy")
            except FileNotFoundError:
                print(f"No file with map: {self.map_name}, source: {self.source}")
            
    def take_reading(self, position):
        """
        take a reading of anomaly world at given coordinates
        """
        x,y = position
        
        if self.anomaly_status == 0:
            return 0.0
        
        else:
            reading = self.anomaly_world[y, x] 
            return reading
            

    def generate_anomaly_world_walled(self):
        """
        generate a numpy array same size as world, where open spaces have a "strength"
        strength is 1/r^2 decay from a given point source, decreasing by 50% for each wall in the way
        walls calculated using raytracing to every point (expensive!) so save the file afterward.
        """
        x0, y0 = self.source
        x1, y1 = (221,316)
        x2, y2 = (88,205)
        
        
        radiating_field = np.zeros_like(self.gridworld)
        open_y, open_x = np.where(self.gridworld == 1)
        print("Generating anomaly diffusion: ")
        iters = len(open_y)
        for count, y in enumerate(open_y):
            print(f"\rLoading ... {count}/{iters}", end="")
            x = open_x[count]

            walls0 = len(np.where(slice_from_coords(self.gridworld, bresenhams_line((x0,y0),(x,y)))==0)[0])
            walls1 = len(np.where(slice_from_coords(self.gridworld, bresenhams_line((x1,y1),(x,y)))==0)[0])
            #distance0 = (np.sqrt((y0 - y) ** 2 + (x0 - x) ** 2))
            sq_norm0 = (y0-y)**2 + (x0-x)**2
            sq_norm1 = (y1-y)**2 + (x1-x)**2
            radiating_field[y,x] = (10 * np.exp(-sq_norm0/1000) * self.wall_strength**walls0) + (5 * np.exp(-sq_norm1/1000) * self.wall_strength**walls1)
            # if distance0 == 0:
            #     radiating_field[y, x] = self.anomaly_strength
            # else:
            #     radiating_field[y, x] =  (self.anomaly_strength / (distance0 ** 2)) + (self.wall_strength**walls0)
                
                
        return radiating_field
        

    
    def generate_anomaly_world(self):
        """
         generate a numpy array same size as world, where open spaces have a "strength"
         strength is 1/r^2 decay from a given point source, not effected by walls
        """
        radiating_field = np.zeros_like(self.gridworld)
        
        x0, y0 = self.source
        for y in range(self.gridworld.shape[0]):
            for x in range(self.gridworld.shape[1]):
                distance = (np.sqrt((y0 - y) ** 2 + (x0 - x) ** 2))               
                if distance != 0:
                    radiating_field[y, x] = self.anomaly_strength / (distance ** 2)
                else:
                    radiating_field[y, x] = self.anomaly_strength
        return radiating_field
        
        
        