import numpy as np
from classes.algorithms import bresenhams_line, slice_from_coords

class RadioAnomaly: 
    def __init__(self, gridworld):
        generate_new = False
        self.gridworld = gridworld
        self.map_name = "cumberland"
        
        
        sy, sx = gridworld.shape
        self.source =(int(sx/2), int(sy/2)) #(600,100) # #middle of the map
        self.anomaly_status = 1
        self.anomaly_strength = 1000
        self.wall_strength = 0.5 # how much to decrease signal by for each wall
        
        if generate_new:
            self.anomaly_world = self.generate_anomaly_world_walled()
            np.save(f"maps/{self.map_name}/{self.source[0]}_{self.source[1]}_{int(100*self.wall_strength)}.npy", self.anomaly_world)
        else:
            try:
                self.anomaly_world = np.load(f"maps/{self.map_name}/{self.source[0]}_{self.source[1]}_{int(100*self.wall_strength)}.npy")
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
        radiating_field = np.zeros_like(self.gridworld)
        open_y, open_x = np.where(self.gridworld == 1)
        print("Generating anomaly diffusion: ")
        iters = len(open_y)
        for count, y in enumerate(open_y):
            print(f"\rLoading ... {count}/{iters}", end="")
            x = open_x[count]
            coords = bresenhams_line((x0,y0),(x,y))
            view = slice_from_coords(self.gridworld, coords)
            walls = len(np.where(view==0)[0])
            distance = (np.sqrt((y0 - y) ** 2 + (x0 - x) ** 2))
            if distance != 0:
                radiating_field[y, x] = ( self.anomaly_strength / (distance ** 2) ) * (self.wall_strength**walls)
            else:
                radiating_field[y, x] = self.anomaly_strength
                
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
        
        
        