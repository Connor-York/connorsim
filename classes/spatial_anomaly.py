import numpy as np

class RadioAnomaly: 
    def __init__(self, gridworld):
        self.gridworld = gridworld
        
        sy, sx = gridworld.shape
        self.source = (int(sx/2), int(sy/2)) #middle of the map
        self.anomaly_strength = 1000
        
        self.anomaly_world = self.generate_anomaly_world()
        
    
    def generate_anomaly_world(self):
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
        
        
        