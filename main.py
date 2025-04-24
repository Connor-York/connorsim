#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from classes import sim_monitor




def main():
    
    map_name = None #"cumberland"
    num_iterations = 100
    num_agents = 6
    monitor = sim_monitor.Monitor(map_name, num_agents)
    
    animate = True
    
    if animate:
        plt.ion()
        background = plt.imshow(monitor.world.gridworld, cmap='gray', vmin=0.0, vmax=1.0)
        positions = np.array([agent.position for agent in monitor.agents])
        sc = plt.scatter(positions[:,0], positions[:,1], c='red', s=50)
        plt.title(f"Step {1}/{num_iterations}")
        plt.draw()
    
    for i in range(num_iterations+1):
        
        monitor.step_agents()
        
        if animate:
            positions = np.array([agent.position for agent in monitor.agents])
            sc.set_offsets(positions)
            plt.title(f"Step {i}/{num_iterations}")
            plt.draw()
            plt.pause(0.05)
        

    
    
if __name__ == "__main__":
    main()