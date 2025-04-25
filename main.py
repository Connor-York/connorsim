#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from classes import sim_monitor


def main():
    
    # *** 0 is wall, 1 is open space for any map interaction ***
    
    map_name = "cumberland" # None or "cumberland"
    num_iterations = 100
    num_agents = 6
    monitor = sim_monitor.Monitor(map_name, num_agents)
    
    print(f"Gridworld shape: {monitor.world.gridworld.shape}")
    print(f"Anomalyworld shape: {monitor.anomaly_class.anomaly_world.shape}")
    
    
    animate = True
    
    if animate:
        plt.ion()
        plt.imshow(monitor.world.gridworld, cmap='gray', vmin=0.0, vmax=1.0)
        
        plt.imshow(monitor.anomaly_class.anomaly_world, cmap='hot', norm=colors.LogNorm(vmin=0.01, vmax=monitor.anomaly_class.anomaly_world.max()), alpha=0.4)
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
        
    plt.ioff()
    plt.show()
    
    
if __name__ == "__main__":
    main()