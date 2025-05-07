#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from classes import sim_monitor

def main(num_agents):
    
    # *** reminders because I keep making these mistakes *** 
    # *** 0 is wall, 1 is open space for any map interaction ***
    # *** Positions are [x,y], any interaction with gridworld, anomaly world, needs to be [y,x] ***
    
    map_name = "cumberland" # None or "cumberland" or "test"
    num_iterations = 500
    num_agents = num_agents
    monitor = sim_monitor.Monitor(map_name, num_agents)
    
    animate = False
    
    
    
    if animate:
        plt.ion()
        plt.imshow(monitor.world.gridworld, cmap='gray', vmin=0.0, vmax=1.0)
        
        plt.imshow(monitor.anomaly_class.anomaly_world, cmap='hot', norm=colors.LogNorm(vmin=0.01, vmax=monitor.anomaly_class.anomaly_world.max()), alpha=0.4)
        positions = np.array([agent.position for agent in monitor.agents])
        vels = np.array([agent.velocity for agent in monitor.agents])
        bests = np.unique(np.array([agent.global_best_position for agent in monitor.agents]))
        pos = plt.scatter(positions[:,0], positions[:,1], c='r', s=5)
        best = plt.scatter(bests[0], bests[1], c='g', s=5)
        plt.title(f"Step {1}/{num_iterations}")
        plt.draw()
    
    for i in range(num_iterations+1):            
        
       
       # print(f"Step {i} / {num_iterations+1}", end='\r')
        monitor.step_agents()
        bests = np.unique(np.array([agent.global_best_position for agent in monitor.agents]), axis=0)
        
        if animate:
            positions = np.array([agent.position for agent in monitor.agents])
            pos.set_offsets(positions)
            best.set_offsets(bests)
            #print(bests)
            plt.title(f"Step {i}/{num_iterations}")
            plt.draw()
            plt.pause(0.05)
        
    plt.ioff()
    plt.show()
    
    return bests
    
    
if __name__ == "__main__":
    #main(num_agents=6)
    for num_agents in range(6):
        num_agents+=1
        s = np.array([344, 249])
        results = [[999] for _ in range(100)]
        for i in range(100):
            print(f"Run {i} / 100", end='\r')
            bests = main(num_agents)
            if len(bests) > 1:
                continue
                #print("Didnt converge")
            else:
                #print(f"Best = {bests[0]}, Dist = {np.linalg.norm(bests[0]-s)}")
                results[i] = np.linalg.norm(bests[0]-s)
        
        results = [x for x in results if x != 999]
        succ = [x for x in results if x <= 5]
        print(f"{num_agents} Agents : Mean dist = {np.mean(results)} | std_dev = {np.std(results)} | succ = {len(succ)}%")# | Mode = {np.mode(results)}")