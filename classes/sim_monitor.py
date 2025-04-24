import numpy as np
from classes import sim_world, sim_agent

class Monitor:
    def __init__(self, map_name, num_agents):
        """
        :class: 'Monitor' is the main class that runs the simulation. It creates the world and the agents, and runs the simulation.
        """
        
        self.world = sim_world.World(map_name)
        self.num_agents = num_agents
        
        self.agents = self.generate_agents()
        
        
    def generate_agents(self):
        """
        :method: 'generate_agents' generates the agents in the world.
        """
        x,y = np.where(self.world.gridworld == 1)
        positions = [[x[i], y[i]] for i in np.random.randint(len(x), size=self.num_agents)]
        
        agents = [sim_agent.Agent(i, positions[i], self.world) for i in range(self.num_agents)]
        return agents
            
            
    def step_agents(self):
        """
        :method: 'step_agents' runs the simulation for one step.
        """
        
        for agent in self.agents:
            agent.agent_step()
            
