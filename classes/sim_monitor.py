import numpy as np
from classes import sim_world, sim_agent
from classes.spatial_anomaly import RadioAnomaly

class Monitor:
    def __init__(self, map_name, num_agents):
        """
        :class: 'Monitor' is the main class that runs the simulation. It creates the world and the agents, and runs the simulation.
        """
        
        self.world = sim_world.World(map_name)
        self.anomaly_class = RadioAnomaly(self.world.gridworld)
        self.num_agents = num_agents
        
        self.step_counter = 0
        self.agents = self.generate_agents()
        
        
        
    def generate_agents(self):
        """
        :method: 'generate_agents' generates the agents in the world.
        """
        y,x = np.where(self.world.gridworld == 1)
        positions = [[x[i], y[i]] for i in np.random.randint(len(x), size=self.num_agents)]
        
        agents = [sim_agent.Agent(i, positions[i], self.world, self.anomaly_class) for i in range(self.num_agents)]
        
        agent_vals = np.array([agent.personal_best_value for agent in agents])
        initial_best_position = agents[np.argmax(agent_vals)].position
        initial_best_value = agents[np.argmax(agent_vals)].personal_best_value
        
        for agent in agents:
            agent.global_best_position = initial_best_position
            agent.global_best_value = initial_best_value
        
        return agents
            
    def handle_messages(self):
        messages = [[agent.personal_best_position, agent.personal_best_value, agent.position, agent.ID] for agent in self.agents]

        return messages
            
    def step_agents(self):
        """
        :method: 'step_agents' runs the simulation for one step.
        """
        self.step_counter += 1
        
        messages = self.handle_messages()
        
        for agent in self.agents:
            agent.agent_step(messages, self.step_counter)
            
