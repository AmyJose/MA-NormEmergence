import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid

class HarvestModel(mesa.Model):
    """Harvest environemnt for resource sharing"""
    def __init__(self, num_agents, seed, width, height):
        super().__init__(seed=seed)

        self.num_agents = num_agents

        self.grid = OrthogonalVonNeumannGrid(
            [width, height],
            torus=False,
            random=self.random
        )
    
    def step(self):
        self.agents.shuffle_do("step")