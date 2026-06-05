import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from harvest_agent import HarvestAgent

class HarvestModel(mesa.Model):
    """Harvest environemnt for resource sharing"""
    def __init__(self, seed, num_agents=4, num_berries=15, width=10, height=10):
        super().__init__(seed=seed)

        self.grid = OrthogonalVonNeumannGrid(
            [width, height],
            torus=False,
            random=self.random
        )

        self.num_agents = num_agents
        self.num_berries = num_berries
        self.berries = set()

        self.spawn_berries()

        self.agents = []
        for i in range(self.num_agents):
            agent = HarvestAgent(self, i)

            cell = self.random.choice(list(self.grid.all_cells.cells))
            self.grid.place_agent(agent, cell)

            self.agents.append(agent)
        
    
    def step(self):
        self.agents.shuffle_do("step")

    def spawn_berries(self):
        self.berries.clear()

        for _ in range(self.num_berries):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            self.berries.add((x, y))

    def get_berry_from_coord(self, cell):
        return self.berries.__contains__(cell)
