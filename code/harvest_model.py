import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from harvest_agent import HarvestAgent

class HarvestModel(mesa.Model):
    """Harvest environemnt for resource sharing"""
    def __init__(self, seed, num_agents=4, num_berries=15, width=10, height=10):
        super().__init__(seed=seed)
        self.width = width
        self.height = height

        self.grid = OrthogonalVonNeumannGrid(
            [self.width, self.height],
            torus=False,
            random=self.random
        )

        self.num_agents = num_agents
        self.num_berries = num_berries
        self.berries = set()

        self.spawn_berries()

        self.harvest_agents = []
        for i in range(self.num_agents):
            agent = HarvestAgent(self, i)

            cell = self.random.choice(list(self.grid.all_cells.cells))
            agent.move_to(cell)

            self.harvest_agents.append(agent)
        
        self.emerged_norms = {}
    
    def step(self):
        self.agents.shuffle_do("step")

        emerged_norms = self.check_emergent_norms()

        self.update_emerged_norms(emerged_norms)

    def spawn_berries(self):
        self.berries.clear()

        for _ in range(self.num_berries):
            berry_cell = self.grid.all_cells.select_random_cell()
            self.berries.add(berry_cell)

    def check_emergent_norms(self, threshold=0.9, min_uses=1):
        all_behaviours = set()

        for agent in self.harvest_agents:
            all_behaviours.update(agent.norms_module.behaviour_base.keys())

        emerged_norms = []

        for behaviour in all_behaviours:
            adopters = sum(
                1
                for agent in self.harvest_agents
                if agent.norms_module.behaviour_base.get(behaviour, {}).get("count", 0) >= min_uses
            )

            adoption_rate = adopters / len(self.harvest_agents)

            if adoption_rate >= threshold:
                emerged_norms.append((behaviour, adoption_rate))

        return emerged_norms
    
    def update_emerged_norms(self, emerged_norms):
        for behaviour, adoption_rate in emerged_norms:
            #if first time were seeing this norm:
            if behaviour not in self.emerged_norms:
                self.emerged_norms[behaviour] = {
                    "first_seen": self.steps,
                    "last_seen": self.steps,
                    "times_emerged": 1,
                    "max_adoption": adoption_rate,
                    "history":[
                        {
                            "step": self.steps,
                            "adoption_rate": adoption_rate
                        }
                    ]
                }
            # we've seen the norm before
            else:
                norm = self.emerged_norms[behaviour]
                norm["last_seen"] = self.steps
                norm["times_emerged"] += 1

                norm["max_adoption"] = max(
                    norm["max_adoption"],
                    adoption_rate
                )
                norm["history"].append(
                    {
                        "step" : self.steps,
                        "adoption_rate": adoption_rate
                    }
                )
