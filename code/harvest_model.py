import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from harvest_agent import HarvestAgent
import pandas as pd
import os
from pathlib import Path

class HarvestModel(mesa.Model):
    """Harvest environemnt for resource sharing"""
    def __init__(self, seed, num_agents=4, num_berries=12, width=8, height=4):
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
        self.max_steps = 500
        self.episode_done = False

        self.episode = 1
        self._init_reporters()
    
    def step(self):
        if self.episode_done:
            return
        
        self.agents.shuffle_do("step")

        emerged_norms = self.check_emergent_norms()
        self.update_emerged_norms(emerged_norms)

        if self.steps >= self.max_steps or all(agent.dead for agent in self.harvest_agents):
            self._collect_model_episode_data()
            self.write_emerged_norms()
            self.episode_done = True
        
        self._collect_agent_data()

    def spawn_berries(self):
        self.berries.clear()

        for _ in range(self.num_berries):
            berry_cell = self.grid.all_cells.select_random_cell()
            self.berries.add(berry_cell)

    def spawn_one_berry(self):
        berry_cell = self.grid.all_cells.select_random_cell()
        self.berries.add(berry_cell)

    def get_agent_by_id(self, agent_id):
        for agent in self.harvest_agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_society_wellbeing(self):
        return [
            agent.get_wellbeing()
            for agent in self.harvest_agents
            if not agent.dead
        ]

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

    def write_emerged_norms(self, filename="emerged_norms.csv"):
        rows = []

        for behaviour, stats in self.emerged_norms.items():

            rows.append({
                "behaviour": behaviour,
                "first_seen": stats["first_seen"],
                "last_seen": stats["last_seen"],
                "times_emerged": stats["times_emerged"],
                "max_adoption": stats["max_adoption"],
            })

        pd.DataFrame(rows).to_csv(filename, index=False)

    def _init_reporters(self, filepath="current_run"):
        os.makedirs("data/results/current_run", exist_ok=True)

        self.filepath = filepath

        self.agent_reporter = pd.DataFrame({
            "agent_id": [],
            "episode": [],
            "step": [],
            "berries": [],
            "berries_consumed": [],
            "berries_thrown": [],
            "health": [],
            "wellbeing": [],
            "action": [],
            "dead": [],
            "num_behaviours": [],
        })

        self.model_episode_reporter = pd.DataFrame({
            "episode": [],
            "end_step": [],
            "max_berries": [],
            "mean_berries": [],
            "max_berries_consumed": [],
            "mean_berries_consumed": [],
            "gini_berries_consumed": [],
            "mean_berries_thrown": [],
            "max_health": [],
            "mean_health": [],
            "median_health": [],
            "variance_health": [],
            "deceased": [],
            "num_emerged_norms": [],
        })

        self.agent_report_path = Path(
            f"data/results/current_run/agent_reports_{self.filepath}.csv"
        )

        self.model_episode_report_path = Path(
            f"data/results/current_run/model_episode_reports_{self.filepath}.csv"
        )

        self.agent_reporter.to_csv(self.agent_report_path, index=False)
        self.model_episode_reporter.to_csv(self.model_episode_report_path, index=False)

    def _collect_agent_data(self):
        rows = []

        for agent in self.harvest_agents:
            rows.append({
                "agent_id": agent.id,
                "episode": self.episode,
                "step": self.steps,
                "berries": agent.berries,
                "berries_consumed": agent.berries_consumed,
                "berries_thrown": agent.berries_thrown,
                "health": agent.health,
                "wellbeing": agent.get_wellbeing(),
                "action": agent.current_action,
                "dead": agent.dead,
                "num_behaviours": len(agent.norms_module.behaviour_base),
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.agent_report_path, mode="a", header=False, index=False)

    def _collect_model_episode_data(self):
        berries = [agent.berries for agent in self.harvest_agents]
        berries_consumed = [agent.berries_consumed for agent in self.harvest_agents]
        berries_thrown = [agent.berries_thrown for agent in self.harvest_agents]
        health = [agent.health for agent in self.harvest_agents]

        row = {
            "episode": self.episode,
            "end_step": self.steps,
            "max_berries": max(berries),
            "mean_berries": sum(berries) / len(berries),
            "max_berries_consumed": max(berries_consumed),
            "mean_berries_consumed": sum(berries_consumed) / len(berries_consumed),
            "gini_berries_consumed": self._gini(berries_consumed),
            "mean_berries_thrown": sum(berries_thrown) / len(berries_thrown),
            "max_health": max(health),
            "mean_health": sum(health) / len(health),
            "median_health": pd.Series(health).median(),
            "variance_health": pd.Series(health).var(),
            "deceased": sum(agent.dead for agent in self.harvest_agents),
            "num_emerged_norms": len(self.emerged_norms),
        }

        pd.DataFrame([row]).to_csv(
            self.model_episode_report_path,
            mode="a",
            header=False,
            index=False
        )

    def _gini(self, values):
        values = sorted(values)
        total = sum(values)

        if total == 0:
            return 0

        n = len(values)
        weighted_sum = sum((i + 1) * value for i, value in enumerate(values))

        return (2 * weighted_sum) / (n * total) - (n + 1) / n