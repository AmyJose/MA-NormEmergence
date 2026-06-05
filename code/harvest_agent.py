from mesa.discrete_space import CellAgent

class HarvestAgent(CellAgent):
    """Agent in the model environment"""

    def __init__(self, id):
        super.__init__()

        self.id = id
        