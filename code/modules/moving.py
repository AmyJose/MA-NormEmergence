class MovingModule:
    def __init__(self, agent):
        self.agent = agent

    def direction_towards_nearest_berry(self):
        current_cell = self.agent.cell

        if not self.agent.model.berries:
            return self.agent.random.choice(["north", "south", "east", "west"])

        target_cell = min(
            self.agent.model.berries,
            key=lambda berry_cell: self._manhattan_distance(
                current_cell.coordinate,
                berry_cell.coordinate
            )
        )

        neighbours = self._neighbour_direction_map()

        best_direction = min(
            neighbours.keys(),
            key=lambda direction: self._manhattan_distance(
                neighbours[direction].coordinate,
                target_cell.coordinate
            )
        )

        return best_direction

    def move(self, direction):
        neighbours = self._neighbour_direction_map()

        if direction not in neighbours:
            return False

        self.agent.move_to(neighbours[direction])
        return True

    def distance_to_berry(self):
        current_cell = self.agent.cell

        if not self.agent.model.berries:
            return 0

        nearest_berry = min(
            self.agent.model.berries,
            key=lambda berry_cell: self._manhattan_distance(
                current_cell.coordinate,
                berry_cell.coordinate
            )
        )

        return self._manhattan_distance(
            current_cell.coordinate,
            nearest_berry.coordinate
        )

    def _neighbour_direction_map(self):
        current_x, current_y = self.agent.cell.coordinate

        direction_map = {}

        for neighbour in self.agent.cell.neighborhood:
            x, y = neighbour.coordinate

            if x == current_x and y == current_y + 1:
                direction_map["north"] = neighbour
            elif x == current_x and y == current_y - 1:
                direction_map["south"] = neighbour
            elif x == current_x + 1 and y == current_y:
                direction_map["east"] = neighbour
            elif x == current_x - 1 and y == current_y:
                direction_map["west"] = neighbour

        return direction_map

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])