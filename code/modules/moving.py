from collections import deque

class MovingModule():
    def __init__(self, agent):
        self.agent = agent
        self.path = []
        self.target_berry_cell = None

    def reset(self):
        self.path = []
        self.target_berry_cell = None

    def move_towards_nearest_berry(self):
        current_cell = self.agent.cell

        #if standing on a berry, collect it
        if current_cell in self.agent.model.berries:
            self.agent.model.berries.remove(current_cell)
            self.reset()
            return True, current_cell
        
        #if no current path or target berry has disappeared, recalc
        if not self.path or self.target_berry_cell not in self.agent.model.berries:
            self.target_berry_cell = self._find_nearest_berry(current_cell)

            if self.target_berry_cell is None:
                return False, current_cell
            
            self.path = self._find_path(current_cell, self.target_berry_cell)

        #if pathfinding failed or already at the target
        if not self.path:
            return False, current_cell
        
        next_cell = self.path.pop(0)
        return False, next_cell
    
    def distance_to_berry(self):
        if not self.path:
            return 0
        return len(self.path)
    
    def _find_nearest_berry(self, current_cell):
        if not self.agent.model.berries:
            return None
        return min(
            self.agent.model.berries,
            key=lambda berry_cell: self._manhattan_distance(
                current_cell.coordinate, berry_cell.coordinate)
        )
    
    def _find_path(self, start, goal):
        if start == goal:
            return []
        
        queue = deque([start])
        came_from = {start : None}
        
        while queue:
            current = queue.popleft()

            if current == goal:
                break

            for neighbour in current.neighborhood:
                if neighbour not in came_from:
                    came_from[neighbour] = current
                    queue.append(neighbour)
            
        if goal not in came_from:
            return []
        
        return self._reconstruct_path(start, goal, came_from)

    def _reconstruct_path(self, start, goal, came_from):
        path = []
        current = goal

        while current != start:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path
    
    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        