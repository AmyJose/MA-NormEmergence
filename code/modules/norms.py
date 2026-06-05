from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class Behaviour:
    pre : tuple
    cons : str

class NormsModule:
    def __init__(self, id):
        self.agent_id = id
        self.behaviour_base = {}

        self.low_berries_threshold = 1
        self.high_berries_threshold = 3
        self.low_health_threshold = 0.6
        self.high_health_threshold = 2.0

    def get_pre(self, berries, health):
        #get state of berries
        if berries == 0:
            b = "no berries"
        elif berries > 0 and berries < self.low_berries_threshold:
            b = "low berries"
        elif berries >= self.low_berries_threshold and berries < self.high_berries_threshold:
            b = "medium berries"
        else:
            "high berries"
        
        #state of health
        if health < self.low_health_threshold:
            h = "low health"
        elif health >= self.low_health_threshold and health < self.high_health_threshold:
            h = "medium health"
        else:
            h = "high health"
        
        pre = ",".join(["IF", b, h])

        return pre
    
    def get_cons(self, action):
        cons = "THEN, "
        if action == "north" or action == "east" or action == "south" or action == "west":
            return cons + "move"
        elif "throw" in action:
            return cons + "throw"
        else:
            return cons + action

    def update_behaviour_base(self, pre, action):
        cons = self.get_cons(action)
        current_norm = ",".join([pre, cons])
        #see if the norm already exists
        norm = self.behaviour_base.get(current_norm)
        if norm != None:
            norm["count"] += 1
        else:
            self.behaviour_base[current_norm] = {"count": 1}

        print(self.behaviour_base)