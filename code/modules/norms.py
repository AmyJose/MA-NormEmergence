class NormsModule:
    def __init__(self, agent):
        self.agent = agent
        self.behaviour_base = {}

        self.low_berries_threshold = 1
        self.high_berries_threshold = 3
        self.low_health_threshold = 2.5
        self.high_health_threshold = 4.0
        self.low_wellbeing_threshold = 250
        self.high_wellbeing_threshold = 400

    def get_pre(self, observation):
        berries = observation["berries"]
        health = observation["health"]
        society_wellbeing = observation["society_wellbeing"]
        
        b = self._bucket_berries(berries)
        h = self._bucket_health(health)
        w = self._bucket_min_wellbeing(society_wellbeing)

        return ",".join(["IF", b, h, w])
    
    def get_cons(self, action):
        if action in ["north", "east", "south", "west"]:
            action = "move"
        elif action.startswith("throw_"):
            action = "throw"
        return ",".join(["THEN", action])

    def update_behaviour_base(self, pre, action):
        cons = self.get_cons(action)
        current_norm = ",".join([pre, cons])
        #see if the norm already exists
        norm = self.behaviour_base.get(current_norm)
        if norm != None:
            norm["count"] += 1
        else:
            self.behaviour_base[current_norm] = {"count": 1}

    def _bucket_berries(self, berries):
        if berries == 0:
            return "no berries"
        elif berries < self.low_berries_threshold:
            return "low berries"
        elif berries < self.high_berries_threshold:
            return "medium berries"
        else:
            return "high berries"
    
    def _bucket_health(self, health):
        if health < self.low_health_threshold:
            return "low health"
        elif health < self.high_health_threshold:
            return "medium health"
        else:
            return "high health"
        
    def _bucket_min_wellbeing(self, society_wellbeing):
        if not society_wellbeing:
            return "no others"

        min_wellbeing = min(society_wellbeing)

        if min_wellbeing < self.low_wellbeing_threshold:
            return "low society wellbeing"
        elif min_wellbeing < self.high_wellbeing_threshold:
            return "medium society wellbeing"
        else:
            return "high society wellbeing"
        