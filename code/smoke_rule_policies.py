from types import SimpleNamespace

from modules.decisions import RuleBasedDecisionModule


class FakeAgent:
    def __init__(self, agent_id, wellbeing):
        self.id = agent_id
        self.wellbeing = wellbeing
        self.dead = False
        self.throw_berry_threshold = 0.3
        self.moving_module = SimpleNamespace(
            direction_towards_nearest_berry=lambda: "north"
        )

    def get_wellbeing(self):
        return self.wellbeing


actor = FakeAgent(0, 20)
other = FakeAgent(1, 10)
actor.model = SimpleNamespace(harvest_agents=[actor, other])

observation = {"berries": 1, "health": 1.0}

self_interested = RuleBasedDecisionModule(actor, "self_interested")
cooperative = RuleBasedDecisionModule(actor, "cooperative")
altruistic = RuleBasedDecisionModule(actor, "altruistic")

assert self_interested.decide(observation) == "eat"
assert cooperative.decide(observation) == "throw_1"
assert altruistic.decide(observation) == "throw_1"

# Equal wellbeing: cooperative does not share; altruistic does.
other.wellbeing = 20
assert cooperative.decide(observation) == "north"
assert altruistic.decide(observation) == "throw_1"

# Actor is worse off: cooperative does not share; altruistic does.
other.wellbeing = 30
assert cooperative.decide(observation) == "north"
assert altruistic.decide(observation) == "throw_1"

# Below the throw threshold: both eat.
low_health = {"berries": 1, "health": 0.2}
assert cooperative.decide(low_health) == "eat"
assert altruistic.decide(low_health) == "eat"

# No berries: all move.
empty_bag = {"berries": 0, "health": 1.0}
for policy in (self_interested, cooperative, altruistic):
    assert policy.decide(empty_bag) == "north"

print("Rule-policy checks passed")