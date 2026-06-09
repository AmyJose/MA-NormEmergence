# Norm Emergence in Multi-Agent Systems

A Python/Mesa implementation of a toy norm emergence environment inspired by research on social norms and resource sharing in multi-agent systems.

The project explores how cooperative behaviours can emerge in a society of autonomous agents competing for limited resources. Agents forage for berries, consume resources to survive, and may choose to share resources with other agents. Over time, the simulation tracks which behaviours become widely adopted and emerge as social norms.

## Current Features

* Mesa-based agent simulation
* Orthogonal Von Neumann grid environment
* Berry foraging and collection
* Health-based survival mechanics
* Resource consumption (eat action)
* Resource sharing (throw action)
* Dead agent handling
* Rule-based decision making
* Behaviour tracking
* Emergent norm detection
* Norm persistence tracking
* CSV export of emerged norms

## Environment

Agents exist on a 2D grid and must gather berries to maintain their health.

Each timestep agents may:

* Move towards the nearest berry
* Eat a berry they are carrying
* Throw a berry to another agent

Health decreases over time. Agents that reach zero health are considered dead and no longer participate in the simulation.

## Current Experimental Configuration

### Environment

| Parameter | Value |
|------------|---------|
| Grid Size | 8 x 4 |
| Initial Agents | 4 |
| Initial Berries | 8 |
| Initial Agent Health | 2.0 |
| Health Decay per Step | 0.03 |
| Health Gain per Berry Eaten | 0.2 |
| Throw Threshold | 0.6 |
| Maximum Episode Length | 200 steps |
| Agent Activation | Random asynchronous order |
| Berry Regrowth | One berry spawned after each berry is eaten |

### Agent Actions

At each timestep an agent may choose one of:

- Move North
- Move South
- Move East
- Move West
- Eat a berry
- Throw a berry to another agent

### Agent Observations

Each agent observes:

- Current health
- Number of berries carried
- Distance to nearest berry
- Society wellbeing

Society wellbeing is calculated as:

```text
wellbeing = (health + (berries × berry_health_payoff))
            / health_decay
```

### Health States

| State | Range |
|---------|---------|
| Low Health | < 0.8 |
| Medium Health | 0.8 ≤ health < 1.5 |
| High Health | ≥ 1.5 |

### Berry States

| State | Range |
|---------|---------|
| No Berries | 0 |
| Medium Berries | 1–2 |
| High Berries | ≥ 3 |

### Episode Termination

An episode ends when:

- All agents have died, or
- The maximum episode length is reached.

## Norm Representation

Behaviours are represented as:

```text
IF <agent state>
THEN <action>
```

For example:

```text
IF low health, medium berries
THEN eat
```

or

```text
IF high health, medium berries
THEN throw
```

A norm is considered to have emerged when a behaviour is adopted by a sufficient proportion of the agent population (90%).

## Current Implementation

The current version uses hand-crafted decision rules to establish a baseline environment and behaviour-tracking system.

This provides a platform for investigating:

* Norm emergence
* Norm persistence
* Resource inequality
* Cooperative behaviour
* Social welfare

before introducing learning algorithms.


## Running

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Run the simulation:

```bash
py code/run.py
```

Emergent norms are written to:

```text
emerged_norms.csv
```
