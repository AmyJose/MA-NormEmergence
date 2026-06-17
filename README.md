# Norm Emergence in Multi-Agent Systems

A Python/Mesa implementation of a norm emergence environment for studying cooperation, resource sharing, and social welfare in multi-agent systems.

The project investigates how behavioural norms emerge from repeated interactions between autonomous agents competing for limited resources. Agents forage for food, consume resources to maintain their health, and may choose to share resources with others. Individual behaviours are tracked over time and analysed to identify emergent norms and patterns of social coordination.

The environment is designed as an experimental platform for comparing different decision-making approaches, including rule-based policies and large language model (LLM) agents.

## Current Features
* Mesa-based agent simulation
* Orthogonal Von Neumann grid environment
* Resource foraging and collection
* Health-based survival mechanics
* Resource consumption (eat)
* Resource sharing (throw)
* Agent mortality
* Rule-based decision making
* LLM-based decision making (experimental)
* Behaviour tracking and behaviour bases
* Emergent norm detection
* Norm persistence tracking
* Per-agent behavioural analysis
* CSV export of simulation results

## Environment

Agents inhabit a two-dimensional grid populated with berry resources. Berries act as the primary survival resource and must be collected and consumed to maintain health.

At each timestep an agent may:

* Move towards the nearest berry
* Eat a berry from its inventory
* Throw a berry to another agent

Health decreases over time. Agents whose health reaches zero are considered dead and no longer participate in the simulation.

Berries are replenished dynamically throughout the episode, creating an ongoing resource allocation problem.

## Agent Observations

Each agent makes decisions using the following information:

- Current health
- Number of berries carried
- Distance to nearest berry
- Society wellbeing

Society wellbeing is calculated as:

```text
wellbeing = (health + (berries × berry_health_payoff))
            / health_decay
```

## Behaviour Representation

Agent behaviours are represented as condition-action rules derived from discretised observations of the environment:

```text
IF <berry state>, <health state>, <society wellbeing state>
THEN <action>
```

For example:

```text
IF low berries, low health, low society wellbeing
THEN move
```

or

```text
IF medium berries, high health, low society wellbeing
THEN throw
```

These behaviour records form each agent's behaviour base and are used to identify emergent norms within the population

## Emergent Norm Detection
A behaviour is considered an emergent norm when it is adopted by a sufficiently large proportion of the population.

The current implementation defines emergence as:

```text
Adoption Rate >= 75% 
```

Norm statistics tracked include:
* First appearance
* Last appearance
* Number of occurances
* Maximum adoption rate
* Persistance over time
* Duration of adoption

Norm emergence is evaluated independently for each discretised environmental state, alloweing different norms to emerge under different resource and wellbeing conditions.

## Experimental Goals
The environment provides a platform for investigating:
* Norm emergence
* Norm persistence
* Cooperative behavior
* Resource inequality
* Social welfare
* Human-inspired decision making
* LLM-driven agents in social environments

Current experiments compare traditional rule-based agents with LLM-controlled agents that make decisions directly from environmental observations.

## Current Configuration

| Parameter | Value |
|------------|---------|
| Grid Size | 8 x 4 |
| Initial Agents | 4 |
| Initial Berries | 8 |
| Initial Agent Health | 1.0 |
| Health Decay per Step | 0.06 |
| Health Gain per Berry Eaten | 0.35 |
| Throw Threshold | 0.3 |
| Maximum Episode Length | 75 steps |
| Agent Activation | Random asynchronous order |
| Berry Regrowth | One berry spawned after each berry is foraged |


### Health States

| State | Range |
|---------|---------|
| Low Health | < 0.3 |
| Medium Health | 0.3 ≤ health < 0.7 |
| High Health | ≥ 0.7 |

### Berry States

Agents have no upper limit on the number of berries they may carry.

| State | Range |
|---------|---------|
| No Berries | 0 |
| Low Berries | 1 |
| Medium Berries | 2 ≤ berries < 4 |
| High Berries | ≥ 4 |

### Society Wellbeing States

Society wellbeing is defined as the minimum wellbeing of all other living agents

| State | Range |
|---------|---------|
| Low Society Wellbeing | < 10 |
| Medium Society Wellbeing | 10 ≤ wellbeing < 25 |
| High Society Wellbeing | ≥ 25 |

### Episode Termination

An episode ends when:

- All agents have died, or
- The maximum episode length is reached.

### Resource Dynamics

The environment maintains a constant supply of berry resources. Whenever a berry is foraged by an agent, a new berry is spawned at a randomly selected unoccupied grid location. Consequently, resource scarcity arises from competition for access to berries rather than resource depletion.

Agent inventories are unbounded, allowing individuals to accumulate and store resources over time.



## Running

Create and activate avirtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
py code/run.py
```

## Saving Runs

The simulation writes output files to:

```text
data/results/current_run/
```

The contents of this directory are overwritten each time a new simulation is executed.

To preserve the results of a particular run, use:

```bash
py code/save_run.py
```

You will be prompted to provide a name for the run. Alternatively, a name can be supplied directly:

```bash
py code/save_run.py llm_qwen3_seed_42
```

Saved runs are copied to:

```text
data/results/saved_runs/<run_name>/
```

Each saved run contains:

```text
agent_reports.csv
behaviour_bases.csv
emerged_norms.csv
model_episode_reports.csv
metadata.json
```

The `metadata.json` file stores information and notes associated with the saved experiment, helping to track and reproduce results.


## Future Work

Planned extensions include:

* Additional agent decision architectures
* Multi-episode learning
* Reinforcement learning agents
* More sophisticated social environments
* Comparative studies of LLM and non-LLM agents
* Investigation of norm emergence under heterogeneous populations