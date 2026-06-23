# Norm Emergence in Multi-Agent Systems

A Python/Mesa implementation of a norm emergence environment for studying cooperation, resource sharing, social welfare and large language model (LLM) decision making in multi-agent systems.

The project investigates how behavioural norms emerge from repeated interactions between autonomous agents competing for limited resources. Agents forage for food, consume resources to maintain their health, and may choose to share resources with others. Individual behaviours are tracked over time and analysed to identify emergent norms, patterns of social coordination and the influence of different decision-making architectures.

The environment serves as an experimental platform for comparing traditional rule-based policies with LLM-controlled agents operating withing mixed populations.

## Features
### Environment
* Mesa-based agent simulation
* Orthogonal Von Neumann grid world
* Resource foraging and collection
* Dynamic berry replenishment
* Health-based survival mechanics
* Resource consumption (eat)
* Resource sharing (throw)
* Agent mortality

### Agent Decision Making
* Rule-based agents
    * Selfish policy
    * Selfless policy
    * Utilitarian policy
* LLM-based agents
    * Prompt-driven behaviour
    * Observation-based decision making
    * Reasoning trace collection

### Norm Analysis
* Behaviour base generation
* Emergent norm detection
* Norm persistence tracking
* Norm adoption analysis
* Per-agent behavioural analysis
* Population-level behavioural analysis

### Experimental Suport
* Reproducible random seeds
* Automated experiment execution
* Structured result storage
* CSV and JSONL exports
* Plot generation and experiment summarisation

## Environment

Agents inhabit a two-dimensional grid populated with berry resources. Berries act as the primary survival resource and must be collected and consumed to maintain health.

At each timestep an agent may:

* Move towards the nearest berry
* Eat a berry from its inventory
* Throw a berry to another agent

Health decreases over time. Agents whose health reaches zero are considered dead and no longer participate in the simulation.

Whenever a berry is foraged, a replacement berry is spawned at a random unoccupied location. Resource scarcity therefore emerges from competition for access to resources rather than depletion of the resource pool itself.

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

Norm emergence is evaluated independently for each discretised environmental state, allowing different norms to emerge under different resource and wellbeing conditions.

## Experimental Design
The primary experiments investigate how prompting influences the behaviour of LLM-controlled agents and the norms that emerge within mixed populations.

Each experimental population contains:
```text
1 LLM Agent
3 Rule-Based Agents
```

### Rule-Based Policies
* Selfish
* Selfless
* Utilitarian

### LLM Prompt Types
* Baseline (no aim statement given)
* Selfish
* Cooperative

For each prompt-policy combination, multiple random seeds are executed to evaluate behavioural consistency and population-level outcomes.

## Experimental Metric
The environment records metrics including:
* Agent health
* Agent wellbeing
* Resource consumption
* Resource sharing
* Resource inequality (Gini index)
* Mortality
* Behaviour diversity
* Emerged norms
* Norm persistence

For LLM agents, reasoning traces and selected actions are also recorded for later analysis.

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
| Berry Regrowth | One berry per foraged berry |


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

Run a simulation:

```bash
py code/run.py
```

## Experimental Outputs
Each run produces a collection of output files, including:

```text
agent_reports.csv
behaviour_bases.csv
emerged_norms.csv
model_episode_reports.csv
metadata.json
llm_reasoning.jsonl
```
### Output Descriptions

| File | Description |
|---------|---------|
| agent_reports.csv | Per-agent state and action data |
| behaviour_bases.csv | Behaviour frequencies for each agent |
| emerged_norms.csv | Emergent norm statistics |
| model_episode_reports.csv | Episode-level aggregate metrics |
| metadata.json | Run configuration and experimental metadata |
| llm_reasoning.jsonl | LLM reasoning traces and selected actions|

## Plot Generation
Saved experiments can be analysed using:
```bash
python saved_runs/plot_figures.py <exp_name>
```
Generated figures and summary CSV files are written to:
```bash
saved_runs/plots/<exp_name>
```

## Future Work

Planned extensions include:

* Multi-episode learning
* Reinforcement learning agents
* Larger populations
* More sophisticated social environments
* Dynamic social networks
* Additional LLM architectures
* Comparative studies of prompting strategies
* Investigation of norm emergence in heterogeneous populations