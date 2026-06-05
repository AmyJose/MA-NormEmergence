from dataclasses import dataclass

@dataclass(frozen=True)
class Behaviour:
    pre : tuple
    acion : str