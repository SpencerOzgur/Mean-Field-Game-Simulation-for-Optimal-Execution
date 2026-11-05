"""
Minimal stand-in for pomegranate<1.0, which ABIDES imports for its order-size
model but which does not build on current Python / NumPy.

Only the one call ABIDES makes is supported: GeneralMixtureModel.from_json(...)
followed by .sample(random_state=...). The mixture (one log-normal plus
near-point-mass normals at round lots) is sampled with the same parameters.

Import this module before importing abides_markets.
"""

import json
import sys
import types

import numpy as np


class GeneralMixtureModel:
    def __init__(self, components, weights):
        self.components = components
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

    @classmethod
    def from_json(cls, payload: str) -> "GeneralMixtureModel":
        spec = json.loads(payload)
        components = [(d["name"], d["parameters"]) for d in spec["distributions"]]
        return cls(components, spec["weights"])

    def sample(self, random_state: np.random.RandomState = None) -> float:
        rs = random_state if random_state is not None else np.random
        name, (mu, sigma) = self.components[rs.choice(len(self.weights), p=self.weights)]
        if name == "LogNormalDistribution":
            return float(rs.lognormal(mean=mu, sigma=sigma))
        if name == "NormalDistribution":
            return float(rs.normal(loc=mu, scale=sigma))
        raise ValueError(f"unsupported distribution {name}")


if "pomegranate" not in sys.modules:
    _mod = types.ModuleType("pomegranate")
    _mod.GeneralMixtureModel = GeneralMixtureModel
    sys.modules["pomegranate"] = _mod
