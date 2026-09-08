from latent import LatentParams
from simulate import SimulationParams
from control import ControlParams
from dataclasses import dataclass
from data_loader import get_hjn_market_params

market = get_hjn_market_params("INTC")

@dataclass
class SubPopParams:
    name: str
    weight: float
    prior: float
    Q0: float
    kappa: float


latent_params = LatentParams(T=1.0, N=1000, lambda01=3.0, lambda10=2.0, theta0=1)

simulation_params = SimulationParams(
    T=1.0,
    N=1000,
    sigma=market["sigma"],
    A1=1.0,
    A0=-1.0,
    S0=market["S0"],
    lambda_=0.05,
)

control_params = ControlParams(T=1.0, N=1000, Q0=1.0)
