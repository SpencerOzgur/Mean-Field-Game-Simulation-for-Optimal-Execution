from latent import LatentParams
from simulate import SimulationParams

latent_params = LatentParams(
    T=1.0,
    N=1000,
    lambda01=0.5,
    lambda10=0.5,
    theta0=0
)

simulation_params = SimulationParams(
    T=1.0,
    N=1000,
    sigma=1.0,
    A1=1.0,
    A0=-1.0,
    lambda_=0.5,
    S0=100
)