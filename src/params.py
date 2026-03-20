from latent import LatentParams
from simulate import SimulationParams
from control import ControlParams

latent_params = LatentParams(
    T=1.0,
    N=200,
    lambda01=0.5,
    lambda10=0.5,
    theta0=0
)

simulation_params = SimulationParams(
    T=1.0,
    N=200,
    sigma=0.5,
    A1=2.0,
    A0=-2.0,
    lambda_=0.5,
    S0=100
)

control_params = ControlParams(
    T=1.0,
    N=200,
    Q0=1.0
)