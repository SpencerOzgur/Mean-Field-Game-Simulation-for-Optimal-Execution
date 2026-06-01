import numpy as np
import latent
from params import latent_params, simulation_params


def test_simulate_latent_path_shape():
    path = latent.simulate_latent_path(params=latent_params)
    assert isinstance(path, np.ndarray)
    assert path.shape == (simulation_params.N + 1,)


def test_simulate_latent_path_states_are_binary():
    path = latent.simulate_latent_path(params=latent_params)
    assert np.all(np.isin(path, [0, 1]))
