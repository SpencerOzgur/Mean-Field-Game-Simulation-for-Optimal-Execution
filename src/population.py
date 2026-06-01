import numpy as np
import control
import equilibrium
import matplotlib.pyplot as plt


def sample_initial_inventory(rng, center: float, scale: float) -> float:
    return rng.normal(loc=center, scale=scale)


def make_equilibrium_params(
    subpops,
    control_params,
    sim_params,
    phi,
    psi,
    a=1.0,
    q0_list=None,
    scale_phi_by_kappa=True,
):
    if q0_list is None:
        q0_list = [sp.Q0 for sp in subpops]

    param_list = []
    for sp, q0 in zip(subpops, q0_list):
        phi_k = phi * sp.kappa if scale_phi_by_kappa else phi

        param = control.EquilibriumControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=q0,
            a=a,
            phi=phi_k,
            psi=psi,
            lam=sim_params.lambda_,
        )
        param_list.append(param)

    return param_list


def simulate_agent_inventory_paths(
    A_hat_k,
    subpops,
    control_params,
    sim_params,
    n_agents=20,
    seed=42,
    phi=0.02,
    psi=10.0,
    a=1.0,
    q0_scales=None,
    clip_q0=False,
    verbose=False,
    plot=False,
    plot_controls=False,
    figsize=(12, 7),
):
    rng = np.random.default_rng(seed)

    A_hat_k = np.asarray(A_hat_k, dtype=np.float64)
    if A_hat_k.ndim != 2:
        raise ValueError("A_hat_k must be a 2-D array of shape (K, N+1)")

    K = len(subpops)
    if A_hat_k.shape[0] != K:
        raise ValueError("A_hat_k must have one row per subpopulation")
    if A_hat_k.shape[1] != sim_params.N + 1:
        raise ValueError("A_hat_k must have shape (K, N+1)")

    if q0_scales is None:
        q0_scales = [0.1 * max(abs(sp.Q0), 1e-6) for sp in subpops]
    if len(q0_scales) != K:
        raise ValueError("q0_scales must have one entry per subpopulation")

    weights = np.array([sp.weight for sp in subpops], dtype=np.float64)

    # 1. Representative mean-field equilibrium
    param_list = make_equilibrium_params(
        subpops=subpops,
        control_params=control_params,
        sim_params=sim_params,
        phi=phi,
        psi=psi,
        a=a,
    )

    A_hat_list = [A_hat_k[k, :-1] for k in range(K)]

    nu_rep_list, q_rep_list, nu_bar = equilibrium.solve_mean_field_fixed_point(
        A_hat_list=A_hat_list,
        weights=weights,
        param_list=param_list,
    )

    agent_inventories = []
    agent_controls = []
    q_bar_mfg = []
    q_bar_emp = []
    nu_bar_emp = []

    if verbose:
        print("Representative params:")
        for sp in subpops:
            print(
                f"{sp.name}: Q0={sp.Q0}, kappa={sp.kappa}, "
                f"phi_eff={phi * sp.kappa}, psi={psi}"
            )

    # 2. Heterogeneous finite-agent simulation
    for k, sp in enumerate(subpops):
        q_group = np.empty((n_agents, sim_params.N + 1), dtype=np.float64)
        nu_group = np.empty((n_agents, sim_params.N), dtype=np.float64)

        for j in range(n_agents):
            Q0_j = sample_initial_inventory(
                rng=rng,
                center=sp.Q0,
                scale=q0_scales[k],
            )

            if clip_q0:
                Q0_j = max(Q0_j, 0.0)

            local_params = make_equilibrium_params(
                subpops=[sp],
                control_params=control_params,
                sim_params=sim_params,
                phi=phi,
                psi=psi,
                a=a,
                q0_list=[Q0_j],
            )[0]

            nu_j, q_j = equilibrium.equilibrium_control_fbsde(
                A_hat=A_hat_k[k, :-1],
                nu_bar=nu_bar,
                params=local_params,
            )

            nu_group[j] = nu_j
            q_group[j] = q_j

        agent_controls.append(nu_group)
        agent_inventories.append(q_group)
        q_bar_mfg.append(q_rep_list[k])
        q_bar_emp.append(q_group.mean(axis=0))
        nu_bar_emp.append(nu_group.mean(axis=0))

    results = {
        "agent_controls": agent_controls,
        "agent_inventories": agent_inventories,
        "q_bar_mfg": np.array(q_bar_mfg),
        "q_bar_emp": np.array(q_bar_emp),
        "nu_bar": nu_bar,
        "nu_rep": np.array(nu_rep_list),
        "q_rep": np.array(q_rep_list),
        "nu_bar_emp": np.array(nu_bar_emp),
    }

    if plot:
        t_grid_q = np.linspace(0.0, sim_params.T, sim_params.N + 1)
        t_grid_nu = np.linspace(0.0, sim_params.T, sim_params.N)

        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]

        # Inventory plot
        plt.figure(figsize=figsize)

        for k, sp in enumerate(subpops):
            color = colors[k % len(colors)]

            # thin individual paths
            for j in range(n_agents):
                plt.plot(
                    t_grid_q,
                    agent_inventories[k][j],
                    color=color,
                    alpha=0.20,
                    linewidth=1.2,
                )

            # mean-field subgroup path
            plt.plot(
                t_grid_q,
                q_bar_mfg[k],
                color=color,
                linewidth=3.0,
                label=f"{sp.name} mean field",
            )

            # empirical subgroup mean
            plt.plot(
                t_grid_q,
                q_bar_emp[k],
                color=color,
                linestyle="--",
                linewidth=3.0,
                label=f"{sp.name} empirical mean",
            )

        plt.title("Individual vs Mean Field Inventory")
        plt.xlabel("Time")
        plt.ylabel("Inventory")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Optional controls plot
        if plot_controls:
            plt.figure(figsize=figsize)

            for k, sp in enumerate(subpops):
                color = colors[k % len(colors)]

                for j in range(n_agents):
                    plt.plot(
                        t_grid_nu,
                        agent_controls[k][j],
                        color=color,
                        alpha=0.20,
                        linewidth=1.2,
                    )

                plt.plot(
                    t_grid_nu,
                    nu_rep_list[k],
                    color=color,
                    linewidth=3.0,
                    label=f"{sp.name} representative control",
                )

                plt.plot(
                    t_grid_nu,
                    nu_bar_emp[k],
                    color=color,
                    linestyle="--",
                    linewidth=3.0,
                    label=f"{sp.name} empirical mean control",
                )

            plt.plot(
                t_grid_nu,
                nu_bar,
                color="black",
                linewidth=3.0,
                label="aggregate mean field control",
            )

            plt.title("Individual vs Mean Field Controls")
            plt.xlabel("Time")
            plt.ylabel("Trading Rate")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
    return results
