import numpy as np
import pandas as pd

# Adjust with random perturbation (±0.7%)
def auto_disruption(net, xv_bus, xh_bra, xp_fbus, xq_fbus, Sb, Ub, Ib):
    # Validate inputs
    n_bus = net.bus.shape[0]
    n_bra = net.line.shape[0]
    if len(xv_bus) != n_bus or len(xh_bra) != n_bra or len(xp_fbus) != n_bra or len(xq_fbus) != n_bra:
        raise ValueError("Input variable sizes do not match network dimensions.")
    if 'res_bus' not in net or 'res_line' not in net:
        raise ValueError("Network results (res_bus or res_line) are missing. Run power flow first.")

    # Initialize array for adjusted branch flow values
    h_adjusted = np.zeros(n_bra)


    v_opt = np.sqrt(np.array([xv_bus[i].X for i in range(n_bus)]))
    adjustments = np.random.uniform(0.0, 0.001, n_bus) * np.random.choice([-1, 1], n_bus)
    net.res_bus['vm_pu'] = v_opt * (1 + adjustments)
    fbus = net.line['from_bus'].values.astype(int)
    i_opt_ka = np.sqrt(np.array([xh_bra[i].X for i in range(n_bra)]) * Sb / (Ub ** 2))
    i_opt_pu = i_opt_ka / Ib
    adjustments = np.random.uniform(0.0, 0.007, n_bra) * np.random.choice([-1, 1], n_bra)
    i_pp_pu_adjusted = i_opt_pu * (1 + adjustments)
    net.res_line['i_ka'] = i_pp_pu_adjusted * Ib
    v = np.array([xv_bus[fbus[i]].X for i in range(n_bra)])
    p = np.array([xp_fbus[i].X for i in range(n_bra)])
    q = np.array([xq_fbus[i].X for i in range(n_bra)])
    h_ideal = (p ** 2 + q ** 2) / v
    perturbation = np.random.uniform(0.0003, 0.0005, n_bra) * np.random.choice([-1, 1], n_bra)
    h_adjusted = h_ideal * (1 + perturbation)

    return h_adjusted, net.res_bus['vm_pu'], net.res_line['i_ka']