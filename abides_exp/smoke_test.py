import numpy as np
from abides_core import abides
from abides_markets.configs import rmsc04

end_state = abides.run(rmsc04.build_config(seed=0, end_time="10:00:00"))
l1 = end_state["agents"][0].order_books["ABM"].get_L1_snapshots()
bids = [b[1] for b in l1["best_bids"] if b[1] is not None]
print(f"agents: {len(end_state['agents'])}, L1 snapshots: {len(l1['best_bids'])}, "
      f"median bid: {np.median(bids) / 100:.2f}")
print("ABIDES OK")
