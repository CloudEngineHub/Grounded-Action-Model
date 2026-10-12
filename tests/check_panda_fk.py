"""GAM.data.panda_fk against the Franka Panda's documented ready pose:  python tests/check_panda_fk.py

Ready pose q = [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]: flange at (0.307, 0, 0.590) m in the base frame, the hand TCP
0.1034 m further along the approach axis -> (0.307, 0, 0.487).
"""
import numpy as np

from GAM.data.panda_fk import tcp_position

q = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
p = tcp_position(q); ref = np.array([0.3069, 0.0, 0.4866])
err = float(np.abs(p - ref).max())
print(f"[panda_fk] ready pose TCP = {np.round(p, 4)}  reference {ref}  max |diff| = {err * 1000:.2f} mm")
assert err < 2e-3, "Panda FK disagrees with the documented ready pose"
print("[panda_fk] ok")
