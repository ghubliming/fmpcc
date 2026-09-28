import numpy as np

BLEND_RADIUS = 0.45
CORRIDOR_LANES = {'L': -0.12, 'C': 0.0, 'R': 0.12}


def straight(p_start, p_end, duration, yaw=0.0):
    """Point to point with the cosine speed profile s(t) = (1 - cos(pi t / T)) / 2; (p, v, a, yaw) at t."""
    p_start, p_end = np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)
    delta, T, yaw = p_end - p_start, float(duration), float(yaw)

    def traj(t):
        if t >= T:
            return p_end.copy(), np.zeros(3), np.zeros(3), yaw
        tau = t / T
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        s_dot = 0.5 * np.pi / T * np.sin(np.pi * tau)
        s_ddot = 0.5 * (np.pi / T) ** 2 * np.cos(np.pi * tau)
        return p_start + s * delta, s_dot * delta, s_ddot * delta, yaw
    return traj


def blended(waypoints, radius, duration, yaw=0.0):
    """Waypoints joined by straight segments and circular arcs of radius <= `radius` at the corners (at most half
    of either adjacent segment), one cosine speed profile over the whole arc length."""
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    wps = [wps[0]] + [w for i, w in enumerate(wps[1:], 1) if np.linalg.norm(w - wps[i - 1]) > 1e-9]
    T, r_max, yaw = float(duration), float(radius), float(yaw)
    seg_vecs = [wps[i + 1] - wps[i] for i in range(len(wps) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    seg_dirs = [v / l for v, l in zip(seg_vecs, seg_lens)]
    fillets = []
    for i in range(1, len(wps) - 1):
        u_in, u_out = seg_dirs[i - 1], seg_dirs[i]
        cos_b = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        beta = float(np.arccos(cos_b))
        if beta < 1e-6:
            fillets.append(None)
            continue
        d = min(r_max * np.tan(beta / 2.0), 0.5 * seg_lens[i - 1], 0.5 * seg_lens[i])
        r = d / np.tan(beta / 2.0)
        n = u_out - cos_b * u_in
        n = n / np.linalg.norm(n)
        p1 = wps[i] - d * u_in
        center = p1 + r * n
        fillets.append({'p1': p1, 'p2': wps[i] + d * u_out, 'center': center, 'r': r, 'beta': beta,
                        'u0': u_in, 'e_r0': (p1 - center) / r})

    def _line(a, u):
        return lambda s: (a + s * u, u, np.zeros(3))

    def _arc(c, r, e_r0, u0):
        def ev(s):
            phi = s / r
            cp, sp = np.cos(phi), np.sin(phi)
            radial = cp * e_r0 + sp * u0
            return c + r * radial, -sp * e_r0 + cp * u0, -radial / r
        return ev

    elements, cursor = [], wps[0]
    for i in range(len(seg_dirs)):
        f = fillets[i] if i < len(fillets) else None
        seg_end = f['p1'] if f is not None else wps[i + 1]
        length = float(np.linalg.norm(seg_end - cursor))
        if length > 1e-9:
            elements.append((length, _line(cursor.copy(), seg_dirs[i])))
        if f is not None:
            elements.append((f['r'] * f['beta'], _arc(f['center'], f['r'], f['e_r0'], f['u0'])))
            cursor = f['p2']
        else:
            cursor = wps[i + 1]
    cum = np.concatenate(([0.0], np.cumsum(np.array([e[0] for e in elements]))))
    L, p_end = float(cum[-1]), wps[-1]

    def traj(t):
        if t >= T:
            return p_end.copy(), np.zeros(3), np.zeros(3), yaw
        tau = t / T
        s = L * 0.5 * (1.0 - np.cos(np.pi * tau))
        s_dot = L * 0.5 * np.pi / T * np.sin(np.pi * tau)
        s_ddot = L * 0.5 * (np.pi / T) ** 2 * np.cos(np.pi * tau)
        idx = max(min(int(np.searchsorted(cum, s, side='right')) - 1, len(elements) - 1), 0)
        p, tang, curv = elements[idx][1](s - cum[idx])
        return p, s_dot * tang, s_ddot * tang + s_dot ** 2 * curv, yaw
    return traj


def reference(scene, lane, rng):
    """Reference path, start position and duration of one flight, drawn as in the demonstrations."""
    z = float(rng.uniform(0.90, 1.30))
    if scene == 'corridor':
        jitter = float(rng.uniform(-0.03, 0.03)) if lane == 'C' else 0.0
        duration = float(rng.uniform(6.0, 10.0))
        y = CORRIDOR_LANES[lane]
        return straight([-2.8, y, z], [2.8, y, z], duration), np.array([-2.8, y + jitter, z]), duration
    if scene == 's_curve':
        jitter = float(rng.uniform(-0.04, 0.04))
        duration = float(rng.uniform(16.0, 22.0))
        y1, y2 = -0.8 + jitter, 0.8 + jitter
        wps = [(-3.2, y1, z), (-0.5, y1, z), (0.0, y1, z), (0.0, y2, z), (0.5, y2, z), (3.2, y2, z)]
        return blended(wps, BLEND_RADIUS, duration), np.array([-3.2, y1, z]), duration
    raise KeyError(f'unknown scene {scene!r}')
