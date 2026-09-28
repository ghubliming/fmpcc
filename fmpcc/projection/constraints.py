import numpy as np


def halfspace_row(line, enlarge, dim, idx):
    """Halfspace through two planar points, feasible 'below' or 'above' the line (for a vertical line: to the
    left / right); enlarge > 0 moves it into the feasible side. Returns (row, d) with row · x <= d."""
    p0, p1 = np.asarray(line[0], dtype=float), np.asarray(line[1], dtype=float)
    side = line[2]
    ix, iy = idx['x'], idx['y']
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    row = np.zeros(dim)
    if dx != 0.0 and dy != 0.0:
        m = dy / dx
        n = np.array([-1.0, 1.0 / m])
        n = n / np.linalg.norm(n)
        if (m > 0 and side == 'below') or (m < 0 and side == 'above'):
            n = -n
        p0e = p0 + enlarge * n
        d = p0e[1] - m * p0e[0]
        if side == 'below':
            row[ix], row[iy] = -m, 1.0
        elif side == 'above':
            row[ix], row[iy], d = m, -1.0, -d
        return row, d
    if dy == 0.0:
        if side == 'above':
            row[iy] = -1.0
            return row, -(p0[1] + enlarge)
        row[iy] = 1.0
        return row, p0[1] - enlarge
    if side == 'above':
        row[ix] = -1.0
        return row, -(p0[0] + enlarge)
    row[ix] = 1.0
    return row, p0[0] - enlarge


def action_bounds(bounds, dim, idx):
    lower, upper = -np.inf * np.ones(dim), np.inf * np.ones(dim)
    for name, (lo, hi) in bounds.items():
        lower[idx[name]], upper[idx[name]] = lo, hi
    return lower, upper


def planar_set(cfg, geometry, tightened):
    """Constraint list of a planar scene (D3IL-avoiding and UAV-pillars) in the order the program expects:
    halfspaces, action bounds, keep-out disks, dynamics."""
    idx = cfg['indices']
    dim = len(idx)
    g = cfg['geometries'][geometry]
    enlarge = cfg['tightening'] if tightened else 0
    out = [('ineq', halfspace_row(line, enlarge, dim, idx)) for line in g['halfspaces']]
    lower, upper = action_bounds(cfg['action_bounds'], dim, idx)
    out += [('lb', lower), ('ub', upper)]
    out += [('sphere_outside', [idx['x'], idx['y']], disk['center'], disk['radius'] + enlarge) for disk in g['disks']]
    out += [('deriv', np.array([idx[x], idx[dx]])) for x, dx in cfg['dynamics']]
    return out


def planar_violation(cfg, geometry, observation, action_dim):
    """Nominal halfspaces and disks violated by the measured position: the list of violation amounts."""
    idx = cfg['indices']
    dim = len(idx)
    g = cfg['geometries'][geometry]
    amounts = []
    for line in g['halfspaces']:
        row, d = halfspace_row(line, 0, dim, idx)
        if observation @ row[action_dim:] >= d:
            amounts.append(observation @ row[action_dim:] - d)
    pos = observation[[idx['x'] - action_dim, idx['y'] - action_dim]]
    for disk in g['disks']:
        if np.linalg.norm(pos - disk['center']) < disk['radius']:
            amounts.append(disk['radius'] - np.linalg.norm(pos - disk['center']))
    return amounts


ALIGNING = {'dx': 0, 'dy': 1, 'dz': 2, 'des_x': 3, 'des_y': 4, 'des_z': 5, 'x': 6, 'y': 7, 'z': 8}


def aligning_set(cfg, tightened, action_normalizer):
    """Constraint list of D3IL-aligning, in the order of the runs of record: workspace box on the measured
    position, action bounds (the training range), dynamics, halfspaces, keep-out disks."""
    margin = cfg['tightening'] if tightened else 0.0
    lb, ub = np.array(cfg['workspace']['lb']), np.array(cfg['workspace']['ub'])
    if tightened and cfg['tightening'] > 0.0:
        lb += cfg['tightening']
        ub -= cfg['tightening']
    out = [('lb', np.concatenate([np.full(6, -np.inf), lb])), ('ub', np.concatenate([np.full(6, np.inf), ub]))]
    a_lb = np.asarray(action_normalizer.mins, dtype=float)
    a_ub = np.asarray(action_normalizer.maxs, dtype=float)
    out += [('lb', np.concatenate([a_lb, np.full(6, -np.inf)])), ('ub', np.concatenate([a_ub, np.full(6, np.inf)]))]
    out += [('deriv', [ALIGNING[x], ALIGNING[dx]]) for x, dx in cfg['dynamics']]
    out += [('ineq', halfspace_row(line, margin, 9, ALIGNING)) for line in cfg['halfspaces']]
    out += [('sphere_outside', [ALIGNING[d] for d in disk['dimensions']], disk['center'], disk['radius'] + margin)
            for disk in cfg['disks']]
    return out


def aligning_violations(path, cfg):
    """Violations of the nominal aligning set by the measured end-effector path (T, 3), per step."""
    pos = np.asarray(path, dtype=float)
    lb, ub = np.array(cfg['workspace']['lb'], dtype=float), np.array(cfg['workspace']['ub'], dtype=float)
    box = np.maximum(np.maximum(0.0, lb - pos), np.maximum(0.0, pos - ub)).max(axis=1)
    half = np.zeros(len(pos))
    for (p1, p2, side) in cfg['halfspaces']:
        x1, y1, dx, dy = float(p1[0]), float(p1[1]), float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])
        nx, ny = (-dy, dx) if side == 'above' else (dy, -dx)
        nl = float(np.hypot(nx, ny))
        sd = nx / nl * (pos[:, 0] - x1) + ny / nl * (pos[:, 1] - y1)
        half = np.maximum(half, np.where(sd < -1e-6, -sd, 0.0))
    disk = np.zeros(len(pos))
    for d in cfg['disks']:
        idx = [ALIGNING[k] - 6 for k in d['dimensions']]
        dist = np.linalg.norm(pos[:, idx] - np.array(d['center'][:len(idx)], dtype=float), axis=1)
        disk = np.maximum(disk, np.maximum(0.0, float(d['radius']) - dist))
    return box, half, disk


UAV = {'dx': 0, 'dy': 1, 'dz': 2, 'x_des': 3, 'y_des': 4, 'z_des': 5, 'x': 6, 'y': 7, 'z': 8}


def _leaned(hs):
    """Unit normal into the feasible side and a point of a vertical wall leaned by `deg` about its line at
    height z_ref: feasible <=> n · (p - P0) >= 0."""
    t = float(np.tan(np.radians(float(hs['lean']['deg']))))
    (x1, y1), (x2, y2) = hs['line']
    nrm = float(np.hypot(x2 - x1, y2 - y1))
    nx, ny = -(y2 - y1) / nrm, (x2 - x1) / nrm
    if hs['side'] != 'above':
        nx, ny = -nx, -ny
    return np.array([nx, ny, t], dtype=float) / float(np.sqrt(1.0 + t * t)), np.array([x1, y1, float(hs['lean']['z_ref'])])


def _live(hs, x):
    return 'x_active' not in hs or x is None or hs['x_active'][0] <= float(x) <= hs['x_active'][1]


def uav_set(cfg, geometry, tightened, bind, action_bound, action_normalizer, current_x=None):
    """Constraint list of a quadrotor scene in the order of the runs of record: workspace box, action bounds,
    dynamics, halfspaces (live on their x span only; leaned planes and x-z roofs included), keep-out disks.
    Every surface is moved outward by the body radius (plus the tightening); `bind` puts the geometry on the
    commanded ('setpoint') or on the measured ('measured') position."""
    g = cfg['geometries'][geometry]
    D = {'x': 3, 'y': 4, 'z': 5} if bind == 'setpoint' else {'x': 6, 'y': 7, 'z': 8}
    off = D['x']
    margin = cfg['body_radius'] + cfg['planning_pad'] + (cfg['tightening'] if tightened else 0.0)
    out = []
    if g.get('workspace'):
        lb, ub = np.array(g['workspace']['lb'], dtype=float), np.array(g['workspace']['ub'], dtype=float)
        tail = 9 - off - 3
        out += [('lb', np.concatenate([np.full(off, -np.inf), lb + margin, np.full(tail, -np.inf)])),
                ('ub', np.concatenate([np.full(off, np.inf), ub - margin, np.full(tail, np.inf)]))]
    if action_bound:
        a_lb = np.asarray(action_normalizer.mins, dtype=float)
        a_ub = np.asarray(action_normalizer.maxs, dtype=float)
        out += [('lb', np.concatenate([a_lb, np.full(6, -np.inf)])), ('ub', np.concatenate([a_ub, np.full(6, np.inf)]))]
    out += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2]), ('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
    for hs in g['halfspaces']:
        if not _live(hs, current_x):
            continue
        if 'lean' in hs:
            n3, P0 = _leaned(hs)
            row = np.zeros(9)
            row[D['x']], row[D['y']], row[D['z']] = -n3
            out.append(('ineq', (row, -(float(n3 @ P0) + margin))))
            continue
        idx = {'x': D['x'], 'y': D['z'] if hs.get('plane', 'xy') == 'xz' else D['y']}
        out.append(('ineq', halfspace_row([hs['line'][0], hs['line'][1], hs['side']], margin, 9, idx)))
    out += [('sphere_outside', [D[d] for d in disk['dimensions']], disk['center'], disk['radius'] + margin)
            for disk in g['disks']]
    return out


def uav_violations(path, cfg, geometry):
    """Executed violations: the measured positions (T, 3) against the nominal geometry grown by the body
    radius. Returns (violating steps, total depth in m)."""
    g = cfg['geometries'][geometry]
    r = float(cfg['body_radius'])
    n, total = 0, 0.0
    for p in np.asarray(path, dtype=float):
        pen = 0.0
        if g.get('workspace'):
            lb, ub = np.array(g['workspace']['lb'], dtype=float), np.array(g['workspace']['ub'], dtype=float)
            pen += float(np.clip((lb + r) - p, 0, None)[np.isfinite(lb)].sum())
            pen += float(np.clip(p - (ub - r), 0, None)[np.isfinite(ub)].sum())
        for hs in g['halfspaces']:
            if not _live(hs, p[0]):
                continue
            if 'lean' in hs:
                n3, P0 = _leaned(hs)
                pen += max(0.0, r - float(n3 @ (p - P0)))
                continue
            (x1, y1), (x2, y2) = hs['line']
            nrm = np.hypot(x2 - x1, y2 - y1)
            nx, ny = (-(y2 - y1) / nrm, (x2 - x1) / nrm)
            q = p[2] if hs.get('plane', 'xy') == 'xz' else p[1]
            signed = nx * (p[0] - x1) + ny * (q - y1)
            pen += max(0.0, r - (signed if hs['side'] == 'above' else -signed))
        for disk in g['disks']:
            idx = [{'x': 0, 'y': 1, 'z': 2}[d] for d in disk['dimensions']]
            pen += max(0.0, (disk['radius'] + r) - float(np.linalg.norm(p[idx] - np.asarray(disk['center'], dtype=float))))
        if pen > 1e-9:
            n += 1
            total += pen
    return n, total
