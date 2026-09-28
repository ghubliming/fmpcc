import numpy as np

GRAVITY_MAG = 9.81


def quat_to_rot(q):
    """MuJoCo quaternion (w, x, y, z) -> rotation matrix (body -> world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


class GeometricController:
    """Cascaded geometric tracking control of the Skydio X2 (Lee et al., 2010; Mellinger and Kumar, 2011):
    position PD with feed-forward -> thrust direction and yaw -> SO(3) attitude PD -> four motor thrusts
    (thrust priority, torque scaled down to at most one half when a motor saturates)."""

    def __init__(self, model, body_name='x2', motors=('thrust1', 'thrust2', 'thrust3', 'thrust4')):
        body_id = model.body(body_name).id
        self.mass = float(model.body_subtreemass[body_id])
        self.inertia = np.asarray(model.body_inertia[body_id], dtype=float).copy()
        cols = []
        for name in motors:
            r = np.asarray(model.site_pos[model.site(name).id], dtype=float)
            gear = np.asarray(model.actuator_gear[model.actuator(name).id], dtype=float)
            cols.append([1.0, r[1], -r[0], gear[5]])
        self.M_inv = np.linalg.inv(np.column_stack(cols))
        self.u_hover = self.mass * GRAVITY_MAG / 4.0
        self.u_max = max(2.0 * self.u_hover, 6.0)
        self.u_min = 0.0
        self.Kp_pos = np.array([4.0, 4.0, 8.0])
        self.Kd_pos = np.array([3.0, 3.0, 4.0])
        self.Kp_att = np.array([70.0, 70.0, 4.0])
        self.Kp_omega = np.array([2.5, 2.5, 1.0])
        self.thrust_floor = 0.1 * self.mass * GRAVITY_MAG

    def compute(self, p, q, v, omega_body, p_des, v_des=None, a_des=None, yaw_des=0.0):
        p, v = np.asarray(p, dtype=float), np.asarray(v, dtype=float)
        omega_body, p_des = np.asarray(omega_body, dtype=float), np.asarray(p_des, dtype=float)
        v_des = np.zeros(3) if v_des is None else np.asarray(v_des, dtype=float)
        a_des = np.zeros(3) if a_des is None else np.asarray(a_des, dtype=float)
        a_cmd = -self.Kp_pos * (p - p_des) - self.Kd_pos * (v - v_des) + a_des
        F_world = self.mass * (a_cmd + np.array([0.0, 0.0, GRAVITY_MAG]))
        F_norm = np.linalg.norm(F_world)
        if F_norm < 1e-6:
            return np.full(4, self.u_hover)
        b3_des = F_world / F_norm
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2_des = np.cross(b3_des, x_c)
        b2_norm = np.linalg.norm(b2_des)
        b2_des = np.array([0.0, 1.0, 0.0]) if b2_norm < 1e-6 else b2_des / b2_norm
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des])
        R = quat_to_rot(q)
        E = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([E[2, 1], E[0, 2], E[1, 0]])
        gyro = np.cross(omega_body, self.inertia * omega_body)
        tau = -self.Kp_att * e_R - self.Kp_omega * omega_body + gyro
        T = float(F_world @ R[:, 2])
        if T < self.thrust_floor:
            T = self.thrust_floor
        u = self.M_inv @ np.array([T, tau[0], tau[1], tau[2]])
        if u.max() > self.u_max or u.min() < self.u_min:
            thrust = u.mean()
            torque = u - thrust
            caps = []
            for tc in torque:
                if tc > 1e-9:
                    caps.append((self.u_max - thrust) / tc)
                elif tc < -1e-9:
                    caps.append((thrust - self.u_min) / (-tc))
            scale = max(min(1.0, min(caps)) if caps else 1.0, 0.5)
            u = np.clip(thrust + scale * torque, self.u_min, self.u_max)
        return u
