import numpy as np

from .controller import GeometricController
from . import scenes

SCALE = 36.0
ALTITUDE = 1.0
ORIGIN = (0.5, 0.035)
START = (0.525, -0.28)
GOAL_Y = -0.1 + 2.5 * 0.18
FIELD = ((0.2, 0.8), (-0.3, 0.4))
SLACK = 8.0
SPEED_MAX = 6.0


def to_world(x_a, y_a):
    """Similarity map from the D3IL table (avoiding frame) to the arena: X = 36 (y_a - 0.035), Y = -36 (x_a - 0.5)."""
    return SCALE * (y_a - ORIGIN[1]), -SCALE * (x_a - ORIGIN[0])


def to_avoiding(X, Y):
    return ORIGIN[0] - Y / SCALE, ORIGIN[1] + X / SCALE


class PillarsPlant:
    """The quadrotor in place of the Panda on D3IL-avoiding (UAV-pillars). Takes the planner's commanded position
    in table coordinates, holds 1.0 m altitude, and tracks each setpoint for one 1 s control period (100 physics
    steps) with the geometric controller, the reference moving towards it at <= 1.0 m/s, no velocity
    feed-forward. A pillar contact, leaving the arena (+8 m) or |v| > 6 m/s ends the flight."""

    def __init__(self, control_hz=1.0, v_max=1.0, max_steps=250):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scenes.xml('pillars'))
        self.data = mujoco.MjData(self.model)
        self.controller = GeometricController(self.model)
        self.dt = float(self.model.opt.timestep)
        self.n_sub = max(1, int(round(1.0 / (self.dt * control_hz))))
        self.v_max, self.max_steps = float(v_max), int(max_steps)
        xs = [to_world(x, y)[0] for x in FIELD[0] for y in FIELD[1]]
        ys = [to_world(x, y)[1] for x in FIELD[0] for y in FIELD[1]]
        self.arena_lb = np.array([min(xs) - SLACK, min(ys) - SLACK, 0.15])
        self.arena_ub = np.array([max(xs) + SLACK, max(ys) + SLACK, ALTITUDE + SLACK])
        self.start_w = np.array([*to_world(*START), ALTITUDE])

    def start(self):
        pass

    def close(self):
        pass

    def reset(self, random=True, context=None):
        m, d = self.model, self.data
        self._mj.mj_resetData(m, d)
        d.qpos[:3] = self.start_w
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = 0.0
        self._mj.mj_forward(m, d)
        for _ in range(int(round(0.5 / self.dt))):
            self._control(self.start_w, np.zeros(3))
        self.reference = self.start_w.copy()
        self.steps, self.success = 0, False
        return self._obs()

    def robot_state(self):
        x, y = self._obs()
        return np.array([x, y, 0.12], dtype=float)

    def step(self, action, gripper_width=None):
        target = np.array([*to_world(float(action[0]), float(action[1])), ALTITUDE])
        contact, diverged = False, None
        for _ in range(self.n_sub):
            gap = target - self.reference
            move = min(np.linalg.norm(gap), self.v_max * self.dt)
            self.reference = self.reference + (gap / np.linalg.norm(gap) * move if move > 1e-12 else 0.0)
            hit = self._control(self.reference, np.zeros(3))
            contact = contact or hit
            if hit:
                break
            diverged = self._diverged()
            if diverged:
                break
        obs = self._obs()
        success = bool(obs[1] > GOAL_Y)
        done = success or contact or diverged is not None or self.steps >= self.max_steps - 1
        self.steps += 1
        self.success = self.success or success
        return obs, 0.0, bool(done), (None, self.success)

    def _control(self, p_des, v_des):
        m, d = self.model, self.data
        d.ctrl[:4] = self.controller.compute(d.qpos[:3].copy(), d.qpos[3:7].copy(), d.qvel[:3].copy(),
                                             d.qvel[3:6].copy(), p_des, v_des)
        self._mj.mj_step(m, d)
        return any(scenes.obstacle_contact(m, d.contact[i]) for i in range(d.ncon))

    def _diverged(self):
        p, v = self.data.qpos[:3], self.data.qvel[:3]
        if np.any(p < self.arena_lb) or np.any(p > self.arena_ub):
            return 'left_arena'
        if np.linalg.norm(v) > SPEED_MAX:
            return 'overspeed'
        if not np.all(np.isfinite(p)):
            return 'nan_state'
        return None

    def _obs(self):
        p = self.data.qpos[:3]
        return np.asarray(to_avoiding(float(p[0]), float(p[1])), dtype=np.float32)
