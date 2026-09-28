import numpy as np


class PredictiveSamplingTracker:
    """MuJoCo MPC predictive sampling (MJX, GPU) as the setpoint tracker: 16 sampled thrust sequences over a
    0.3 s horizon, cost ‖p - p_des‖² + 0.1 ‖v‖², five improvement iterations per control step, collisions off
    in the planning model. Needs the second environment (requirements-mjx.txt)."""

    def __init__(self, model, n_samples=16, horizon=0.3, n_improve=5, vel_weight=0.1):
        import jax
        import jax.extend.backend as jax_backend
        import jax.numpy as jnp
        from mujoco import mjx
        from . import predictive_sampling as ps
        if not hasattr(jax_backend, 'backends'):
            jax_backend.backends = lambda: {d.platform for d in jax.devices()}
        self._jax, self._jnp, self._mjx, self._ps = jax, jnp, mjx, ps
        steps = max(4, int(round(horizon / float(model.opt.timestep))))

        def cost(_m, d, p_des):
            return jnp.sum((d.qpos[:3] - p_des) ** 2) + float(vel_weight) * jnp.sum(d.qvel[:3] ** 2), ()

        contype, conaffinity = model.geom_contype.copy(), model.geom_conaffinity.copy()
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        try:
            self._mx_model = mjx.put_model(model)
        finally:
            model.geom_contype[:] = contype
            model.geom_conaffinity[:] = conaffinity
        self._planner = ps.Planner(model=self._mx_model, cost=cost, noise_scale=0.3, horizon=steps, nspline=steps,
                                   nsample=n_samples, interp='zero',
                                   instruction_fn=lambda _m, _d: (jnp.zeros(3), jnp.zeros(0)))
        self._policy = jnp.zeros((steps, int(model.nu)))
        self._rng = jax.random.PRNGKey(0)
        self._n_improve = int(n_improve)
        self._improve = jax.jit(ps.improve_policy)

    def compute(self, p, q, v, om, p_des, v_des=None, a_des=None, yaw_des=0.0):
        jnp = self._jnp
        data = self._mjx.make_data(self._mx_model).replace(qpos=jnp.array([*p, *q], dtype=jnp.float64),
                                                            qvel=jnp.array([*v, *om], dtype=jnp.float64))
        p_des = jnp.array(p_des, dtype=jnp.float64)
        for _ in range(self._n_improve):
            self._rng, key = self._jax.random.split(self._rng)
            self._policy, _ = self._improve(self._planner, data, p_des, self._policy, key)
        action = np.array(self._policy[0])
        self._policy = self._ps.resample(self._planner, self._policy, 1)
        return action[:4]
