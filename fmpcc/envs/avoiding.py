def make(plant='panda'):
    """D3IL obstacle avoidance with the Panda (IK + joint PD), or the same task flown by the quadrotor."""
    if plant == 'panda':
        from .d3il.envs.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
        return ObstacleAvoidanceEnv()
    if plant == 'quadrotor':
        from .quadrotor.pillars import PillarsPlant
        return PillarsPlant()
    raise KeyError(f'unknown plant {plant!r}')
