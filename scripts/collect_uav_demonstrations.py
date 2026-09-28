import argparse

from fmpcc.data.uav_demonstrations import collect


def main():
    p = argparse.ArgumentParser(description='Generate the quadrotor demonstrations of one scene.')
    p.add_argument('--scene', required=True, choices=['corridor', 's_curve'])
    p.add_argument('--trials', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='datasets/uav')
    args = p.parse_args()
    collect(args.scene, args.trials, f'{args.out}/{args.scene}', seed=args.seed)


if __name__ == '__main__':
    main()
