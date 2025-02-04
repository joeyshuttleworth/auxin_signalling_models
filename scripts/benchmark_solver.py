#!/usr/bin/env python3

import argparse
import os
import timeit

import matplotlib.pyplot as plt
import numpy as np

from auxin_signalling_pathway.auxin_signalling_pathway import \
    auxin_signalling_pathway as asp_class


def setup_output_directory(dir_name, subdir_name):
    dirname = os.path.join(dir_name, subdir_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', default='output')
    parser.add_argument('-n', '--no_evals', default=50, type=int)
    parser.add_argument('--figsize', default=[5, 4], type=float, nargs=2)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = setup_output_directory(args.output, "benchmark")

    kwargs_list = [
        {'nARFs': 1, 'nIAAs': 1},
        {'nARFs': 2, 'nIAAs': 1},
        {'nARFs': 1, 'nIAAs': 2},
        {'nARFs': 4, 'nIAAs': 4}
    ]

    for i, kwargs in enumerate(kwargs_list):
        asp_model = asp_class(**kwargs)
        # odes, _, _, _ = asp_model.generate_ode_system()
        param_dict = asp_model.get_default_parameters()
        ic_dict = asp_model.get_default_initial_conditions()

        # print("============\n\nodes\n\n===============")
        # print(dict(zip(ic_dict.keys(), odes)))

        solver = asp_model.make_forward_solver()

        params = np.array([param_dict[key] for key in param_dict])\
            .astype(np.float64)

        params = np.array([p for p in param_dict.values()]).astype(np.float64)
        ic = np.array([x for x in ic_dict.values()]).astype(np.float64)

        ts = np.linspace(0, 10, 1000)

        timer = timeit.default_timer()
        for _ in range(args.no_evals):
            res, _ = solver(ts, params, ic, 1.0)

        timer_end = timeit.default_timer()

        elapsed_time = timer_end - timer
        print(f"{elapsed_time} seconds taken for {args.no_evals} evaluations")

        fig = plt.figure(figsize=args.figsize)
        ax = fig.subplots()
        ax.plot(ts, res / res.max(axis=1)[:, None], label=ic_dict.keys())
        ax.legend()
        fig.savefig(os.path.join(output_dir, f"simulation_{i}"))


if __name__ == "__main__":
    main()
