#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import networkx as nx

from numbalsoda import lsoda, lsoda_sig
import numpy as np
import numba as nb
from numba import cfunc, njit

from auxin_signalling_pathway.auxin_signalling_pathway import \
    auxin_signalling_pathway as asp_class


def setup_output_directory(dir_name, subdir_name):
    dirname = os.path.join(dir_name, subdir_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname


class TestGeneratePathway(unittest.TestCase):

    def setUp(self):
        test_output_dir = setup_output_directory(
            'test_output', f"{type(self).__name__}")
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

        kwargs_list = [
            {'nARFs': 1, 'nIAAs': 1},
            {'nARFs': 2, 'nIAAs': 1},
            {'nARFs': 1, 'nIAAs': 2},
            {'nARFs': 4, 'nIAAs': 4}
        ]

        self.test_models =  [asp_class(**kwargs) for
                             kwargs in kwargs_list]

    def test_graph_generation(self):
        kwargs_list = [
            {'nARFs': 1, 'nIAAs': 1},
            {'nARFs': 2, 'nIAAs': 1},
            {'nARFs': 1, 'nIAAs': 2},
            {'nARFs': 4, 'nIAAs': 4}
        ]

        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)
            logging.info(nx.write_network_text(asp_model.graph))
            asp_model.draw_graph()
            plt.savefig(os.path.join(self.output_dir, f"test_graph_{i}"))
            plt.clf()

            with open(os.path.join(self.output_dir,
                                   f"test_graph_{i}.txt"), 'w') as fout:

                rhs_func = asp_model.generate_rhs_func()

            logging.info(asp_model._reactions_set)
            logging.info(asp_model.generate_ode_system())

    def generate_solve_ode(self, asp_model, i):
        odes, free_variables, transition_rates, constraints = \
            asp_model.generate_ode_system()
        with open(os.path.join(self.output_dir,
                                f"test_graph_{i}.txt"), 'w') as fout:
            fout.write(f"ODEs: {odes}\n")
            fout.write(f"constraints: {constraints}\n")
            fout.write(f"free variables: {free_variables}\n")
            fout.write(f"transition rates: {transition_rates}\n")

        rhs_func, params = asp_model.generate_rhs_func(njitted=False)

        ic = asp_model.get_default_initial_conditions()
        ic = np.array([val for key, val in ic.items()
                        if key != asp_model.empty_set_node]).astype(np.float64)

        params = np.array(params).astype(np.float64)
        data = np.array([1.0] + list(params)).astype(np.float64) * 0.99
        n_data = len(data)

        logging.info("Derivative is ", rhs_func(ic, params,
                                                np.array([.99])))

        rhs_func = njit(rhs_func)
        print('res', rhs_func(ic, params, np.array([.99])))

        cfunc_rhs = asp_model.make_cfunc_rhs()

        crhs_ptr = cfunc_rhs.address
        ts = np.linspace(0, 1000, 100)
        res = lsoda(crhs_ptr, ic, np.array(ts), data=data)
        print(res)

    def test_generate_solve_ode(self):
        for i, asp_model in enumerate(self.test_models):
            self.generate_solve_ode(asp_model, i)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
