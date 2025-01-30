#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import networkx as nx

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

                odes, free_variables, transition_rates, constraints = \
                    asp_model.generate_ode_system()

                rhs_func = asp_model.generate_rhs_func()
                print(rhs_func)

                fout.write(f"ODEs: {odes}\n")
                fout.write(f"constraints: {constraints}\n")
                fout.write(f"free variables: {free_variables}\n")
                fout.write(f"transition rates: {transition_rates}\n")

            logging.info(asp_model._reactions_set)
            logging.info(asp_model.generate_ode_system())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
