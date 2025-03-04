import logging
import os
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp

from auxin_signalling_pathway.reduced_auxin_signalling_pathway import \
    reduced_auxin_signalling_pathway as reduced_asp_class


def setup_output_directory(dir_name, subdir_name):
    dirname = os.path.join(dir_name, subdir_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname


class TestReducedModel(unittest.TestCase):

    def setUp(self):
        test_output_dir = setup_output_directory(
            'test_output', f"{type(self).__name__}")
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

        self.kwargs_list = [
            {'nARFs': 1, 'nIAAs': 1},
            {'nARFs': 2, 'nIAAs': 1},
            {'nARFs': 1, 'nIAAs': 2},
            {'nARFs': 4, 'nIAAs': 4}
        ]

    def test_graph_generation(self):
        kwargs_list = [
            {'nARFs': 1, 'nIAAs': 1},
            {'nARFs': 2, 'nIAAs': 1},
            {'nARFs': 1, 'nIAAs': 2},
            {'nARFs': 4, 'nIAAs': 4}
        ]

        for i, kwargs in enumerate(kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            # logging.info(nx.write_network_text(asp_model.graph))
            asp_model.draw_graph()
            plt.savefig(os.path.join(self.output_dir, f"test_graph_{i}"))
            plt.clf()

            with open(os.path.join(self.output_dir,
                                   f"test_graph_{i}.txt"), 'w') as fout:
                fout.write(repr(asp_model.generate_ode_system()))

    def generate_solve_ode(self, asp_model, i):
        odes, free_variables, transition_rates, constraints = \
            asp_model.generate_ode_system()
        with open(os.path.join(self.output_dir,
                               f"test_graph_{i}.txt"), 'w') as fout:
            fout.write(f"ODEs: {odes}\n")
            fout.write(f"constraints: {constraints}\n")
            fout.write(f"free variables: {free_variables}\n")
            fout.write(f"transition rates: {transition_rates}\n")

        ic_dict = asp_model.get_default_initial_conditions()
        param_dict = asp_model.get_default_parameters()

        solver = asp_model.make_forward_solver()
        ts = np.linspace(0, 1000, 1000)
        res, success = solver(ts, param_dict, ic_dict, 0.99)

        return res, success

    def test_generate_solve_ode(self):
        for i, kwargs in enumerate(self.kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            res, success = self.generate_solve_ode(asp_model, i)

            res = res / res.max(axis=1)[:, None]
            ts = np.linspace(0, 1, res.shape[0])
            plt.plot(ts, res)
            plt.savefig(os.path.join(self.output_dir, f"solution_{i}"))
            plt.clf()

        self.assertTrue(success and np.all(np.isfinite(res)))

    def test_substitution(self):
        for _, kwargs in enumerate(self.kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            subs_dict = {}

            for rna_symbol, iaa_symbol in zip(asp_model.rna_symbols,
                                              asp_model.iaa_symbols):
                subs_dict[f"k__{rna_symbol}__{iaa_symbol}"] = "k__RNA__IAA"

            for key, val in asp_model.transition_rates.items():
                if "bind" in key:
                    subs_dict[key] = 0.0

            for arf_symbol in asp_model.arf_symbols:
                for iaa_symbol in asp_model.iaa_symbols:
                    for rna_symbol in asp_model.rna_symbols:
                        subs_dict[f"k__G__{arf_symbol}__{iaa_symbol}__{rna_symbol}"] = 0.0

            odes, free_variables, transition_rates, constraints = asp_model.generate_ode_system(parameter_subs_dict=subs_dict)

            for k, v in subs_dict.items():
                self.assertIn(v, list(transition_rates.keys()))

                # If the value has been substituted with a constant (i.e. 0)
                # skip this check
                if sp.sympify(v).free_symbols:
                    self.assertIn(sp.sympify(v), odes.free_symbols)



if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.INFO)
    unittest.main()
