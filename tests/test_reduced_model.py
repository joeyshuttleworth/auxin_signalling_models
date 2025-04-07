import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from auxin_signalling_pathway import auxin_signalling_pathway as asp_class
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
            {'nARFs': 3, 'nIAAs': 3},
        ]

    def test_graph_generation(self):
        for i, kwargs in enumerate(self.kwargs_list):
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

        fig = plt.figure()
        ax = fig.subplots()
        for i, kwargs in enumerate(self.kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            res, success = self.generate_solve_ode(asp_model, i)

            res = res / res.max(axis=1)[:, None]
            ts = np.linspace(0, 1, res.shape[0])
            ax.plot(ts, res, label=asp_model.get_state_variables())
            ax.legend()
            fig.savefig(os.path.join(self.output_dir, f"solution_{i}"))
            ax.cla()

        self.assertTrue(success and np.all(np.isfinite(res)))
        plt.close(fig)

    def test_reduced_model_validation(self):
        for kwargs in self.kwargs_list:
            asp_model_r = reduced_asp_class(**kwargs)
            asp_model_full = asp_class(**kwargs)
            asp_model_full.add_transition_rate(
                "k__RNA__IAA", 1.0, 'Production rate for IAAs')

            parameter_subs_dict = asp_model_r.parameter_subs_dict.copy()

            reduced_ic_dict = asp_model_r.get_default_initial_conditions()

            solver_r = asp_model_r.make_forward_solver()

            solver_full = asp_model_full.make_forward_solver(
                parameter_subs_dict=parameter_subs_dict)

            ts = np.linspace(0, 100, 250)
            param_dict = asp_model_r.get_default_parameters()
            res, succ = solver_r(ts, param_dict, reduced_ic_dict, 1.0)
            self.assertTrue(succ and np.all(np.isfinite(res)))

            full_params = asp_model_full.get_default_parameters(
                parameter_subs_dict=parameter_subs_dict)
            full_ic_dict = asp_model_full.get_default_initial_conditions()
            res, succ = solver_full(
                ts, full_params, full_ic_dict, 1.0, strict=False)

    def test_output_latex(self):
        for i, kwargs in enumerate(self.kwargs_list):
            asp_model_r = reduced_asp_class(**kwargs)

            latex_out = asp_model_r.output_latex()

            with open(os.path.join(self.output_dir, f"reduced_model_{i}.tex"), 'w') as fout:
                fout.write(latex_out)

    def test_output_julia(self):
        for i, kwargs in enumerate(self.kwargs_list[:3]):
            asp_model_r = reduced_asp_class(**kwargs)

            jl_out = asp_model_r.output_julia()

            with open(os.path.join(self.output_dir, f"reduced_model_{i}.jl"), 'w') as fout:
                fout.write(jl_out)

    def test_get_jacobian(self):
        #  TODO Check answer against reference jacobian
        for i, kwargs in enumerate(self.kwargs_list[:2]):
            asp_model_r = reduced_asp_class(**kwargs)
            asp_model_r.get_jacobian_expr()

        self.assertTrue(True)


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.INFO)
    unittest.main()
