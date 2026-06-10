import logging
import os
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import sympy as sp

try:
    import myokit
except ImportError:
    myokit = None

from auxin_signalling_models.AuxinSignallingPathway import \
    AuxinSignallingPathway as asp_class


def setup_output_directory(dir_name, subdir_name):
    dirname = os.path.join(dir_name, subdir_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname


test_output_dir = setup_output_directory(
    'test_output', f"{Path(__file__).name.split('.')[0]}")

output_dir = test_output_dir

model_names = [f"model{i}" for i in range(15)]
logging.info("outputting to " + test_output_dir)

kwargs_list = [
    {'nARFs': 1, 'nIAAs': 1},
    {'nARFs': 2, 'nIAAs': 1},
    {'nARFs': 1, 'nIAAs': 2},
    {'nARFs': 2, 'nIAAs': 2},
    {'nARFs': 3, 'nIAAs': 1},
    {'nARFs': 4, 'nIAAs': 2},
]

kwargs_list = [kw | {"include_arf_transcription": False} for kw in kwargs_list] + \
    [kw | {"include_arf_transcription": True} for kw in kwargs_list]

test_models = [asp_class(**kwargs) for
               kwargs in kwargs_list]


class TestGeneratePathway:
    def test_graph_generation(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)
            asp_model.draw_graph()
            plt.savefig(os.path.join(output_dir, f"test_graph_{i}"))
            plt.clf()

            with open(os.path.join(output_dir,
                                   f"test_graph_{i}.txt"), 'w') as fout:
                fout.write(repr(asp_model.generate_ode_system()))

            logging.info(asp_model._reactions_set)
            logging.info(asp_model.generate_ode_system())

    def generate_solve_ode(self, asp_model, i):
        odes, free_variables, parameters, constraints = \
            asp_model.generate_ode_system()
        with open(os.path.join(output_dir,
                               f"test_graph_{i}.txt"), 'w') as fout:
            fout.write(f"ODEs: {odes}\n")
            fout.write(f"constraints: {constraints}\n")
            fout.write(f"free variables: {free_variables}\n")
            fout.write(f"transition rates: {parameters}\n")

        ic_dict = asp_model.get_default_initial_conditions()
        param_dict = asp_model.get_default_parameters()

        solver = asp_model.make_forward_solver()
        ts = np.linspace(0, 0.1, 2)
        res, success = solver(ts, param_dict, ic_dict, 0.99)

        return res, success

    def test_generate_solve_ode(self):
        for i, asp_model in enumerate(test_models):
            res, success = self.generate_solve_ode(asp_model, i)

        assert success and np.all(np.isfinite(res))

    def test_substitution(self):
        for _, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)
            subs_dict = {}

            for rna_symbol, iaa_symbol in zip(asp_model.rna_variables,
                                              asp_model.iaa_variables):
                subs_dict[f"k__{rna_symbol.name}__{iaa_symbol.name}"] = "k__RNA__IAA"

            for par in asp_model.get_parameters():
                if "bind" in par.name:
                    subs_dict[par.name] = 0.0

            for arf in asp_model.arf_variables:
                for iaa in asp_model.iaa_variables:
                    for rna in asp_model.rna_variables:
                        subs_dict[f"k__G__{arf.name}__{iaa.name}__{rna.name}"] = 0.0

            odes, free_variables, parameters, constraints \
                = asp_model.generate_ode_system(parameter_subs_dict=subs_dict)

            logging.info(f"Transition rates: {parameters}")
            for k, v in subs_dict.items():
                # If the value has been substituted with a constant (i.e. 0)
                # skip this check
                if sp.sympify(v).free_symbols:
                    assert v in list(parameters.keys())
                    assert sp.sympify(v) in odes.free_symbols

            logging.info(parameters)

    def test_output_latex(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            latex_out = asp_model.output_latex()

            with open(os.path.join(output_dir, f"full_model_{i}.tex"), 'w') as fout:
                fout.write(latex_out)

    def test_myokit_output(self):
        pytest.importorskip("myokit")
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            myokit_model = asp_model.generate_myokit_model()

            model_name = asp_model.get_model_name()

            myokit.save(os.path.join(output_dir, f"full_model_{model_name}.mmt"),
                        myokit_model)

    def test_output_julia(self):
        for i, kwargs in enumerate(kwargs_list[:3]):
            asp_model = asp_class(**kwargs)
            jl_out = asp_model.output_julia()

            with open(os.path.join(output_dir,
                                   f"model_{i}.jl"), 'w') as fout:
                fout.write(jl_out)

    def test_get_model_name(self):
        for kwargs in kwargs_list:
            asp_model = asp_class(**kwargs)
            name = asp_model.get_model_name()
            assert isinstance(name, str)

    def test_print_parameters(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            pretty_str = asp_model.generate_pretty_print_parameter_dict()
            assert isinstance(pretty_str, dict)

            with open(os.path.join(output_dir, f"full_model_{i}_params.txt"), 'w') as fout:
                fout.write(str(pretty_str))

    def test_print_states(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            pretty_str = asp_model.generate_pretty_print_state_dict()
            assert isinstance(pretty_str, dict)

            with open(os.path.join(output_dir, f"full_model_{i}_states.txt"), 'w') as fout:
                fout.write(str(pretty_str))

    def test_exception_bad_parameter_label(self):
        asp_model = asp_class(**kwargs_list[0])
        with pytest.raises(ValueError):
            asp_model.add_model_parameter(
                "1/////this is a bad parameter parameter")
        asp_model = asp_class(**kwargs_list[0])

    def test_get_parameter_info(self):
        for kwargs in kwargs_list:
            asp_model = asp_class(**kwargs)

            param_info, state_info = asp_model.get_variable_info()

            param_info.to_csv(os.path.join(
                output_dir, f"{asp_model.get_model_name()}_param_info.csv"))
            state_info.to_csv(os.path.join(
                output_dir, f"{asp_model.get_model_name()}_state_info.csv"))

            assert len(param_info.index) == len(
                asp_model.get_parameter_names())
            assert len(state_info.index) == len(
                asp_model.get_state_variable_names())

    def generate_state_var_graph(self):
        asp_model = asp_class(**kwargs_list[0])
        # state_vars = asp_model.get_state_variables()

        if not os.environ.get("DISPLAY"):
            # Mock Tk so that no actual window is created
            tk.Tk = MagicMock()
            tk.Toplevel = MagicMock()
            tk.messagebox = MagicMock()

        graph = asp_model.generate_state_variable_graph()
        nx.draw_graph(graph)
        plt.savefig(os.path.join(output_dir, "state_var_graph"))
