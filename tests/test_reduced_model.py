import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auxin_signalling_models import AuxinSignallingPathway as asp_class
from auxin_signalling_models.ReducedAuxinSignallingPathway import \
    ReducedAuxinSignallingPathway as reduced_asp_class

this_dir = Path(__file__).parent
golden_dir = os.path.join(this_dir.resolve(), "test_output_golden")

test_output_dir = os.path.join(
    "test_output", f"{Path(__file__).name.split('.')[0]}")

os.makedirs(test_output_dir, exist_ok=True)

output_dir = test_output_dir

model_names = [f"model{i}" for i in range(15)]
logging.info("outputting to %s", test_output_dir)

kwargs_list = [
    {'nARFs': 1, 'nIAAs': 1, 'include_arf_transcription': False},
    {'nARFs': 1, 'nIAAs': 2, 'include_arf_transcription': False},
]

# Add transcription flag to kwarg list
arg_list_with_arf = [{**d, 'include_arf_transcription': True}
                     for d in kwargs_list]

kwargs_list = [item for pair in zip(kwargs_list, arg_list_with_arf)
               for item in pair]


class TestReducedModel:
    def test_graph_generation(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            asp_model.draw_graph()
            plt.savefig(os.path.join(output_dir, f"test_graph_{i}"))
            plt.clf()

            with open(os.path.join(output_dir,
                                   f"test_graph_{i}.txt"), 'w') as fout:
                fout.write(repr(asp_model.generate_ode_system()))

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
        ic_dict = {k: 1.0 for k in ic_dict}

        param_dict = asp_model.get_default_parameters()
        param_dict = {k: 1.0 for k in param_dict}

        solver = asp_model.make_forward_solver()
        ts = np.linspace(0, 0.1, 2)
        res, success = solver(ts, param_dict, ic_dict, 0.99)

        return res, success

    def test_promoter_ss_func(self):
        model = reduced_asp_class()
        vars, promoter_ss_func = \
            model.get_promoter_steady_state_func()

        state_vars = sorted(model.get_state_variable_names(include_promoter_vars=False))

        params = sorted(model.get_default_parameters().keys())

        x = [1.0 for s in state_vars]

        p = []

        assert(len(vars) == no_promoter_vars)
        vec = promoter_ss_func(x, p, [1.0])

        assert(len(vec) == len(vars))


    def test_multi_promoter_model(self):
        asp_model = reduced_asp_class()

        asp_model.add_state("G2", category="promoter")
        asp_model.add_state("G2__ARF_1", category="promoter")
        asp_model.add_reaction(["G2"], ["G2__ARF_1"], [0.1], [1])

        solver = asp_model.make_forward_solver()
        ts = np.linspace(0, 0.1, 2)

        param_dict = asp_model.get_default_parameters()
        param_dict = {k: 1.0 for k in param_dict}

        ic_dict = asp_model.get_default_initial_conditions()
        ic_dict = {k: 1.0 for k in ic_dict}

        res, success = solver(ts, param_dict, ic_dict, 0.99)

        assert success

    def test_generate_solve_ode(self, request):
        ignore_golden = request.config.getoption("ignore_golden")

        fig = plt.figure()
        ax = fig.subplots()
        for i, kwargs in enumerate(kwargs_list):
            asp_model = reduced_asp_class(**kwargs)
            res, success = self.generate_solve_ode(asp_model, i)

            ts = np.linspace(0, 1, res.shape[0])
            ax.plot(ts, res,
                    label=asp_model.get_state_variable_names(include_promoter_vars=False))
            ax.legend()
            fig.savefig(os.path.join(output_dir, f"solution_{i}"))
            ax.cla()

            txt_savename = f"solution_{i}.txt"
            np.savetxt(os.path.join(output_dir, txt_savename),
                       res, fmt="%.18e")

            if not ignore_golden:
                # Compare against golden version
                golden_file = os.path.join(golden_dir, Path(output_dir).name,
                                           txt_savename)

                assert os.path.exists(
                    golden_file), f"Generated {txt_savename} but it doesn't exist in test_output_golden: {golden_file}"

                golden_res = np.genfromtxt(golden_file, delimiter=' ')
                assert np.mean(
                    (golden_res.flatten() - res.flatten())**2) <= 1e-8

        assert success and np.all(np.isfinite(res))
        plt.close(fig)

    def test_reduced_model_validation(self):
        for kwargs in kwargs_list:
            asp_model_r = reduced_asp_class(**kwargs)
            asp_model_full = asp_class(**kwargs)
            asp_model_full.add_model_parameter(
                "k__RNA__IAA", 1.0, 'Production rate for IAAs')
            ts = np.linspace(0, 0.1, 2)
            param_dict = asp_model_r.get_default_parameters()

            parameter_subs_dict = asp_model_r.parameter_subs_dict.copy()

            reduced_ic_dict = asp_model_r.get_default_initial_conditions()

            solver_r = asp_model_r.make_forward_solver()

            solver_full = asp_model_full.make_forward_solver(
                parameter_subs_dict=parameter_subs_dict)

            ts = np.linspace(0, 0.1, 2)
            param_dict = asp_model_r.get_default_parameters()
            res, succ = solver_r(ts, param_dict, reduced_ic_dict, 1.0)
            assert succ and np.all(np.isfinite(res))

            full_params = asp_model_full.get_default_parameters(
                parameter_subs_dict=parameter_subs_dict)
            full_ic_dict = asp_model_full.get_default_initial_conditions()
            res, succ = solver_full(
                ts, full_params, full_ic_dict, 1.0, strict=False)

    def test_model_with_arf_transcription(self, request):
        ignore_golden = request.config.getoption("ignore_golden")

        for i, _kwargs in enumerate(kwargs_list[:2]):
            kwargs = _kwargs.copy()
            kwargs['include_arf_transcription'] = True
            asp_model_r = reduced_asp_class(**kwargs)
            solver_r = asp_model_r.make_forward_solver()
            ts = np.linspace(0, 0.1, 2)

            ic_dict = asp_model_r.get_default_initial_conditions()
            ic_dict = {k: 1.0 for k in ic_dict}

            param_dict = asp_model_r.get_default_parameters()
            param_dict = {k: 1.0 for k in param_dict}

            res, succ = solver_r(ts, param_dict, ic_dict, 1.0, strict=False)
            assert succ and np.all(np.isfinite(res))

            save_fname = f"test_model_with_arf_transcription_{i}.txt"
            np.savetxt(os.path.join(output_dir, save_fname), res, fmt="%.18e")

            if not ignore_golden:
                # Compare against golden version
                golden_file = os.path.join(golden_dir, Path(output_dir).name,
                                           save_fname)

                golden_res = np.genfromtxt(golden_file, delimiter=' ')
                assert np.mean(
                    (golden_res.flatten() - res.flatten())**2) < 1e-8

    def test_output_latex(self, request):
        ignore_golden = request.config.getoption("ignore_golden")

        for i, kwargs in enumerate(kwargs_list):
            asp_model_r = reduced_asp_class(**kwargs)

            latex_out = asp_model_r.output_latex()
            save_fname = f"reduced_model_{i}.tex"

            with open(os.path.join(output_dir, save_fname), 'w') as fout:
                fout.write(latex_out)

            if not ignore_golden:
                # Compare against golden version
                golden_file = os.path.join(golden_dir, Path(output_dir).name,
                                           save_fname)

                with open(golden_file, "r") as fin:
                    latex_str = fin.read()

                assert latex_str == latex_out

    def test_output_julia(self, request):
        ignore_golden = request.config.getoption("ignore_golden")
        for i, kwargs in enumerate(kwargs_list):
            asp_model_r = reduced_asp_class(**kwargs)

            jl_out = asp_model_r.output_julia()

            save_fname = f"reduced_model_{i}.jl"
            with open(os.path.join(output_dir, save_fname), 'w') as fout:
                fout.write(jl_out)

            if not ignore_golden:
                # Compare against golden version
                golden_file = os.path.join(golden_dir, Path(output_dir).name,
                                           save_fname)

                with open(golden_file, "r") as fin:
                    jl_str = fin.read()
                    assert jl_str == jl_out

    def test_get_promoter_state_variables(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model_r = reduced_asp_class(**kwargs)

            promoter_vars = asp_model_r.get_promoter_state_variables()

            for v in promoter_vars:
                assert v.category == "promoter"

            assert len(promoter_vars) == len(set(promoter_vars))

        return

    def test_get_jacobian(self):
        #  TODO Check answer against reference jacobian
        for i, kwargs in enumerate(kwargs_list[:2]):
            asp_model_r = reduced_asp_class(**kwargs)
            asp_model_r.get_jacobian_expr()

    def test_print_parameters(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            pretty_str = asp_model.generate_pretty_print_parameter_dict()
            assert isinstance(pretty_str, dict)

            with open(os.path.join(output_dir, f"reduced_model_{i}_params.txt"), 'w') as fout:
                fout.write(str(pretty_str))

    def test_print_states(self):
        for i, kwargs in enumerate(kwargs_list):
            asp_model = asp_class(**kwargs)

            pretty_str = asp_model.generate_pretty_print_state_dict()
            assert isinstance(pretty_str, dict)

            with open(os.path.join(output_dir, f"reduced_model_{i}_states.txt"), 'w') as fout:
                fout.write(str(pretty_str))
