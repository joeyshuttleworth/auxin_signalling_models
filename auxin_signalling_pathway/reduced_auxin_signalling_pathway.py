import numba as nb
import numpy as np
import sympy as sp
from numba import cfunc, njit
from numbalsoda import lsoda_sig
from sympy.printing.julia import julia_code

from auxin_signalling_pathway import auxin_signalling_pathway


class reduced_auxin_signalling_pathway(auxin_signalling_pathway):
    def __init__(self, nARFs=1, nIAAs=1, aux_func=None):
        super().__init__(nARFs, nIAAs, aux_func)

        subs_dict = {}

        for arf in self.arf_symbols:
            promoter_symbol = f"G__{arf}"
            self.remove_state(promoter_symbol)

            if promoter_symbol in self.promoter_symbols:
                self.promoter_symbols.remove(promoter_symbol)

        for rna_symbol, iaa_symbol in zip(self.rna_symbols,
                                          self.iaa_symbols):
            subs_dict[f"k__{rna_symbol}__{iaa_symbol}"] = "k__RNA__IAA"
            self.remove_transition_rate(f"k__{rna_symbol}__{iaa_symbol}")

        self.add_transition_rate(
            "k__RNA__IAA", 1.0, "Rate of IAA production from RNA")

        for key in self.transition_rates:
            if "bind" in key:
                subs_dict[key] = 0.0

        # Simplyfy the model by removing rates which are assumed to be 0
        for arf_symbol in self.arf_symbols:
            for iaa_symbol in self.iaa_symbols:
                for rna_symbol in self.rna_symbols:
                    subs_dict[f"k__G__{arf_symbol}__{iaa_symbol}__{rna_symbol}"] = "0.0"
                    self.remove_transition_rate(
                        f"k__G__{arf_symbol}__{iaa_symbol}__{rna_symbol}")

                    # subs_dict[f"k__G__{rna_symbol}"] = "0.0"
                    # self.remove_transition_rate(f"k__G__{rna_symbol}")

        self.parameter_subs_dict = subs_dict

    def generate_ode_system(self, reduced=True, do_QSS_sub=True, parameter_subs_dict=None,
                            **kwargs):
        """Generate a list of sympy expressions comprising an ODE system

        :returns: A tuple: the list of ODEs, a list of "free variables" e.g.
        the auxin concentration, a dictionary of parameters, and a list of
        constraints (sympy Expressions)

        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict

        odes, free_variables, transition_rates, constraints = \
            super().generate_ode_system(state_variables=super().get_state_variables(),
                                        parameter_subs_dict=parameter_subs_dict,
                                        **kwargs)
        if reduced:
            state_vars = super().get_state_variables()
            promoter_vars = [s for s in state_vars if s[0] == 'G']
            promoter_odes = sp.Matrix([[ode for state_var, ode in zip(state_vars, odes)
                                        if state_var in promoter_vars]]).T

            if do_QSS_sub:
                new_odes = []
                for state_var, ode in zip(state_vars, odes):
                    if state_var in promoter_vars:
                        new_odes.append(ode)
                        continue
                    free_symbols = set([str(s) for s in ode.free_symbols])
                    for s in [s for s in free_symbols if s
                              in self.QSS_bound_species]:
                        v = self.QSS_bound_species[s]
                        if state_var not in v:
                            ode = ode.subs({
                                s: sp.sympify(s)**2 / sum([sp.sympify(x)
                                                           for x in [s] + list(v)])
                            })
                            ode = ode.subs({s: 0.0 for s in v})
                    new_odes.append(ode)

                odes = sp.Matrix([new_odes]).T

            odes = odes.subs(parameter_subs_dict)

            promoter_odes = sp.Matrix([[promoter_odes]]).T
            include_indices = [i for i, state in enumerate(state_vars)
                               if state not in promoter_vars]
            odes = sp.Matrix(np.array(odes).flatten()[include_indices])
            if constraints:
                constraints = list(constraints) + [promoter_odes]
            else:
                constraints = [promoter_odes]

        return odes, free_variables, transition_rates, constraints

    def get_default_initial_conditions(self):
        all_ic = super().get_default_initial_conditions()
        state_vars = list(all_ic.keys())
        promoter_vars = [s for s in state_vars if s[0] == 'G']

        return {
            key: var for key, var in all_ic.items()
            if key not in promoter_vars
        }

    def get_state_variables(self):
        """ Construct a list of the model's state variables

        :returns: a list of the model's state variables, sorted in
        alphabetical order, and excluding the empty-set node, and the auxin
        concentration

        """
        state_vars = [s for s in sorted(self.graph.nodes()) if s[0] != 'G']
        return sorted(list([str(s) for s in state_vars
                            if s != self.empty_set_node
                            and s != self.aux_symbol]))

    def make_cfunc_rhs(self, aux_func=None, parameter_subs_dict=None):
        """Generate a c function which evaluates the RHS of the ODE system

        :returns: a c function with signature lsoda_sig for use solving our ODE
        system

        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict

        promoter_vars = [s for s in sorted(self.graph.nodes()) if s[0] == 'G']
        n_promoter_vars = len(promoter_vars)

        # Number of params + 'free variables' i.e. auxin
        n_data = len(self.get_default_parameters()) + 1
        rhs_func = self.generate_rhs_func(parameter_subs_dict=parameter_subs_dict,
                                          njitted=True)

        promoter_expr_jac_func, inhomog_expr_func = self.get_promoter_ode_funcs(
            parameter_subs_dict=parameter_subs_dict)

        odes, free_variables, transition_rates, constraints = \
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)
        n_y = len(odes)

        @cfunc(lsoda_sig)
        def cfunc_rhs(t, y, dy, data):
            y = nb.carray(y, n_y).astype(np.float64)
            dy = nb.carray(dy, n_y)
            data = nb.carray(data, n_data)

            if aux_func:
                aux = aux_func(t)
            else:
                aux = data[0]

            aux = np.array([aux])

            params = data[1:].copy().flatten().astype(np.float64)

            promoter_vals = np.full(n_promoter_vars, 0.0)

            # steady state doesn't depend on promoter states provided conservation
            promoter_expr_jac = promoter_expr_jac_func(
                y, params, aux).astype(np.float64)

            # if np.linalg.det(promoter_expr_jac @ promoter_expr_jac.T) > 1e-12\
            #    and np.all(np.isfinite(promoter_expr_jac)):

            inhomog_val = inhomog_expr_func(y, params, aux).astype(np.float64)
            promoter_vals = np.linalg.solve(
                promoter_expr_jac, -inhomog_val).flatten()
            G = 1.0 - np.sum(promoter_vals)
            promoter_vals = np.concatenate((np.array([G]), promoter_vals))

            res = rhs_func(y, params,
                           aux, promoter_vals).astype(np.float64).flatten()

            assert res.shape == y.shape
            dy[:] = res[:]
            return

        return cfunc_rhs

    def output_latex(self, **kwargs):
        return super().output_latex(reduced=True, **kwargs)

    def get_promoter_ode_funcs(self, parameter_subs_dict=None,
                               njitted=True):
        odes, free_variables, _, constraints = \
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)

        #  Assume the promoter expression is the only constraint

        promoter_vars = [s for s in sorted(self.graph.nodes()) if s[0] == 'G']

        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})

        promoter_expr_jac = self.simplify_expr(
            promoter_expr.jacobian(sp.Matrix([promoter_vars[1:]]).T))

        promoter_expr_inhomog = self.simplify_expr(promoter_expr - promoter_expr_jac
                                                   @ sp.Matrix([promoter_vars[1:]]).T)

        subs_dict = {p: '0' for p in self.promoter_symbols}

        #  We know that these expressions dont depend on the promoter variables
        #  so this is a cheap and easy way to simplify them
        promoter_expr_inhomog = promoter_expr_inhomog.subs(subs_dict)
        promoter_expr_jac = promoter_expr_jac.subs(subs_dict)

        parameter_vars = sorted(list(self.get_default_parameters().keys()))
        state_vars = sorted(self.get_state_variables())

        promoter_inputs = (state_vars, parameter_vars, free_variables)
        promoter_expr_jac_func = sp.lambdify(promoter_inputs,
                                             promoter_expr_jac)

        inhomog_expr_func = sp.lambdify(
            promoter_inputs, promoter_expr_inhomog)

        if njitted:
            promoter_expr_jac_func = njit(promoter_expr_jac_func)
            inhomog_expr_func = njit(inhomog_expr_func)

        return promoter_expr_jac_func, inhomog_expr_func

    def get_promoter_expr(self, parameter_subs_dict=None):
        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict

        odes, free_variables, transition_rates, constraints = \
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)

        promoter_vars = [s for s in super().get_state_variables()
                         if s[0] == 'G']
        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})

        promoter_expr_jac = sp.expand(
            promoter_expr.jacobian(sp.Matrix([promoter_vars[1:]]).T)
        )

        promoter_expr_inhomog = sp.expand(sp.Matrix(promoter_expr) -
                                          sp.Matrix(promoter_expr_jac)
                                          @
                                          sp.Matrix([promoter_vars[1:]]).T)

        subs_dict = {p: '0' for p in self.promoter_symbols}
        promoter_expr_inhomog = promoter_expr_inhomog.subs(subs_dict)
        promoter_expr_jac = promoter_expr_jac.subs(subs_dict)

        promoter_expr = -promoter_expr_jac.LUsolve(promoter_expr_inhomog)

        return self.simplify_expr(promoter_expr)

    def generate_rhs_func(self, njitted=True, parameter_subs_dict=None):
        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        state_variables = self.get_state_variables()

        param_names = list(transition_rates.keys())

        state_vars = super().get_state_variables()
        promoter_vars = [s for s in state_vars if s[0] == 'G']

        for s in odes.free_symbols:
            input_symbols = param_names + state_variables + promoter_vars + free_variables
            if str(s) not in input_symbols:
                raise ValueError(f"symbol {s} not found in {input_symbols}")

        _rhs_func = sp.lambdify((state_variables, param_names,
                                 free_variables, promoter_vars), odes)
        if njitted:
            _rhs_func = njit(_rhs_func)

        return _rhs_func

    def get_promoter_steady_state_expr(self, parameter_subs_dict=None):
        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        promoter_vars = [s for s in sorted(self.graph.nodes()) if s[0] == 'G']

        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})

        promoter_expr_jac = sp.Matrix(
            promoter_expr.jacobian(sp.Matrix([promoter_vars[1:]]).T)
        )

        promoter_expr_inhomog = self.simplify_expr(promoter_expr - promoter_expr_jac
                                                   @ sp.Matrix([promoter_vars[1:]]).T)
        promoter_ss_expr = -promoter_expr_jac.LUsolve(promoter_expr_inhomog)

        return promoter_ss_expr

    def get_jacobian_expr(self, parameter_subs_dict=None):
        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)
        promoter_vars = [s for s in sorted(self.graph.nodes()) if s[0] == 'G']
        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})
        state_vars = self.get_state_variables()

        promoter_ss_expr = self.get_promoter_steady_state_expr(
            parameter_subs_dict)

        sub_dict = {g: '0' for g in self.promoter_symbols}
        promoter_ss_expr = promoter_ss_expr.subs(sub_dict)
        promoter_ss_jac = self.simplify_expr(
            promoter_ss_expr.jacobian(state_vars)
        )

        jac = sp.Matrix(self.simplify_expr(odes.jacobian(self.get_state_variables())
                                           + odes.jacobian(sp.Matrix(promoter_vars[1:])) @ promoter_ss_jac))

        return jac

    def output_julia(self, aux_func_name="", aux_func_params=[]):
        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system()

        state_vars = self.get_state_variables()
        default_parameters = self.get_default_parameters()

        promoter_vars = [s for s in super().get_state_variables()
                         if s[0] == 'G']
        code_str = f"function {self.__class__.__name__}_f_deriv(z, p, t=0.0)\n"

        code_str += "\t(" + ", ".join(state_vars) + ") = z\n\n"

        if not aux_func_name:
            code_str += "\t(" + ", ".join(default_parameters.keys() + aux_func_params) + ", " + \
                f"{self.aux_symbol}) = p\n\n"
        else:
            code_str += "\t(" + \
                ", ".join(list(default_parameters.keys()) + aux_func_params)\
                + ") = p\n\n"
            code_str += f"\t{self.aux_symbol} = {aux_func_name}(t, " + ", ".join(aux_func_params) + ")\n\n"

        code_str += "(" + ", ".join(promoter_vars[1:]) + ") = " + \
            str(julia_code(self.get_promoter_expr())) + "\n\n"
        code_str += f"\t {promoter_vars[0]} = " \
            f"{1 - sum(self.get_promoter_expr())}\n\n"

        code_str += "\treturn " + str(julia_code(odes)) + "\n\n"
        code_str += "end"
        return code_str

    def generate_jacobian_func(self, njitted=False, parameter_subs_dict=None):
        jac_expr = self.get_jacobian_expr()

        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        state_names = self.get_state_variables()
        param_names = list(transition_rates.keys())

        inputs = (state_names, param_names, free_variables)

        jac_func = sp.lambdify(inputs, jac_expr)

        if njitted:
            njit(jac_func)

        return jac_func

    def generate_myokit_model(self):
        raise NotImplementedError(
            "Myokit output not yet available for reduced models")

    def simplify_expr(self, expr):
        # for g in self.promoter_symbols:
        #     expr = sp.collect(expr, g)
        return expr
