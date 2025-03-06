import sympy as sp

import numpy as np
import numba as nb
from numba import cfunc, njit
from numbalsoda import lsoda_sig

from auxin_signalling_pathway import auxin_signalling_pathway

class reduced_auxin_signalling_pathway(auxin_signalling_pathway):
    def __init__(self, nARFs=1, nIAAs=1, aux_func=None):
        super().__init__(nARFs, nIAAs, aux_func)

        subs_dict = {}

        for rna_symbol, iaa_symbol in zip(self.rna_symbols,
                                            self.iaa_symbols):
            subs_dict[f"k__{rna_symbol}__{iaa_symbol}"] = "k__RNA__IAA"

        for key, val in self.transition_rates.items():
            if "bind" in key:
                subs_dict[key] = 0.0

        for arf_symbol in self.arf_symbols:
            for iaa_symbol in self.iaa_symbols:
                for rna_symbol in self.rna_symbols:
                    subs_dict[f"k__G__{arf_symbol}__{iaa_symbol}__{rna_symbol}"] = 0.0

        self.QSS_bound_species = {arf: (f"G__{arf}",) for arf in self.arf_symbols} |\
            {dimer: (f"G__{dimer[3:]}",) for dimer in self.dimer_symbols
             if dimer.split('__')[1] in self.arf_symbols}

        self.parameter_subs_dict =  subs_dict


    def generate_ode_system(self, reduced=True, **kwargs):
        """Generate a list of sympy expressions comprising an ODE system

        :returns: A tuple: the list of ODEs, a list of "free variables" e.g.
        the auxin concentration, a dictionary of parameters, and a list of
        constraints (sympy Expressions)

        """
        odes, free_variables, transition_rates, constraints = \
            super().generate_ode_system(state_variables=super().get_state_variables(),
                                        **kwargs)

        if reduced:
            state_vars = super().get_state_variables()
            promoter_vars = [s for s in state_vars if s[0]=='G']
            promoter_odes = sp.Matrix([[ode for state_var, ode in zip(state_vars, odes)
                                        if str(state_var[0]) in promoter_vars]]).T
            promoter_odes = sp.Matrix([promoter_odes]).T
            include_indices = [i for i, state in enumerate(state_vars)
                               if state not in promoter_vars]
            print(odes)
            odes = sp.Matrix(np.array(odes).flatten()[include_indices])
            constraints += [promoter_odes]

            print(include_indices, state_vars)

        return odes, free_variables, transition_rates, constraints


    def get_default_initial_conditions(self):
        all_ic = super().get_default_initial_conditions()
        state_vars = list(all_ic.keys())
        promoter_vars = [s for s in state_vars if s[0]=='G']

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


    def make_cfunc_rhs(self, aux_func=None, parameter_subs_dict={}):
        """Generate a c function which evaluates the RHS of the ODE system

        :returns: a c function with signature lsoda_sig for use solving our ODE
        system

        """

        odes, free_variables, transition_rates, constraints = \
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)

        # Assume the promoter expression is the only constraint
        promoter_exprs = constraints[0]
        promoter_vars = [s for s in sorted(self.graph.nodes()) if s[0]=='G']
        n_promoter_vars = len(promoter_vars)

        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})

        promoter_expr_jac = promoter_expr.jacobian(sp.Matrix([promoter_vars[1:]]).T)

        # For complicated models, this simplification may take some time
        # TODO rewrite more efficiently
        promoter_expr_inhomog = sp.simplify(promoter_expr - promoter_expr_jac\
                                            @ sp.Matrix([promoter_vars[1:]]).T)

        parameter_vars = list(transition_rates.keys())

        state_vars = self.get_state_variables()

        promoter_inputs = (state_vars, parameter_vars, free_variables)
        promoter_expr_jac_func = njit(sp.lambdify(promoter_inputs, promoter_expr_jac))
        inhomog_expr_func = njit(sp.lambdify(promoter_inputs, promoter_expr_inhomog))

        n_y = len(self.get_state_variables())

        # Number of params + 'free variables' i.e. auxin
        n_data = len(self.get_default_parameters()) + 1
        rhs_func = self.get_rhs_func(parameter_subs_dict=parameter_subs_dict,
                                     njitted=True)

        promoter_vals = np.full(n_promoter_vars, 1.0 / n_promoter_vars)
        ic_vals = np.full(n_y, 1.0)
        parameter_vals = np.array([v[0] for v in transition_rates.values()])

        print(state_vars, odes)
        print(len(ic_vals), len(odes))

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

            promoter_vals = np.full(n_promoter_vars, 1.0 / n_promoter_vars)

            # # steady state doesn't depend on promoter states provided conservation
            # promoter_expr_jac = promoter_expr_jac_func(y, params, aux).astype(np.float64)

            # inhomog_val = inhomog_expr_func(y, params, aux).astype(np.float64)
            # promoter_vals = np.linalg.solve(promoter_expr_jac, inhomog_val).flatten()
            # G = 1.0 - np.sum(promoter_vals)
            # promoter_vals = np.concatenate((np.array([G]), promoter_vals))

            # TODO add variable to scale promoter vars
            # gscale = 1.0
            # promoter_vals = promoter_vals * gscale

            res = rhs_func(y, params,
                           aux, promoter_vals).astype(np.float64).flatten()

            assert res.shape == y.shape
            dy[:] = res[:]
            return

        return cfunc_rhs

    def generate_rhs_func(self, njitted=True, parameter_subs_dict=None):
        odes, free_variables, transition_rates, \
            _ = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                         reduced=True)

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        state_variables = self.get_state_variables()
        param_names = sp.Matrix(list(transition_rates.keys()))

        state_vars = super().get_state_variables()
        promoter_vars = [s for s in state_vars if s[0]=='G']

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])

        _rhs_func = sp.lambdify((state_variables, param_names,
                                     free_variables, promoter_vars), odes)
        if njitted:
            _rhs_func = njit(_rhs_func)

        return _rhs_func


    def generate_jacobian_func(self, njitted=False):
        raise NotImplementedError("")
