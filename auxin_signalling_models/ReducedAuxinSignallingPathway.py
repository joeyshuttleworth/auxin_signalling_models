from typing import Callable, List

import numpy as np
import sympy as sp
from sympy.printing.julia import julia_code

from .AuxinSignallingPathway import AuxinSignallingPathway


class ReducedAuxinSignallingPathway(AuxinSignallingPathway):
    def __init__(self, nARFs: int = 1, nIAAs: int = 1,
                 aux_func: Callable[[float], float] = None, simple_promoter_states: bool = False,
                 **kwargs):
        """A simplified version of a auxin_signalling_pathway model, with
        promoter-protein dynamics simplified with a quasi steady-state
        assumption.

        Additionally, all ARF translation rates are assumed to be equal, as are Aux/IAA translation rates. Optionally, the model can be further simplified by removing states and reactions where the promoter is bound to a monomer

        :param nARFs: The number of distinct categories of ARF genes to include in the model
        :param nIAAs: The number of distinct categories of Aux/IAA genes to include in the model
        :param aux_func: Optional time-dependent auxin-signal input function of the form f(t)
        :simple_promoter_states: If true, remove states where the promoter is bound to monoer a - only allow dimer bounding
        """

        super().__init__(nARFs=nARFs, nIAAs=nIAAs, aux_func=aux_func, **kwargs)

        subs_dict = {}

        for state in [s for s in self.get_state_variables()
                      if s.category == "promoter" and len(s.components) > 0]:

            if state.components[0] in [s.name for s in self.arf_variables]:
                self.remove_state(state)

                # Remove related parameters from model
                assoc_rate = "k__" + state.name
                dissoc_rate = "d__" + state.name

                self.remove_parameter(assoc_rate)
                self.remove_parameter(dissoc_rate)

        if simple_promoter_states:
            for promoter in self.promoter_variables[:]:
                components = promoter.components

                if components[0].category == "dimer":
                    dimer = components[0]

                    if len(dimer.components) != 2:
                        raise RuntimeError(f"Invalid dimer variable: {dimer}")

                    if dimer.components[0] != dimer.components[1]:
                        self.remove_state_variable_by_name(promoter)

        for iaa_var in self.iaa_variables:
            iaa_symbol = iaa_var.name
            rna_symbol = f"R__{iaa_symbol}"
            subs_dict[f"k__{rna_symbol}__{iaa_symbol}"] = "k__RNA__IAA"

        if self.include_arf_transcription:
            for arf_var in self.arf_variables:
                arf_symbol = arf_var.name
                rna_symbol = f"R__{arf_symbol}"
                subs_dict[f"k__{rna_symbol}__{arf_symbol}"] = "k__RNA__ARF"

        self.add_model_parameter(
            "k__RNA__IAA",
            1.0,
            "Rate of IAA production from RNA",
            "production")

        if self.include_arf_transcription:
            self.add_model_parameter(
                "k__RNA__ARF",
                1.0,
                "Rate of ARF production from RNA",
                "production"
            )

        for key in list(self.get_parameter_names()):
            if "bind" in key:
                self.remove_parameter(key)

        self.parameter_subs_dict = subs_dict

    def generate_ode_system(self, reduced: bool = True,
                            parameter_subs_dict: List = None, **kwargs):
        """Generate a list of sympy expressions comprising an ODE system.

        :param parameter_subs_dict: a dictionary of parameter
            substitutions
        :param reduced: If true, return a system using a quasi steady-
            state assumption for protein-promoter dynamics
        :param do_QSS_sub: If True, simplify the model using the
            promoter-protein quasi steady-state assumption
        :returns: A tuple: the list of ODEs, a list of "free variables"
            e.g. the auxin concentration, a dictionary of parameters,
            and a list of constraints (sympy Expressions)
        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        odes, free_variables, parameters, constraints =\
            super().generate_ode_system(state_variables=super().get_state_variable_names(),
                                        parameter_subs_dict=parameter_subs_dict,
                                        **kwargs)

        if reduced:
            state_vars = super().get_state_variable_names(include_promoter_vars=True)
            promoter_vars = [
                s.name for s in self.get_state_variables() if s.category == "promoter"]
            promoter_odes = sp.Matrix([[ode for state_var, ode in zip(state_vars, odes)
                                        if state_var in promoter_vars]]).T

            odes = odes.subs(parameter_subs_dict).subs(parameter_subs_dict)

            promoter_odes = sp.Matrix([[promoter_odes]]).T
            include_indices = [i for i, state in enumerate(state_vars)
                               if state not in promoter_vars]
            odes = sp.Matrix(np.array(odes).flatten()[include_indices])
            if constraints:
                constraints = list(constraints) + [promoter_odes]
            else:
                constraints = [promoter_odes]

        return odes, free_variables, parameters, constraints

    def get_default_initial_conditions(self):
        """Create a dictionary of state variables and corresponding initial
        conditions.

        :returns: a dictionary with a key for each state variable
            corresponding to various default initial conidtions
        """

        all_ic = super().get_default_initial_conditions()
        promoter_var_labels = [s.name for s in self.get_state_variables()
                               if s.category == "promoter"]

        ret_dict = {
            key: var for key, var in all_ic.items()
            if key not in promoter_var_labels
        }

        return {k: ret_dict[k] for k in sorted(list(ret_dict.keys()))}

    def get_state_variables(self, include_promoter_vars: bool = True,
                            *args, **kwargs):
        """Construct a list of the model's state variables.

        :returns: a list of the model's state variables
        """

        state_vars = super().get_state_variables(*args, **kwargs)

        if not include_promoter_vars:
            state_vars = [s for s in state_vars if s.category != "promoter"]

        return sorted(state_vars)

    def output_latex(self, **kwargs):
        """Construct a LaTeX string describing the model mathematically.

        :returns: A string containg a LaTeX equations for each of the
            models state variables
        """

        return super().output_latex(reduced=True, **kwargs)

    def generate_promoter_steady_state_func(self, parameter_subs_dict=None):
        """
        Generate a function to compute steady states for the QSS promoter component

        :param parameter_subs_dict: Dictionary of parameter substitutions to use

        :return: A function mapping a state-variable vector, parameter vector and auxiliary variables to a vector of steady-state values
        """

        promoter_vars_included, all_expressions = self.get_promoter_steady_state_exprs(
            parameter_subs_dict=parameter_subs_dict, use_cse=False)

        promoter_var_labels = promoter_vars_included[0]
        subs_dict = {promoter_var_labels[0]:
                     1 - sum([sp.sympify(s) for s in promoter_var_labels[1:]])}

        state_labels = sorted(self.get_state_variable_names(
            include_promoter_vars=False))
        parameter_labels = sorted(self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict).keys())

        print(all_expressions[0][1])
        expr = all_expressions[0][1].subs({k: 0 for k in promoter_var_labels})

        _dim = len(expr) + 1

        print(expr)

        inputs = (state_labels, parameter_labels, [self.auxin_variable_label])
        _func = sp.lambdify(inputs, expr)

        if len(all_expressions) != 1:
            raise NotImpementedError(
                f"Not implemented for models with less/more than one component: {len(all_expressions)} components present")

        def ret_func(x, p, aux, _dim=_dim):
            vec = np.zeros(_dim)
            vec[1:] = _func(x, p, aux).flatten()
            vec[0] = 1 - vec[1:].sum()

            return vec.flatten()

        return promoter_vars_included, ret_func

    def get_promoter_steady_state_exprs(self, parameter_subs_dict=None, use_cse=True):
        """Generate a sympy expression for the steady-state of the promoter
        component (fast) of the model under a quasi steady-state assumption.

        :param parameter_subs_dict: A dictionary of parameter
            substitutions
        :param use_cse: If True, use sympy common sub-expressions to
            simplify the returned expressions
        :return: A sympy expression for promoter dynamics (ODE), or if
            use_cse is True, equivalent lists of substitutions and
            expressions
        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict

        odes, free_variables, parameters, constraints =\
            self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                     reduced=True)

        all_state_vars = sorted(list(s for s in self.get_state_variables()))

        promoter_vars = sorted([s.name for s in all_state_vars
                                if s.category == "promoter"])

        # TODO use constraints field to find variables to simplify out

        promoter_expr = constraints[0].T

        connectivity_matrix = sp.Matrix(
            [promoter_expr]).T.jacobian(sp.Matrix([promoter_vars]))

        connectivity_matrix =\
            np.array([[connectivity_matrix[i, j]
                       for j in range(connectivity_matrix.shape[1])]
                      for i in range(connectivity_matrix.shape[0])]).astype(bool)

        connectivity_matrix &= connectivity_matrix.T

        connected_components = _get_connected_components(connectivity_matrix)

        if use_cse:
            all_replacements = []
            all_expressions = []
        else:
            all_expressions = []

        promoter_vars_included = []
        full_promoter_vars = promoter_vars
        full_promoter_expr = promoter_expr

        for component in connected_components:
            promoter_vars = [v for i, v in enumerate(full_promoter_vars)
                             if i in component]

            promoter_var_vec = sp.Matrix(promoter_vars[1:])

            promoter_expr = full_promoter_expr[list(component), :]

            sub_promoter_expr = sp.Matrix([promoter_expr[1:, :]])

            sub_promoter_expr = \
                sub_promoter_expr.subs(
                    {promoter_vars[0]: 1 - sum([sp.sympify(s)
                                                for s in promoter_vars[1:]])})

            if len(promoter_var_vec) >= 1:
                promoter_expr_jac = sp.expand(
                    sub_promoter_expr.jacobian(promoter_var_vec)
                )
            else:
                raise ValueError(
                    f"Single disconnected promoter state found {promoter_vars[0]}")

            promoter_expr_inhomog = sp.expand(sp.Matrix(sub_promoter_expr) -
                                              sp.Matrix(promoter_expr_jac)
                                              @
                                              promoter_var_vec)

            subs_dict = {s: '0' for s in full_promoter_vars}

            promoter_expr_inhomog = promoter_expr_inhomog.subs(subs_dict)
            promoter_expr_jac = promoter_expr_jac.subs(subs_dict)

            sub_promoter_expr = - \
                promoter_expr_jac.LUsolve(promoter_expr_inhomog)

            if use_cse:
                replacements, expressions = sp.cse(sub_promoter_expr)
                all_replacements.append(replacements)
                expressions = [sp.sympify(
                    1 - sum(promoter_var_vec))] + expressions
                all_expressions.append(expressions)
                promoter_vars_included.append(promoter_vars)

            else:
                expressions = [sp.sympify(
                    1 - sum(promoter_var_vec))] + [sub_promoter_expr]
                all_expressions.append(expressions)
                promoter_vars_included.append(promoter_vars)

        if use_cse:
            return promoter_vars_included, all_replacements, all_expressions
        else:
            return promoter_vars_included, all_expressions

    def generate_rhs_func(self, parameter_subs_dict=None,
                          compute_QSS_values=True):
        """Generate a derivative function for the model.

        Note that the returned function treats promoter variables as a separate
        argument.

        :param parameter_subs_dict: A dictionary of parameter
            substitutions
        :returns: A derivative function for the model with parameters
            for the models state variables, parameters, free variables,
            and promoter states

        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        param_names = list(parameters.keys())
        state_vars = super().get_state_variables()
        promoter_var_labels = [
            s.name for s in state_vars if s.category == "promoter"
        ]

        state_var_labels = self.get_state_variable_names(
            include_promoter_vars=False)

        for s in odes.free_symbols:
            input_symbols = param_names + state_var_labels + \
                promoter_var_labels + free_variables
            if str(s) not in input_symbols:
                raise ValueError(f"symbol {s} not found in {input_symbols}")

        t_sym = sp.sympify("_t")

        _rhs_func = sp.lambdify((state_var_labels, t_sym, param_names,
                                 free_variables, promoter_var_labels), odes)
        if not compute_QSS_values:
            return _rhs_func

        _, promoter_exprs =\
            self.get_promoter_steady_state_exprs(parameter_subs_dict=parameter_subs_dict,
                                                 use_cse=False)

        funcs = []
        promoter_inputs = (state_var_labels, param_names, free_variables)
        for expr in promoter_exprs:
            expr[1] = expr[1].subs({s: 0 for s in promoter_var_labels})

            _func = sp.lambdify((promoter_inputs), expr[1])
            _dim = len(expr[1]) + 1

            def func(x, p, aux, _func=_func, _dim=_dim):
                vals = np.zeros(_dim)
                vals[1:] = _func(x, p, aux).flatten()
                vals[0] = 1 - vals[1:].sum()
                return vals

            funcs.append(func)

        def rhs_func(y, t, p, aux):
            promoter_vals = np.concatenate([f(y, p, aux)
                                            for f in funcs])
            return np.array(_rhs_func(y, t, p, aux, promoter_vals),
                            dtype=np.float64).flatten()

        return rhs_func

    def get_promoter_state_variables(self, *args, **kwargs):
        """List and sort the promoter state variables found in the model.

        :return: Sorted list of promoter state variables
        """

        all_states = self.get_state_variables(*args, **kwargs)
        return [s for s in all_states if s.category == "promoter"]

    def get_jacobian_expr(self, parameter_subs_dict=None):
        """Construct a sympy expression for the jacobian of the modes'
        derivative function (excluding promoter variables).

        :return: A sympy Matrix for the jacobian of the models
            derivative
        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)
        promoter_vars = sorted([s.name for s in self.get_state_variables()
                                if s.category == "promoter"])

        state_vars = sorted(self.get_state_variable_names(
            include_promoter_vars=False))

        # Start with non QSS terms
        jac = sp.Matrix(self.simplify_expr(odes.jacobian(state_vars)))

        promoter_vars_list, expressions_list = \
            self.get_promoter_steady_state_exprs(parameter_subs_dict=parameter_subs_dict,
                                                 use_cse=False)

        # Loop over each strongly-connected QSS component, adding additional terms
        for promoter_vars, expressions in zip(promoter_vars_list, expressions_list):
            promoter_expr = expressions[1]
            promoter_expr = promoter_expr.subs({'G': 1.0 - sum([sp.sympify(s)
                                                                for s in promoter_vars[1:]])})

            sub_dict = {g: '0' for g in promoter_vars}
            promoter_expr = promoter_expr.subs(sub_dict)

            promoter_ss_jac = self.simplify_expr(
                promoter_expr.jacobian(state_vars)
            )

            jac += odes.jacobian(
                sp.Matrix(promoter_vars[1:])) @ promoter_ss_jac

        return jac

    def get_default_parameters(self, parameter_subs_dict: dict = None,
                               omit_unused: bool = False):
        """Return a default parameter set.

        :param parameter_subs_dict: Dictionary of parameter substitutions to apply
        :param omit_unused: If True, remove parameters that don't appear in model equations
        :return: a dictionary of parameter values
        """

        dict = super().get_default_parameters(parameter_subs_dict=parameter_subs_dict,
                                              omit_unused=False)

        if omit_unused:
            odes, free_variables, _, constraints\
                = self.generate_ode_system(reduced=True,
                                           parameter_subs_dict=parameter_subs_dict)

            promoter_odes = constraints[0]

            dict = {
                p: v for p, v in dict.items()
                if p in [str(s) for s in odes.free_symbols]
                or p in [str(s) for s in promoter_odes.free_symbols]
                or p in [str(s) for s in free_variables]
            }

        return dict

    def output_julia(self, aux_func_name: str = "", aux_func_params: List = [],
                     parameter_subs_dict: dict = None):
        """Generate julia code describing the model including the quasi steady-
            state assumption.

            :param aux_func_name: A julia function name for an externally
                defined auxin-input signal
            :param aux_func_params: Optional parameters to include for the
                auxin input function
            :param parameter_subs_dict: A dictionary of parameter
                substitutions
            :returns: A string contianing a julia function definition
            """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(
                parameter_subs_dict=parameter_subs_dict)

        state_vars = self.get_state_variable_names(include_promoter_vars=False)
        odes, vars = list(zip(*[[ode, var] for ode, var in zip(odes, state_vars)
                                if "promoter" not in var]))

        default_parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict,
            omit_unused=True
        )

        promoter_vars = [s.name for s in super().get_state_variables()
                         if s.category == "promoter"]
        code_str = f"function {self.__class__.__name__}_f_deriv!(dz, z, p, t=0.0)\n"

        code_str += "\t(" + ", ".join(state_vars) + ") = z\n\n"

        if not aux_func_name:
            code_str += "\t(" + ", ".join(list(default_parameters.keys()) + aux_func_params) + ", " +\
                f"{self.auxin_variable_label}) = p\n\n"
        else:
            code_str += "\t(" +\
                ", ".join(sorted(list(default_parameters.keys())) + aux_func_params)\
                + ") = p\n\n"
            code_str += f"\t{self.auxin_variable_label} = {aux_func_name}(t, " + ", ".join(
                aux_func_params) + ")\n\n"

        # Use common sub expressions to simplify function
        # This is especially important if you're going to do auto-diff in julia
        promoter_subsets, all_replacements, all_expressions =\
            self.get_promoter_steady_state_exprs(use_cse=True)

        if parameter_subs_dict is None:
            parameter_subs_dict = self.get_parameter_substitution_dict()

        for i, (promoter_vars, replacements, expressions) in \
                enumerate(zip(promoter_subsets, all_replacements, all_expressions)):
            dummy_vars = [v for (v, _) in replacements]

            # Substitute in some extra characters so we don't shadow vars
            dummy_sub_dict = {v: f"__{i}__{v}" for v in dummy_vars}
            for r, expr in replacements:
                code_str += f"\t{dummy_sub_dict[r]} =" +\
                    f"{julia_code(expr.subs(dummy_sub_dict).subs(parameter_subs_dict))}" + "\n"

            code_str += "\t(" + ", ".join(promoter_vars[1:]) + ") = " +\
                julia_code(expressions[1].subs(dummy_sub_dict).subs(parameter_subs_dict)) + \
                "\n\n"

            code_str += f"\t {promoter_vars[0]} = " +\
                julia_code(expressions[0].subs(dummy_sub_dict).subs(
                    parameter_subs_dict)) + "\n\n"

        code_str += "\tdz .= (" + str(julia_code(odes))[1:-1] + ")\n\n"
        code_str += "end"

        return code_str

    def generate_jacobian_func(self, parameter_subs_dict: dict = None):
        """ Define a function which computes the jacobian of the dynamical system defined by the model

        :param parameter_subs_dict: A dict providing a list of parameter substituions to make

        :return: A function returning a jacobian taking a 3-tuple, state-variable values, parameter values, and other variables, as parameters
        """

        jac_expr = self.get_jacobian_expr()

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        all_state_vars = self.get_state_variables()
        state_vars = [v for v in all_state_vars if v.category != "promoter"]
        state_names = [v.name for v in state_vars]

        param_names = list(parameters.keys())

        t_sym = sp.sympify("_t")
        inputs = (state_names, t_sym, param_names, free_variables)

        jac_func = sp.lambdify(inputs, jac_expr, modules=["numpy"])

        return jac_func

    def generate_myokit_model(self, *args, **kwargs):
        """Generate myokit code for the model.

        This method is unimplemented

        :raises: NotImplementedError
        """
        raise NotImplementedError(
            "Myokit output not yet available for reduced models"
        )

    def simplify_expr(self, expr):
        """Simplify a given expression found in the model.

        A method for expression simplification may be implemented in
        future

        :param expr: The expression to be simplified
        :return: The same expression
        """

        return expr


def _get_connected_components(mat):
    """
    Compute strongly-connected components of a matrix.

    Matrix is explored from column 0 downwards.

    :param: mat a adjacency matrix for a graph

    :returns: A list of connected components (a list of sets of integers).
    """

    assert mat.shape[0] == mat.shape[1]
    N = mat.shape[0]

    assert N > 0

    unexplored = list(range(N))

    node = unexplored[0]

    components = []
    this_component = []

    def get_next_node(m, connected_nodes, unexplored_nodes):
        for i in set(unexplored_nodes):
            for j in set(connected_nodes):
                if m[i, j] and m[j, i]:
                    return i
        return None

    while node is not None:
        unexplored.remove(node)
        this_component.append(node)
        node = get_next_node(mat, this_component, unexplored)

        if node is None:
            components.append(set(this_component))
            if len(unexplored) > 0:
                node = unexplored[0]
            this_component = []

    return components
