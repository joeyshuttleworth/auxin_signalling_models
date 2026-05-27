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

            odes = odes.subs(parameter_subs_dict)

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

    def get_promoter_ode_funcs(self, parameter_subs_dict=None):
        """Construct two functions used to compute the solution to the
        promoter-protein component of the model under a quasi steady-state
        assumption.

        :param parameter_subs_dict: A dictionary of parameter substitutions

        :return: 2-tuple - an expression for the jacobian of the derivative function, and an expression for the inhomogenous part
        """

        odes, free_variables, _, constraints =\
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)

        #  Assume the promoter expression is the only constraint

        promoter_vars = sorted([s.name for s in self.get_state_variables()
                                if s.category == "promoter"])

        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})

        promoter_expr_jac = self.simplify_expr(
            promoter_expr.jacobian(sp.Matrix([promoter_vars[1:]]).T))

        promoter_expr_inhomog = self.simplify_expr(promoter_expr - promoter_expr_jac
                                                   @ sp.Matrix([promoter_vars[1:]]).T)

        subs_dict = {p: '0' for p in [s.name for s in self.promoter_variables]}

        #  We know that these expressions dont depend on the promoter variables
        #  so this is a cheap and easy way to simplify them
        promoter_expr_inhomog = promoter_expr_inhomog.subs(subs_dict)
        promoter_expr_jac = promoter_expr_jac.subs(subs_dict)

        parameter_vars = sorted(list(self.get_default_parameters().keys()))
        state_vars = sorted(self.get_state_variable_names(
            include_promoter_vars=False))

        promoter_inputs = (state_vars, parameter_vars, free_variables)

        promoter_expr_jac_func = sp.lambdify(promoter_inputs,
                                             promoter_expr_jac)

        inhomog_expr_func = sp.lambdify(
            promoter_inputs, promoter_expr_inhomog)

        return promoter_expr_jac_func, inhomog_expr_func

    def get_promoter_expr(self, parameter_subs_dict=None, use_cse=True):
        """Generate a sympy expression for the time-derivative of the promoter
        component of the model under a quasi steady-state assumption.

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
            self.generate_ode_system(reduced=True,
                                     parameter_subs_dict=parameter_subs_dict)

        promoter_vars = sorted([s.name for s in super().get_state_variables()
                                if s.category == "promoter"])

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

        subs_dict = {p: '0' for p in [s.name for s in self.promoter_variables]}
        promoter_expr_inhomog = promoter_expr_inhomog.subs(subs_dict)
        promoter_expr_jac = promoter_expr_jac.subs(subs_dict)

        promoter_expr = -promoter_expr_jac.LUsolve(promoter_expr_inhomog)

        if use_cse:
            replacements, expressions = sp.cse(promoter_expr)
            return replacements, expressions

        return promoter_expr

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
            s.name for s in state_vars if s.category == "promoter"]

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

        # steady state doesn't depend on promoter states provided conservation
        promoter_expr_jac_func, inhomog_expr_func\
            = self.get_promoter_ode_funcs(parameter_subs_dict=parameter_subs_dict)

        def rhs_func(y, t, p, aux):
            promoter_expr_jac = promoter_expr_jac_func(y, p, aux)\
                .astype(np.float64)
            inhomog_val = inhomog_expr_func(y, p, aux).astype(np.float64)
            promoter_vals = np.linalg.solve(
                promoter_expr_jac, -inhomog_val).flatten()
            G = 1.0 - np.sum(promoter_vals)
            promoter_vals = np.concatenate((np.array([G]), promoter_vals))

            return np.array(_rhs_func(y, t, p, aux, promoter_vals),
                            dtype=np.float64).flatten()

        return rhs_func

    def get_promoter_state_variables(self, *args, **kwargs):
        """List and sort the promoter state variables found in the model.

        :return: Sorted list of promoter state variables
        """

        all_states = self.get_state_variables(*args, **kwargs)
        return [s for s in all_states if s.category == "promoter"]

    def get_promoter_steady_state_expr(self, parameter_subs_dict=None):
        """Make a sympy expression giving quasi steady state promoter variables
        given values for all other state variables.

        :param parameter_subs_dict: A dictionary of parameter
            substitutions
        :return: A sympy expression for the promoter quasi steady state
        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(parameter_subs_dict=parameter_subs_dict,
                                                   reduced=True)

        promoter_vars = sorted([s.name for s in self.get_state_variables()
                                if s.category == "promoter"])

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

        promoter_exprs = constraints[0]
        promoter_expr = sp.Matrix([promoter_exprs[1:]]).T
        promoter_expr = promoter_expr.subs({'G': 1 - sum([sp.sympify(s)
                                                          for s in promoter_vars[1:]])})
        state_vars = sorted(self.get_state_variable_names(
            include_promoter_vars=False))

        promoter_ss_expr = self.get_promoter_steady_state_expr(
            parameter_subs_dict)

        sub_dict = {g: '0' for g in [s.name for s in
                                     self.promoter_variables]}
        promoter_ss_expr = promoter_ss_expr.subs(sub_dict)
        promoter_ss_jac = self.simplify_expr(
            promoter_ss_expr.jacobian(state_vars)
        )

        jac = sp.Matrix(self.simplify_expr(odes.jacobian(state_vars)
                                           + odes.jacobian(sp.Matrix(promoter_vars[1:])) @ promoter_ss_jac))

        return jac

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
        default_parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

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
        replacements, expressions = self.get_promoter_expr(use_cse=True)
        for r, expr in replacements:
            code_str += f"\t{r} = {julia_code(expr)}" + "\n"

        code_str += "\t(" + ", ".join(promoter_vars[1:]) + ") = " +\
            julia_code(expressions[0]) + "\n\n"

        code_str += f"\t {promoter_vars[0]} = " \
            f"1 - ({' + '.join(promoter_vars[1:])})\n\n"

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
