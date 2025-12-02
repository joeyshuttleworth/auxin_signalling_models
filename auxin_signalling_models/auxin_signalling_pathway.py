import logging
from typing import Callable, List

import myokit
import networkx as nx
import numba as nb
import numpy as np
import seaborn as sns
import sympy as sp
from numba import cfunc, njit
from numbalsoda import lsoda, lsoda_sig
from pyvis.network import Network
from sympy.printing.julia import julia_code

from .reaction import Reaction


class auxin_signalling_pathway():
    def __init__(self, nARFs: int = 1, nIAAs: int = 1, include_arf_transcription: bool = False,
                 aux_func: Callable[[float], float] = None):
        """
        A graph-based representation of an auxin-signalling pathway in terms of chemical reactions and chemical species

        The model is described in terms of:
        - auxin_signalling_pathway.graph where the nodes are various protein complexes, hormones and promoter states, and the edges are Reaction objects describing a particular chemical reaction

        - A dictionary mapping parameters to descriptions and default values (2-tuple)

        - An optional parameter substitution dictionary which allows the definition of hyperpparameters (for example)

        :param nARFs: The number of distinct categories of ARF genes to include in the model
        :param nIAAs: The number of distinct categories of Aux/IAA genes to include in the model
        :param include_arf_transcription: If true, explicitly model ARF transcription using the models protein-promoter dynamics
        :param aux_func: Optional time-dependent auxin-signal input function of the form f(t)
        """

        self.include_arf_transcription = include_arf_transcription
        self.QSS_bound_species = {}
        self.parameter_subs_dict = {}

        # Dictionary storing tuples: label => description and default values
        self.parameters = {}

        self.nARFs = nARFs
        self.nIAAs = nIAAs

        self.graph = nx.Graph()

        self._reactions_set = set()

        self.empty_set_node = '__phi'

        self.add_state(self.empty_set_node, category='None')

        self.strict = True

        self.parameter_sub_dict = {}

        #  Initialise parameter symbols
        arf_symbols = [f"ARF_{i + 1}" for i in range(nARFs)]

        for arf in arf_symbols:
            self.add_state(arf, category='ARF')

        iaa_symbols = [f"iaa_{i + 1}" for i in range(nIAAs)]
        for iaa in iaa_symbols:
            self.add_state(iaa, category='IAA')

        self.promoter_symbols = ["G"] + [f"G__{arf1}__{arf2}"
                                         for arf1 in arf_symbols
                                         for arf2 in arf_symbols if arf2 >= arf1]\
            + [f"G__{arf}" for arf in arf_symbols]\
            + [f"G__{arf}__{iaa}" for arf in arf_symbols
               for iaa in iaa_symbols]
        self.promoter_symbols = list(set(self.promoter_symbols))

        for promoter in self.promoter_symbols:
            self.add_state(promoter, category='promoter')

        dimer_symbols = [f"D__{s1}__{s2}"
                         for s1 in arf_symbols + iaa_symbols
                         for s2 in arf_symbols + iaa_symbols if s2 >= s1]
        self.dimer_symbols = dimer_symbols

        for dimer in dimer_symbols:
            self.add_state(dimer, category='dimer')

        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]

        if include_arf_transcription:
            rna_symbols += [f"R__{arf}" for arf in arf_symbols]

        self.rna_symbols = rna_symbols

        for rna in rna_symbols:
            self.add_state(rna, category='rna')

        self.aux_symbol = 'x'
        self.add_state(self.aux_symbol, category='None')

        self.all_state_vars = sorted(arf_symbols + iaa_symbols
                                     + self.promoter_symbols + dimer_symbols
                                     + rna_symbols)

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_decay = sp.sympify('d__arf__all')
        self.add_model_parameter(arf_decay, 1.0,
                                 "decay rate of ARFs")

        for _, arf in enumerate(arf_symbols):
            self.add_reaction([], [arf], fwd_rates=[],
                              bwd_rates=[arf_decay])

        if not include_arf_transcription:
            arf_production = sp.sympify('k__arf__all')
            self.add_model_parameter(arf_production, 1.0,
                                     "production rate of ARFs")

            for _, arf in enumerate(arf_symbols):
                self.add_reaction([], [arf], fwd_rates=[
                    arf_production], bwd_rates=[])

        # iaa_production = sp.sympify('k__iaa__all')
        iaa_decay = sp.sympify('d__iaa__all')

        self.add_model_parameter(iaa_decay, 1.0,
                                 "decay rate of Aux/IAAs")

        for _, iaa in enumerate(iaa_symbols):
            self.add_reaction([], [iaa], bwd_rates=[iaa_decay])

        for d in dimer_symbols:
            association_rate = sp.sympify(f"k__{d}")
            dissociation_rate = sp.sympify(f"d__{d}")

            self.add_model_parameter(association_rate,
                                     0.001,
                                     f"Association rate of dimer {d}")
            self.add_model_parameter(dissociation_rate,
                                     0.1,
                                     f"Dissociation rate of dimer {d}")

            reactants = str(d).split('__')[1:]
            self.add_reaction(reactants, [d], [association_rate],
                              [dissociation_rate])

        for iaa in iaa_symbols:
            iaa_decay_rate = f"d__{iaa}__{self.aux_symbol}"
            self.add_model_parameter(iaa_decay_rate, 0.1,
                                     "rate of auxin-mediated decay of Aux/IAA")
            self.add_reaction([iaa], [self.empty_set_node],
                              [iaa_decay_rate],
                              enzymes=[self.aux_symbol])

        for promoter in self.promoter_symbols:
            if promoter == 'G':
                continue

            # TODO clean up this code
            reactant = promoter[3:]
            if '__' in reactant:
                reactant = f"D__{reactant}"

            promoter_binding_rate = f"k__{promoter}"
            promoter_unbinding_rate = f"d__{promoter}"

            self.add_model_parameter(promoter_binding_rate,
                                     1.0,
                                     f"binding rate for promoter {promoter}")
            self.add_model_parameter(promoter_unbinding_rate,
                                     1.0,
                                     "dissociation rate for promoter"
                                     f"{promoter}")

            self.add_reaction(
                [sp.sympify('G')], [promoter],
                [promoter_binding_rate], [],
                enzymes=[reactant]
            )

            self.add_reaction(
                [sp.sympify('G')], [promoter],
                [], [promoter_unbinding_rate],
                enzymes=[]
            )

        for promoter in self.promoter_symbols:
            for rna_symbol in rna_symbols:
                rate = f"k__{promoter}__{rna_symbol}"

                self.add_model_parameter(rate,
                                         1.0,
                                         f"Rate of transcription "
                                         f"for {rna_symbol} for promoter "
                                         f"{promoter}"
                                         )

                self.add_reaction(
                    [], [rna_symbol], [rate],
                    [], [promoter], reaction_type="linear"
                )

        for promoter in self.promoter_symbols:
            if promoter == 'G':
                continue

            reactants = sorted(promoter.split('__')[1:])

            if len(reactants) == 2:
                # TODO use different transition rates
                partially_bound_promoter = f"G__{reactants[0]}"

                if partially_bound_promoter in self.promoter_symbols:
                    promoter_binding_rate = f"k__{partially_bound_promoter}__bind__{reactants[1]}"
                    promoter_unbinding_rate = f"d__{promoter}__unbind__{reactants[1]}"

                    if promoter_binding_rate not in self.parameters:
                        self.add_model_parameter(promoter_binding_rate)
                        self.add_model_parameter(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [promoter_binding_rate], [],
                            enzymes=[reactants[1]])

                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [], [promoter_unbinding_rate]
                        )

                partially_bound_promoter = f"G__{reactants[1]}"
                if partially_bound_promoter in self.promoter_symbols:
                    promoter_binding_rate = f"k__{partially_bound_promoter}__bind__{reactants[0]}"
                    promoter_unbinding_rate = f"d__{promoter}__unbind__{reactants[0]}"

                    if promoter_binding_rate not in self.parameters:
                        self.add_model_parameter(promoter_binding_rate)
                        self.add_model_parameter(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [promoter_binding_rate], [],
                            enzymes=[reactants[0]])

                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [promoter_unbinding_rate]
                        )

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        self.add_model_parameter(rna_decay_rate,
                                 0.01,
                                 "RNA decay rate")

        for rna_symbol in rna_symbols:
            self.add_reaction([self.empty_set_node], [
                rna_symbol], [], [rna_decay_rate])
            prot = rna_symbol.split('__')[1]

            production_rate = f"k__{rna_symbol}__{prot}"
            rate_default_val = 0.0

            self.add_model_parameter(production_rate, rate_default_val,
                                     f"Rate of production of {prot} from"
                                     f"RNA {rna_symbol}")

            self.add_reaction([], [prot],
                              fwd_rates=[production_rate],
                              enzymes=[rna_symbol])

        self.iaa_symbols = iaa_symbols
        self.arf_symbols = arf_symbols

        self.state_var_eqns = state_var_eqns

        self.rhs_func = None
        self.cfunc_rhs = None

        self.jac_func = None

    def add_model_parameter(self, label: str, default_value: float = 0.0,
                            description: str = "", strict: bool = True):
        """
        Add a new parameter to the model

        :param label: Label to use for the new parameter. Should be a valid Sympy symbol
        :param default_value: Default value for the newly added parameter
        :param description: Short description of the new model parameter
        :param strict: If False, overwrite an existing parameter of the same name

        :raises ValueError: When strict is True and a parameter with the same label already exists
        :raises ValueError: When label isn't a valid Python variable identifier

        """

        label = str(label)
        logging.debug(f"Adding label {label}")

        # Check if label can be converted to a sympy Symbol
        if not label.isidentifier():
            raise ValueError(
                f"label:{label} is not a valid variable identifier")

        if strict is None:
            strict = self.strict

        if strict and label in self.parameters:
            raise ValueError(f"label {label} already present "
                             f"in model: {self.parameters}")

        self.parameters[label] = (default_value, description)

    def add_state(self, label, category=None, strict=True):
        """
        Add a state variable to the model e.g. a monomer, dimer or other chemical species.

        :param label: Label for the new state variable
        :param category: Optional category field for the new state variable e.g ARF, Aux/IAA, or promoter
        :param strict: If True an exception is thrown when a state variable of the same name already exists in the model

        :raises ValueError: When strict is True and a duplicate state-variable is added
        """

        label = str(label)

        if strict and label in self.graph.nodes():
            raise ValueError(
                f"Tried to add state {label} to the model but it already exists (strict=True)")

        self.graph.add_node(label, category=category)

    def add_reaction(self, reactants=[], products=[], fwd_rates=[],
                     bwd_rates=[], enzymes=[], add_states=False, strict=None,
                     reaction_type='linear'):
        """Add a chemical reaction to the model. Reactions are stored in self.graph

        :param reactants: the reactants of the reaction
        :param products: the products of the reaction (cannot be empty or None)
        :param fwd_rates: parameters associated with the forwards reaction
        (reactants -> products)
        :param bwd_rates: parameters associated with the reverse
        reaction (products->reactants)
        :param enzymes: other state variables that contribute
        to the reaction
        but aren't consumed
        :param add_states: boolean flag, if true missing
        states are added to the model
        :param strict: Boolean flag, if true, an exception is thrown
        if any of the state variables are missing from the model

        @return None

        """

        reactants = tuple([str(r) for r in reactants])
        products = tuple([str(p) for p in products])
        fwd_rates = tuple([str(r) for r in fwd_rates])
        bwd_rates = tuple([str(p) for p in bwd_rates])
        enzymes = tuple([str(e) for e in enzymes])

        reaction = Reaction(reactants, products,
                            fwd_rates, bwd_rates, enzymes,
                            reaction_type=reaction_type)

        self._reactions_set.add(
            reaction
        )

        if strict is None:
            strict = self.strict

        logging.debug(
            f"adding reaction from {reactants} to {products}\
            with rates {fwd_rates} and {bwd_rates}"
        )

        if reactants is None:
            reactants = []

        if products is None:
            products = []

        if len(reactants) + len(products) == 0:
            raise Exception("")

        if len(products) == 0:
            raise Exception("")

        if add_states:
            for state in set(reactants + products + enzymes):
                if state not in self.graph.nodes:
                    self.add_state(state)
        elif strict:
            for state in set(reactants + products + enzymes):
                if state not in self.graph.nodes:
                    raise Exception(f"Tried to add reaction with state {state} "
                                    f"but {state} does not exist in the model"
                                    "(strict=True)")

        for rate in bwd_rates + fwd_rates:
            if rate not in self.parameters:
                if strict:
                    raise Exception(f"rate {rate} not found in model. "
                                    f"model has:\n{self.parameters}")
                else:
                    self.add_model_parameter(rate)

        if len(fwd_rates) == 1:
            bwd_rates = bwd_rates * len(reactants)

        if len(bwd_rates) == 1:
            fwd_rates = fwd_rates * len(products)

        bwd_rates = [str(sym) for sym in bwd_rates]
        fwd_rates = [str(sym) for sym in fwd_rates]

        if len(reactants) == 0:
            reactants = (self.empty_set_node,)

        for fwd_rate, reactant in zip(fwd_rates, reactants):
            for product in products:
                self.graph.add_edge(
                    reactant, product, weight=fwd_rate,
                    reaction=reaction.__dict__)

        for bwd_rate, reactant in zip(bwd_rates, reactants):
            for product in products:
                self.graph.add_edge(
                    product, reactant, weight=bwd_rate,
                    reaction=reaction.__dict__)

        for enzyme in enzymes:
            if fwd_rates:
                for product in products:
                    self.graph.add_edge(
                        enzyme, product, weight=fwd_rates[0],
                        reaction=reaction.__dict__,
                        color='grey',
                        category='enzymatic'
                    )

            if bwd_rates:
                for reactant in reactants:
                    self.graph.add_edge(
                        enzyme, reactant, weight=bwd_rates[0],
                        reaction=reaction.__dict__,
                        color='grey',
                        category='enzymatic'
                    )

    def _get_reaction_terms(self, reaction, variable):
        """Get a term for the ODE formulation of a model from a particular reaction

        This function is used to generate an ODE system describing the
        time-evoluation of the model

        :param reaction: a Reaction describing the reaction of interest
        :param variable: the state variable for which terms are being generated

        :returns: a list of sympy expressions resulting from the reaction

        """
        terms = [sp.sympify('0')]

        if variable in reaction.enzymes:
            return [0]

        if variable not in reaction.reactants + reaction.products:
            raise Exception("")

        if variable in reaction.reactants and variable in reaction.products:
            raise Exception("")

        if reaction.reaction_type == 'linear':
            if variable in reaction.products:
                terms += self._get_reaction_terms(
                    reaction.reversed(), variable)

            fwd_rate = sp.sympify(
                reaction.fwd_rates[0]) if reaction.fwd_rates else 0
            bwd_rate = sp.sympify(
                reaction.bwd_rates[0]) if reaction.bwd_rates else 0

            if variable in reaction.reactants:
                terms.append(-sp.prod([sp.sympify(reactant)
                                       for reactant in reaction.reactants])
                             * fwd_rate)
                terms.append(sp.prod([sp.sympify(product)
                                      for product in reaction.products])
                             * bwd_rate)

                if reaction.enzymes and [t for t in terms if t != 0]:
                    terms = [t * sp.prod([sp.sympify(x)
                                          for x in reaction.enzymes])
                             for t in terms]
                elif reaction.enzymes:
                    terms = [-sp.prod([sp.sympify(x) for x in reaction.enzymes])
                             * fwd_rate,
                             sp.prod([sp.sympify(x) for x in reaction.enzymes])
                             * bwd_rate]
            return terms
        else:
            raise NotImplementedError("")

    def get_reactions_set(self):
        """
        Get a list of all of the chemical reactions defined in the model

        Chemical reactions are stored as auxin_signalling_models.Reaction objects

        :returns: reaction_set
        """

        return self._reactions_set

    def generate_ode_system(self, parameter_subs_dict: dict = None,
                            state_variables: List = None):
        """Generate a list of sympy expressions comprising an ODE system

        :params parameter_subs_dict: a dictionary of parameter substitutions
        :param state_variables: Optional list of state variables defining the ordering of state variables in the model

        :returns: A tuple: the list of ODEs, a list of "free variables" e.g.
        the auxin concentration, a dictionary of parameters, and a list of
        constraints (sympy Expressions)

        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        free_variables = [self.aux_symbol]

        if state_variables is None:
            state_variables = self.get_state_variables()

        odes = [sp.sympify('0') for _ in state_variables]

        for reaction in self._reactions_set:
            for variable in reaction.reactants + reaction.products:
                if str(variable) not in state_variables:
                    continue

                var_index = state_variables.index(variable)
                new_terms = self._get_reaction_terms(reaction, variable)

                odes[var_index] += sp.Add(*new_terms)

        sub_dict = {sp.sympify(self.empty_set_node): 1.0}
        odes = sp.Matrix([odes]).T.subs(sub_dict)

        # TODO include the fact that the promoter variables sum to 1
        constraints = []

        if parameter_subs_dict:
            odes = odes.subs(parameter_subs_dict)

        parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

        return odes, free_variables, parameters, constraints

    def get_state_variables(self, parameter_subs_dict=None):
        """ Construct a list of the model's state variables

        :returns: a list of the model's state variables, sorted in
        alphabetical order, and excluding the empty-set node, and the auxin
        concentration

        """
        state_vars = list([str(s) for s in self.graph.nodes()
                           if s != self.empty_set_node
                           and s != self.aux_symbol])

        if parameter_subs_dict:
            state_vars = [parameter_subs_dict[s] if s in parameter_subs_dict else s
                          for s in state_vars]

        return sorted(state_vars)

    def make_forward_solver(self, parameter_subs_dict=None):
        """Generate a function which simulates the model

        :returns: a function returning model output under different times,
        parameter values, initial conditions and auxin concentrations

        """

        cfunc = self.make_cfunc_rhs(parameter_subs_dict=parameter_subs_dict)
        cfunc_ptr = cfunc.address

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        def solver(ts, params_dict, ic_dict, aux, strict=True):
            default_params_dict = self.get_default_parameters(
                parameter_subs_dict=parameter_subs_dict)
            default_ic_dict = self.get_default_initial_conditions()

            params_dict = _sorted_dict(params_dict)
            ic_dict = _sorted_dict(ic_dict)

            if strict:
                if keys := [k for k in default_params_dict if k not in params_dict]:
                    raise Exception(
                        f"Parameter dictionary didnt contain keys: {keys}")
                if keys := [k for k in default_ic_dict if k not in ic_dict]:
                    raise Exception(
                        f"IC dictionary didnt contain keys: {keys}")

            for k in params_dict:
                if k not in default_params_dict:
                    raise ValueError(f"specified rate {k} not found in model:"
                                     f" {default_params_dict.keys()}")
            for k in ic_dict:
                if k not in default_ic_dict:
                    raise ValueError(f"specified state {k} not found in model:"
                                     f" {default_ic_dict.keys()}")

            params = np.array([
                params_dict[key] if key in params_dict
                else default_params_dict[key]
                for key in default_params_dict
            ])

            ic = np.array([
                ic_dict[key] if key in ic_dict else default_ic_dict[key]
                for key in default_ic_dict
            ])

            ts = np.array(ts)

            data = np.full(len(params) + 1, np.nan)
            data[0] = aux
            data[1:] = params.copy().flatten()

            return lsoda(cfunc_ptr, ic.copy().astype(np.float64),
                         np.array(ts),
                         data=data.astype(np.float64))

        return solver

    def get_cfunc_rhs(self, **kwargs):
        if self.cfunc_rhs is None and not kwargs:
            self.cfunc = self.make_cfunc_rhs()
        elif kwargs:
            return self.make_cfunc_rhs()

        return self.cfunc_rhs

    def make_cfunc_rhs(self, aux_func=None, parameter_subs_dict=None):
        """Generate a c function which evaluates the RHS of the ODE system

        :returns: a c function with signature lsoda_sig for use solving our ODE
        system

        """

        n_y = len(self.get_state_variables())

        # Number of params + 'free variables' i.e. auxin
        n_data = len(self.get_default_parameters(parameter_subs_dict=parameter_subs_dict))\
            + 1
        rhs_func = self.generate_rhs_func(parameter_subs_dict=parameter_subs_dict,
                                          njitted=True)

        @cfunc(lsoda_sig)
        def cfunc_rhs(t, y, dy, data):
            y = nb.carray(y, n_y).astype(np.float64)
            dy = nb.carray(dy, n_y)
            data = nb.carray(data, n_data)

            params = data[1:].copy().flatten().astype(np.float64)

            if aux_func:
                aux_func(t)
            else:
                aux = data[0:1]

            res = rhs_func(y, params,
                           aux).astype(np.float64).flatten().flatten()

            assert (res.shape == y.shape)
            dy[:] = res[:]
            return

        return cfunc_rhs

    def generate_jacobian_func(self, njitted=True):
        odes, free_variables, _, \
            _ = self.generate_ode_system()

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        jacobian = odes.jacobian(self.get_state_variables())

        ic_dict = self.get_default_initial_conditions()

        state_variables = sp.Matrix([s for s in ic_dict])
        params_dict = self.get_default_parameters()
        param_names = sp.Matrix(list(params_dict.keys()))

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])

        _jac_func = sp.lambdify((state_variables, param_names,
                                 free_variables), jacobian)
        _jac_func = njit(_jac_func)

        def jac_func(y, p, x):
            ret = _jac_func(y, p, x)
            return ret

        if njitted:
            jac_func = njit(jac_func)

        return jac_func

    def get_jacobian_func(self):
        if self.jac_func:
            return self.jac_func
        else:
            return self.generate_jacobian_func()

    def generate_rhs_func(self, njitted=True, parameter_subs_dict=None):
        odes, free_variables, parameters, \
            _ = self.generate_ode_system(
                parameter_subs_dict=parameter_subs_dict)

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        ic_dict = self.get_default_initial_conditions()

        state_variables = sp.Matrix([s for s in ic_dict])

        default_parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

        param_names = [sp.sympify(p) for p in default_parameters.keys()]

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])
        rhs_func = sp.lambdify((state_variables, param_names,
                                free_variables), odes)

        if njitted:
            rhs_func = njit(rhs_func)

        return rhs_func

    def get_default_initial_conditions(self, parameter_subs_dict=None):
        """
        Create a dictionary of state variables and corresponding initial conditions

        :param parameter_subs_dict: Optinal dictionary of parameter substitutions to perform

        :returns: a dictionary with a key for each state variable corresponding
        to various default initial conidtions

        """

        state_vars = self.get_state_variables(
            parameter_subs_dict=parameter_subs_dict)
        ic_dict = {str(s): 0.1 for s in state_vars if s != self.empty_set_node
                   and s != self.aux_symbol}

        promoter_vars = [s for s in ic_dict if s[0] == 'G']
        for v in promoter_vars:
            ic_dict[v] = 0.0
        ic_dict['G'] = 1.0

        for arf in self.arf_symbols:
            ic_dict[arf] = 1.0

        ic_dict = _sorted_dict(ic_dict)

        return ic_dict

    def get_default_parameters(self, parameter_subs_dict=None):
        """ Return a default parameter set

        :return: a dictionary of parameter values
        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        param_dict = {parameter_subs_dict[k] if k in parameter_subs_dict
                      and len(sp.sympify(parameter_subs_dict[k]).free_symbols) == 1
                      else k:
                      v[0]
                      for k, v in self.parameters.items()}

        # sort parameters
        param_dict = _sorted_dict(param_dict)

        return param_dict

    def draw_graph(self, output_path=None, **kwargs):
        """ Output a graph of the model using networkx and pygraphvis

        :param output_path: an optional path for outputting the diagram
        """

        if 'pos' not in kwargs:
            kwargs['pos'] = nx.circular_layout(self.graph)

        if 'with_labels' not in kwargs:
            kwargs['with_labels'] = True

        graph = self.graph.copy()

        categories = [None] + sorted([val for val in nx.get_node_attributes(
            graph, 'category').values() if val is not None])

        n_categories = len(categories)
        palette = sns.color_palette("husl", n_categories)

        if 'node_color' not in kwargs:
            cmap = [categories.index(graph.nodes[node]['category'])
                    if 'category' in graph.nodes[node] else 0
                    for node in graph.nodes]

            cmap = [palette[index] for index in cmap]
            kwargs['node_color'] = cmap

        if 'edge_color' not in kwargs:
            # Colour nodes
            e_categories = ['None'] + list(
                nx.get_edge_attributes(graph,
                                       'category').values())
            e_categories = sorted([str(x) for x in e_categories])

            n_edge_categories = len(e_categories)
            e_palette = sns.color_palette("husl", n_edge_categories)
            if n_edge_categories == 2:
                e_palette = ['black', 'grey']

            e_cmap = []
            #  Colour edges
            for _, _, d in graph.edges(data=True):
                cat = d['category'] if 'category' in d else 'None'
                d['category'] = cat
                e_cmap.append(e_palette[e_categories.index(cat)])

            kwargs['edge_color'] = e_cmap

        nx.draw(graph, **kwargs)

        network = Network('500px', '500px')

        if output_path:
            network.from_nx(graph)
            network.show(output_path, notebook=False)

    def output_latex(self, **kwargs):
        """
        Construct a LaTeX string describing the model mathematically

        :returns: A string containg a LaTeX equations for each of the models state variables
        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(**kwargs)

        subs_dict = {}

        letters = ['U', 'V', 'W', 'X', 'Y', 'Z',]
        for arf, l in zip(self.arf_symbols, letters):
            subs_dict[arf] = l

        letters = ['u', 'v', 'w', 'x', 'y', 'z',]
        for iaa, l in zip(self.iaa_symbols, letters):
            subs_dict[iaa] = l

        for d in self.dimer_symbols:
            components = d.split('__')[1:]
            letter1, letter2 = [subs_dict[c] for c in components]
            subs_dict[d] = f'D_{letter1}_{letter2}'

        for i, promoter in enumerate(sorted(self.promoter_symbols)[1:]):
            subs_dict[promoter] = r"G_" + str(i + 1)

        for i, p in enumerate(parameters.keys()):
            subs_dict[p] = f"p_{i}"

        odes = odes.subs(subs_dict)
        return sp.latex(odes)

    def output_julia(self, aux_func_name="", aux_func_params=[],
                     parameter_subs_dict=None):
        """
        Generate julia code describing the model including the quasi steady-state assumption

        :param aux_func_name: A julia function name for an externally defined auxin-input signal
        :param aux_func_params: Optional parameters to include for the auxin input function
        :param parameter_subs_dict: A dictionary of parameter substitutions

        :returns: A string contianing a julia function definition
        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(
                parameter_subs_dict=parameter_subs_dict)

        state_vars = self.get_state_variables()
        default_parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

        code_str = f"function {self.__class__.__name__}_f_deriv(z, p, t=0.0)\n"

        code_str += "\t(" + ", ".join(state_vars) + ") = z\n\n"

        if not aux_func_name:
            code_str += "\t(" + ", ".join(list(default_parameters.keys()) + aux_func_params) + ", " + \
                f"{self.aux_symbol}) = p\n\n"
        else:
            code_str += "\t(" + \
                ", ".join(sorted(list(default_parameters.keys())) + aux_func_params)\
                + ") = p\n\n"
            code_str += f"\t{self.aux_symbol} = {aux_func_name}(t, " + ", ".join(
                aux_func_params) + ")\n\n"

        code_str += "\treturn " + "[" + str(julia_code(odes))[1:-1] + "]\n\n"
        code_str += "end"

        return code_str

    def get_jacobian_expr(self, parameter_subs_dict=None):
        """
        Symbolically the jacobian of the model's derivative function

        Not implemented

        :param parameter_subs_dict: Dictionary of parameter substitutions

        :raises NotImplementedError:
        """
        raise NotImplementedError()

    def remove_parameter(self, rate: str, strict: bool = True):
        """
        Remove a parameter from the model.

        This method removes the parameter from the dictionary, self.parameters, and from the default parameter substitution dictionary (if it's there). If strict is True, a RuntimeError is raised if there is an attempt to remove a non-existant parameter.

        :param strict: If True, do not allow non-existant parameters to be removed
        """

        if strict and rate not in self.parameters:
            raise RuntimeError(f"Tried to remove parameter {rate} but it doesn't exist in the model")

        if rate in self.parameter_subs_dict:
            self.parameter_subs_dict.pop(rate)

        if rate in self.parameters:
            self.parameters.pop(rate)

    def remove_state(self, state: str, strict: bool = True):
        """
        Remove a state from the model.

        Removes some chemical species (i.e. a protein, dimer, or promoter state) from the model. If strict is True, an error is thrown if there is an attempt to remove a non-existant state.

        :param state: The state to remove
        :param strict: If True, the 
        """

        node_list = list(self.graph.nodes())

        if strict and state not in node_list:
            raise RuntimeError(f"Tried to remove state {state} but it doesn't exist in the model")

        if state in node_list:
            self.graph.remove_node(state)

        if state in self.promoter_symbols:
            self.promoter_symbols.remove(state)

        if state in self.arf_symbols:
            self.arf_symbols.remove(state)

        if state in self.iaa_symbols:
            self.iaa_symbols.remove(state)

        if state in self.rna_symbols:
            self.rna_symbols.remove(state)

        reaction_set_list = []

        # Remove any reaction that includes this state, also
        for reaction in self._reactions_set:
            if state not in list(reaction.products) + list(reaction.reactants) \
               + list(reaction.enzymes):
                reaction_set_list.append(reaction)

        self._reactions_set = set(reaction_set_list)

    def generate_myokit_model(self, name: str = "",
                              eliminate_promoter_state: str = ""):
        """
        Generate myokit code for the model. This method is unimplemented

        :param name: A label for the created myokit model
        :eliminate_promoter_state: The state to be eliminated from the promoter component of the model (str)

        :raises: NotImplementedError
        """

        if not name:
            name = self.__class__.__name__

        odes, free_variables, _, _ = self.generate_ode_system()

        state_variables = self.get_state_variables()
        param_dict = self.get_default_parameters()

        model = myokit.Model(name)

        #  Nuclear Auxin Pathway
        comp_name = 'NAP'
        model.add_component(comp_name)
        comp = model[comp_name]

        if eliminate_promoter_state == '':
            eliminate_promoter_state = sorted(self.promoter_symbols)[0]

        # Set up time variable
        model.add_component('engine')
        model['engine'].add_variable('time')
        model['engine']['time'].set_binding('time')
        model['engine']['time'].set_rhs(0.0)

        #  Setup auxin symbol
        comp.add_variable(self.aux_symbol)
        comp[self.aux_symbol].set_rhs(0.0)
        comp[self.aux_symbol].set_binding('pace')

        for parameter, ic in param_dict.items():
            comp.add_variable(parameter)
            comp[parameter].set_rhs(ic)

        for state in state_variables:
            comp.add_variable(state)

        if eliminate_promoter_state:
            mk_expr = 1 - sum(sp.Matrix([s for s in self.promoter_symbols if s
                                         != eliminate_promoter_state]))
            mk_expr = str(mk_expr)
            comp[eliminate_promoter_state].set_rhs(mk_expr)

        default_initial_conditions = self.get_default_initial_conditions()
        for state, ode in zip(state_variables, odes):
            if state == eliminate_promoter_state:
                continue
            ic = default_initial_conditions[state]
            comp[state].promote(ic)
            mk_expr = str(ode).replace('**', '^')
            comp[state].set_rhs(mk_expr)

        return model

    def get_model_name(self):
        """
        Compute a name for the model which includes the parameters used to create it

        :returns: A string describing the model
        """

        name = f"{self.__class__.__name__}_{self.nARFs}_ARFs_{self.nIAAs}_IAAs"
        if self.include_arf_transcription:
            name += "_with_arf_transcription"
        return name

    def set_parameter_substitution_dict(self, param_subs_dict, strict=False):
        """
        Define a dictionary of parameter substitutions to use in the model.

        :param parameter_subs_dict:
        :param strict: If true, don't allow parameters to be substituted with undefined symbols

        :raises ValueError: When a parameter found in the substitution dict isn't found in the model and strict is True
        """

        if not strict:
            param_subs_dict = {k: v for k, v in param_subs_dict
                               if k in self.get_default_parameters().keys()}

        else:
            for k in param_subs_dict:
                params = list(self.get_default_parameters(
                    parameter_subs_dict={}).keys())
                if k not in params:
                    raise ValueError(f"Tried to substitute for parameter {k}"
                                     " not found in model parameters "
                                     f"{params}")

            for expr in param_subs_dict.values():
                for symb in sp.sympify(expr).free_symbols:
                    params = list(self.get_default_parameters(
                        parameter_subs_dict={}).keys())
                    if str(symb) not in params:
                        raise ValueError(f"Tried add expression {expr} to model"
                                         " but parameter {symb} not found in model parameters "
                                         f"{params}")

        self.parameter_subs_dict = param_subs_dict

    def get_promoter_parameters(self):
        """
        Construct a list of model parameters which relate to the protein-promoter dynamics in the model

        :returns: A list of parameters
        """

        pp_reactions = []
        promoter_symbols = [s for s in self.get_state_variables()
                            if s[0] == 'G']

        for r in self._reactions_set:
            if np.all(np.isin(r.reactants + r.products, promoter_symbols)):
                pp_reactions.append(r)

        pp_params = []

        for r in pp_reactions:
            pp_params += [p for p in r.fwd_rates + r.bwd_rates]

        return pp_params

    def get_parameter_substitution_dict(self):
        """
        Return the models dictionary of parameter substitutions.

        :returns: a sorted dictionary of parameter substitutions
        """

        return _sorted_dict(self.parameter_substitution_dict)

    def pretty_print_parameter_dict(self):
        """
        Print all model parameters in a LaTeX-maths format

        :returns: Dictionary of parameter labels in a LaTeX maths format
        """

        ret_dict = {p: self.pretty_print_parameter(p)
                    for p in sorted(list(self.get_default_parameters().keys()))}

        return ret_dict

    @classmethod
    def pretty_print_parameter(cls, par):
        """
        Print a parameter in a LaTeX math format.

        :param par: The parameter label to convert

        :returns: A LaTeX-maths string
        """
        parts = par.split("__")
        head = parts[0]
        top = parts[1]
        rest = parts[2:]

        return f"{head}_{{{top}_{{{', '.join(rest)}}}}}"


def _sorted_dict(_dict):
    """
    Returns a dictionary where the keys are sorted (i.e. alphabetically)

    Note that this is only guaranteed to work for Python>=3.7 where insertion order is respected

    :param _dict: A dictionary to be sorted

    :returns: A dictionary with the same keys and values but sorted by key
    """

    #  Insertion order is respected in Python>=3.7
    ret_dict = {k: _dict[k] for k in sorted(_dict)}
    return ret_dict
