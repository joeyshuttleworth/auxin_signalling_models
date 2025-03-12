import logging

import networkx as nx
import numba as nb
import numpy as np
import seaborn as sns
import sympy as sp
from numba import cfunc, njit
from numbalsoda import lsoda, lsoda_sig
from pyvis.network import Network

from .reaction import Reaction


class auxin_signalling_pathway():
    def __init__(self, nARFs=1, nIAAs=1, aux_func=None):

        self.QSS_bound_species = {}
        self.parameter_subs_dict = {}

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

        # Dictionary defining each transition rate
        self.transition_rates = {}

        self.promoter_symbols = ["G"] + [f"G__{arf}__{arf1}"
                                         for arf1 in arf_symbols
                                         for arf2 in arf_symbols if arf1 >= arf2]\
            + [f"G__{arf}" for arf in arf_symbols]\
            + [f"G__{arf}__{iaa}" for arf in arf_symbols
               for iaa in iaa_symbols]

        for promoter in self.promoter_symbols:
            self.add_state(promoter, category='promoter')

        dimer_symbols = [f"D__{s1}__{s2}"
                         for s1 in arf_symbols + iaa_symbols
                         for s2 in arf_symbols + iaa_symbols if s2 >= s1]
        self.dimer_symbols = dimer_symbols

        for dimer in dimer_symbols:
            self.add_state(dimer, category='dimer')

        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]
        self.rna_symbols = rna_symbols

        for rna in rna_symbols:
            self.add_state(rna, category='rna')

        self.aux_symbol = 'x'
        self.add_state(self.aux_symbol, category='None')

        self.all_state_vars = sorted(arf_symbols + iaa_symbols
                                     + self.promoter_symbols + dimer_symbols
                                     + rna_symbols)

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_production = sp.sympify('arf__k')
        arf_decay = sp.sympify('arf__d')

        self.add_transition_rate(arf_production, 1.0,
                                 "production rate of ARFs")
        self.add_transition_rate(arf_decay, 1.0,
                                 "decay rate of ARFs")

        iaa_production = sp.sympify('iaa__k')
        iaa_decay = sp.sympify('iaa__d')
        self.add_transition_rate(iaa_production, 1.0,
                                 "production rate of Aux/IAAs")
        self.add_transition_rate(iaa_decay, 1.0,
                                 "decay rate of Aux/IAAs")

        for _, arf in enumerate(arf_symbols):
            self.add_reaction([], [arf], fwd_rates=[
                arf_production], bwd_rates=[arf_decay])

        for _, arf in enumerate(iaa_symbols):
            self.add_reaction([], [arf], fwd_rates=[
                iaa_production], bwd_rates=[iaa_decay])

        for d in dimer_symbols:
            association_rate = sp.sympify(f"k__{d}")
            dissociation_rate = sp.sympify(f"d__{d}")

            self.add_transition_rate(association_rate,
                                     0.001,
                                     f"Association rate of dimer {d}")
            self.add_transition_rate(dissociation_rate,
                                     0.1,
                                     f"Dissociation rate of dimer {d}")

            reactants = str(d).split('__')[1:]
            self.add_reaction(reactants, [d], [association_rate],
                              [dissociation_rate])

        for iaa in iaa_symbols:
            iaa_decay_rate = f"d__{iaa}_{self.aux_symbol}"
            self.add_transition_rate(iaa_decay_rate, 0.1,
                                     "rate of auxin-mediated decay of Aux/IAA")
            self.add_reaction([iaa], [self.empty_set_node],
                              [iaa_decay_rate],
                              enzymes=[self.aux_symbol])

        for promoter in self.promoter_symbols:
            if promoter == 'G':
                continue

            # TODO clean up this code
            reactant = promoter[3:]
            if '__' in reactant:
                reactant = f"D__{reactant}"

            promoter_binding_rate = f"k__{promoter}"
            promoter_unbinding_rate = f"d__{promoter}"

            self.add_transition_rate(promoter_binding_rate,
                                     0.1,
                                     f"binding rate for promoter {promoter}")
            self.add_transition_rate(promoter_unbinding_rate,
                                     0.01,
                                     "dissociation rate for promoter"
                                     f"{promoter}")

            self.add_reaction(
                [sp.sympify('G')], [promoter],
                [promoter_binding_rate], [promoter_unbinding_rate],
                enzymes=[reactant]
            )

        for promoter in self.promoter_symbols:
            for rna_symbol in rna_symbols:
                rate = f"k__{promoter}__{rna_symbol}"

                self.add_transition_rate(rate,
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

                    if promoter_binding_rate not in self.transition_rates:
                        self.add_transition_rate(promoter_binding_rate)
                        self.add_transition_rate(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [promoter_binding_rate], [promoter_unbinding_rate],
                            enzymes=[reactants[1]])

                partially_bound_promoter = f"G__{reactants[1]}"
                if partially_bound_promoter in self.promoter_symbols:
                    promoter_binding_rate = f"k__{partially_bound_promoter}__bind__{reactants[0]}"
                    promoter_unbinding_rate = f"d__{promoter}__unbind__{reactants[0]}"

                    if promoter_binding_rate not in self.transition_rates:
                        self.add_transition_rate(promoter_binding_rate)
                        self.add_transition_rate(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [promoter_binding_rate], [promoter_unbinding_rate],
                            enzymes=[reactants[0]])

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        rna_production_rate = sp.sympify('k__R')
        self.add_transition_rate(rna_decay_rate,
                                 1.0 / 240,
                                 "RNA decay rate")
        self.add_transition_rate(rna_production_rate,
                                 0.01,
                                 "base RNA production rate")

        for rna_symbol in rna_symbols:
            self.add_reaction([self.empty_set_node], [
                rna_symbol], [], [rna_decay_rate])
            iaa = rna_symbol.split('__')[1]

            iaa_production_rate = f"k__{rna_symbol}__{iaa}"

            rate_default_val = 1.0 if rna_symbols.index(rna_symbol)\
                == iaa_symbols.index(iaa) else 0.0

            self.add_transition_rate(iaa_production_rate, rate_default_val,
                                     f"Rate of production of {iaa} from"
                                     f"RNA {rna_symbol}")

            self.add_reaction([], [iaa],
                              fwd_rates=[iaa_production_rate],
                              enzymes=[rna_symbol])

        for iaa in iaa_symbols:
            production_rate = sp.sympify(f"k__{iaa}")

            self.add_transition_rate(production_rate, 0.0,
                                     f"Production rate for {iaa}")

            self.add_reaction([self.empty_set_node], [iaa],
                              [production_rate],
                              enzymes=rna_symbols)

        self.iaa_symbols = iaa_symbols
        self.arf_symbols = arf_symbols

        self.state_var_eqns = state_var_eqns

        self.rhs_func = None
        self.cfunc_rhs = None

        self.jac_func = None

    def get_rhs_func(self, **kwargs):
        if self.rhs_func is None and not kwargs:
            self.rhs_func = self.generate_rhs_func()
        elif kwargs:
            return self.generate_rhs_func(**kwargs)
        return self.rhs_func

    def add_transition_rate(self, rate, default_value=0.0,
                            description="", strict=True):

        rate = str(rate)
        logging.debug(f"Adding rate {rate}")

        if strict is None:
            strict = self.strict

        if strict and rate in self.transition_rates:
            raise Exception(f"rate {rate} already present "
                            f"in model: {self.transition_rates}")

        self.transition_rates[rate] = (default_value, description)

    def add_state(self, label, category=None, strict=True):
        label = str(label)

        if strict and label in self.graph.nodes():
            raise Exception("")

        self.graph.add_node(label, category=category)

    def add_reaction(self, reactants=[], products=[], fwd_rates=[],
                     bwd_rates=[], enzymes=[], add_states=False, strict=None,
                     reaction_type='linear'):
        """Add a chemical reaction to the model.

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
            if rate not in self.transition_rates:
                if strict:
                    raise Exception(f"rate {rate} not found in model"
                                    f"model has:\n{self.transition_rates}")
                else:
                    self.add_transition_rate(rate)

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
        return self._reactions_set

    def generate_ode_system(self, parameter_subs_dict=None,
                            state_variables=None):
        """Generate a list of sympy expressions comprising an ODE system

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

        transition_rates = _sorted_dict(self.transition_rates.copy())

        # Ensure that parameters actually appear in the model
        if parameter_subs_dict:
            transition_rates = {parameter_subs_dict[key]
                                if key in parameter_subs_dict
                                and len(sp.sympify(parameter_subs_dict[key]).free_symbols) == 1
                                else key:
                                (value[0], "")
                                for key, value in transition_rates.items()
                                }

        return odes, free_variables, transition_rates, constraints

    def get_state_variables(self):
        """ Construct a list of the model's state variables

        :returns: a list of the model's state variables, sorted in
        alphabetical order, and excluding the empty-set node, and the auxin
        concentration

        """
        return sorted(list([str(s) for s in self.graph.nodes()
                            if s != self.empty_set_node
                            and s != self.aux_symbol]))

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

            ic_dict = {k: ic_dict[k] for k in sorted(ic_dict.keys())}
            params_dict = {k: params_dict[k]
                           for k in sorted(params_dict.keys())}

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
            data[1:] = params.copy()
            return lsoda(cfunc_ptr, ic.copy(),
                         np.array(ts),
                         data=data.copy())

        return solver

    def get_cfunc_rhs(self, **kwargs):
        if self.cfunc_rhs is None and not kwargs:
            self.cfunc = self.make_cfunc_rhs()
        elif kwargs:
            return self.make_cfunc_rhs()

        return self.cfunc_rhs

    def make_cfunc_rhs(self, aux_func=None, parameter_subs_dict={}):
        """Generate a c function which evaluates the RHS of the ODE system

        :returns: a c function with signature lsoda_sig for use solving our ODE
        system

        """

        n_y = len(self.get_state_variables())

        # Number of params + 'free variables' i.e. auxin
        n_data = len(self.get_default_parameters()) + 1
        rhs_func = self.generate_rhs_func(parameter_subs_dict=parameter_subs_dict,
                                          njitted=True)

        @cfunc(lsoda_sig)
        def cfunc_rhs(t, y, dy, data):
            y = nb.carray(y, n_y)
            dy = nb.carray(dy, n_y)
            data = nb.carray(data, n_data)

            params = data[1:].copy().flatten().astype(np.float64)

            if aux_func:
                aux_func(t)
            else:
                aux = data[0:1]

            res = rhs_func(y, params,
                           aux).astype(np.float64).flatten()
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
        odes, free_variables, transition_rates, \
            _ = self.generate_ode_system(
                parameter_subs_dict=parameter_subs_dict)

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        ic_dict = self.get_default_initial_conditions()

        state_variables = sp.Matrix([s for s in ic_dict])

        param_names = [x for p in transition_rates.keys()
                       for x in sp.sympify(p).free_symbols]

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])
        rhs_func = sp.lambdify((state_variables, param_names,
                                free_variables), odes)

        params = np.array(
            [x[0] for x in transition_rates.values()]).astype(np.float64)
        ic = np.array([x for x in ic_dict.values()]).astype(np.float64)
        rhs_func(ic, params, np.array([.99]))

        if njitted:
            rhs_func = njit(rhs_func)

        return rhs_func

    def get_default_initial_conditions(self):
        """List the initial conditions of the system


        :returns: a dictionary with a key for each state variable corresponding
        to various default initial conidtions

        """

        state_vars = self.get_state_variables()
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
            parameter_subs_dict = self.parameter_subs_dict

        param_dict = {parameter_subs_dict[k] if k in parameter_subs_dict
                      and len(sp.sympify(parameter_subs_dict[k]).free_symbols) == 1
                      else k:
                      v[0]
                      for k, v in self.transition_rates.items()}

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

    def output_latex(self):
        odes, free_variables, transition_rates, \
            constraints = self.generate_ode_system()

        return sp.latex(odes)

    def remove_state(self, state):
        self.graph.remove_node(state)

        reaction_set_list = []
        for reaction in self._reactions_set:
            if state not in list(reaction.products) + list(reaction.reactants) \
               + list(reaction.enzymes):
                reaction_set_list.append(reaction)

        self._reactions_set = set(reaction_set_list)


def _sorted_dict(_dict):
    #  Insertion order is respected (?)
    ret_dict = {k: _dict[k] for k in sorted(_dict)}
    return ret_dict
