import logging

import numpy as np
import networkx as nx
import seaborn as sns
import sympy as sp
from pyvis.network import Network
from numba import njit

from .reaction import Reaction


class auxin_signalling_pathway():
    def __init__(self, nARFs=1, nIAAs=1):

        self.nARFs = nARFs
        self.nIAAs = nIAAs

        self.graph = nx.Graph()

        self._reactions_set = set()

        self.empty_set_node = '__phi'

        self.add_state(self.empty_set_node, category='None')

        self.strict = False

        #  Initialise parameter symbols
        arf_symbols = [f"ARF_{i + 1}" for i in range(nARFs)]
        self.arf_symbols = arf_symbols

        for arf in arf_symbols:
            self.add_state(arf, category='ARF')

        iaa_symbols = [f"iaa_{i + 1}" for i in range(nIAAs)]

        for iaa in iaa_symbols:
            self.add_state(iaa, category='IAA')

        # Dictionary defining each transition rate
        self.transition_rates = {}

        self.promoter_symbols = ["G"] + [f"G__{arf}__{arf1}"
                                         for arf in arf_symbols
                                         for arf1 in arf_symbols]\
            + [f"G__{arf}" for arf in arf_symbols]\
            + [f"G__{arf}__{iaa}" for arf in arf_symbols
               for iaa in iaa_symbols]

        for promoter in self.promoter_symbols:
            self.add_state(promoter, category='promoter')

        dimer_symbols = [f"D__{s1}__{s2}"
                         for s1 in arf_symbols + iaa_symbols
                         for s2 in arf_symbols + iaa_symbols if s2 >= s1]

        for dimer in dimer_symbols:
            self.add_state(dimer, category='dimer')

        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]
        self.rna_symbols = rna_symbols

        for rna in rna_symbols:
            self.add_state(rna, category='rna')

        self.aux_symbol = 'x'
        self.add_state(self.aux_symbol, category='None')

        self.all_state_vars = sorted(arf_symbols + iaa_symbols +
                                     self.promoter_symbols + dimer_symbols
                                     + rna_symbols)

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_production = sp.sympify('arf_k')
        arf_decay = sp.sympify('arf_d')

        self.add_transition_rate(arf_production)
        self.add_transition_rate(arf_decay)

        for _, arf in enumerate(arf_symbols + iaa_symbols):
            self.add_reaction([], [arf], fwd_rates=[
                arf_production], bwd_rates=[arf_decay])

        for d in dimer_symbols:
            # TODO define parameter labels
            association_rate = sp.sympify(f"k__{d}")
            dissociation_rate = sp.sympify(f"d__{d}")

            self.add_transition_rate(association_rate)
            self.add_transition_rate(dissociation_rate)

            reactants = str(d).split('__')[1:]
            self.add_reaction(reactants, [d], [association_rate], [
                dissociation_rate])

        for iaa in iaa_symbols:
            iaa_decay_rate = 'd_a'
            self.add_transition_rate(iaa_decay_rate)
            self.add_reaction([iaa], [self.empty_set_node],
                              [iaa_decay_rate],
                              enzymes=[self.aux_symbol])

        for promoter in self.promoter_symbols:
            if promoter == 'G':
                continue

            reactants = sorted(promoter.split('__')[1:])

            promoter_binding_rate = f"g__{promoter}"
            promoter_unbinding_rate = f"d__{promoter}"

            self.add_transition_rate(promoter_binding_rate)
            self.add_transition_rate(promoter_unbinding_rate)

            if len(reactants) == 2:
                reactants = [f"D__{reactants[0]}__{reactants[1]}"]
            elif len(reactants) > 3:
                raise NotImplementedError()

            self.add_reaction(
                reactants + [sp.sympify('G')], [promoter],
                [promoter_binding_rate], [promoter_unbinding_rate])

            for rna_symbol in rna_symbols:
                max_rate = f"kmax__{rna_symbol}_{promoter}"
                half_rate_conc = f"Ka__{rna_symbol}_{promoter}"
                n = f"n__{rna_symbol}__{promoter}"

                self.add_transition_rate(max_rate)
                self.add_transition_rate(half_rate_conc)
                self.add_transition_rate(n)

                self.add_reaction(
                    [], [rna_symbol], [max_rate, half_rate_conc, n],
                    [], [promoter], reaction_type='tf-hill-act'
                )

            if len(reactants) == 2:
                # TODO use different transition rates

                partially_bound_promoter = f"G__{reactants[0]}"
                self.add_reaction(
                    [reactants[1]] + [partially_bound_promoter], [promoter],
                    [promoter_binding_rate], [promoter_unbinding_rate])

                partially_bound_promoter = f"G__{reactants[1]}"
                self.add_reaction(
                    [reactants[0]] + [partially_bound_promoter], [promoter],
                    [promoter_binding_rate], [promoter_unbinding_rate])

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        rna_production_rate = sp.sympify('k__R')
        self.add_transition_rate(rna_decay_rate)
        self.add_transition_rate(rna_production_rate)

        for rna_symbol in rna_symbols:
            self.add_reaction([self.empty_set_node], [
                rna_symbol], [], [rna_decay_rate])
            iaa = rna_symbol.split('__')[1]
            self.add_reaction([], [iaa],
                             fwd_rates=[rna_production_rate],
                             enzymes=[rna_symbol])

        for iaa in iaa_symbols:
            production_rate = sp.sympify(f"k__{iaa}")
            decay_rate = sp.sympify(f"d__{iaa}")

            self.add_transition_rate(production_rate)
            self.add_transition_rate(decay_rate)

            self.add_reaction([self.empty_set_node], [iaa],
                             [production_rate],
                              enzymes=rna_symbols)
            self.add_reaction([iaa], [self.empty_set_node],
                             [production_rate])

        #  TODO convert network to ODE
        self.state_var_eqns = state_var_eqns

    def add_transition_rate(self, rate, strict=None, description=''):

        rate = str(sp.sympify(rate))

        logging.debug(f"Adding rate {rate}")

        if strict is None:
            strict = self.strict

        if strict and rate in self.transition_rates:
            raise Exception("")

        self.transition_rates[rate] = description

    def add_state(self, label, category=None, strict=True):

        label = str(label)

        if strict and label in self.graph.nodes():
            raise Exception("")

        self.graph.add_node(label, category=category)

    def add_reaction(self, reactants=[], products=[], fwd_rates=[],
                     bwd_rates=[], enzymes=[], add_states=False, strict=True,
                     reaction_type='linear'):

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
            for state in set(reactants + products):
                if state not in self.graph.nodes:
                    self.add_state(state)

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
    @staticmethod
    def _get_reaction_terms(reaction, variable):
        terms = [sp.sympify('0')]

        if variable in reaction.enzymes:
            return [0]

        if variable not in reaction.reactants + reaction.products:
            raise Exception("")

        if variable in reaction.reactants and variable in reaction.products:
            raise Exception("")

        if reaction.reaction_type == 'linear':
            if variable in reaction.products and reaction.bwd_rates:
                reaction = reaction.reversed()
            fwd_rate = sp.sympify(
                reaction.fwd_rates[0]) if reaction.fwd_rates else 0
            bwd_rate = sp.sympify(
                reaction.bwd_rates[0]) if reaction.bwd_rates else 0

            if variable in reaction.reactants and reaction.fwd_rates:
                if not reaction.enzymes:
                    terms.append(-sp.prod([sp.sympify(reactant)
                                           for reactant in reaction.reactants])
                                 * fwd_rate)
                    terms.append(sp.prod([sp.sympify(product)
                                          for product in reaction.products])
                                 * bwd_rate)
                if reaction.enzymes and [t for t in terms if t != 0]:
                    terms = [t * sp.prod(list(reaction.enzymes)) for t in terms]
            return terms

        elif reaction.reaction_type == "tf-hill-act":
            if variable in reaction.products:
                assert len(reaction.products) == 1
                assert len(reaction.reactants) == 0
                assert len(reaction.enzymes) == 1

                y = sp.sympify(reaction.enzymes[0])
                k, ka, n = [sp.sympify(x) for x in reaction.fwd_rates]
                terms.append(k * y / (ka**n + y**n))
            return terms

        elif reaction.reaction_type == "tf-hill-repr":
            if variable in reaction.products:
                assert len(reaction.products) == 0
                assert len(reaction.enzymes) == 1
                y = sp.sympify(reaction.enzymes[0])
                k, ka, n = [sp.sympify(x) for x in reaction.fwd_rates]
                terms.append(k * (1 - y) / (ka**n + y**n))
            return terms

        else:
            raise NotImplementedError("")

    def get_reactions_set(self):
        return self._reactions_set

    def generate_ode_system(self):

        free_variables = [self.aux_symbol]

        state_variables = sorted(self.graph.nodes())

        odes = [sp.sympify('0') for _ in state_variables]

        for reaction in self._reactions_set:
            for variable in reaction.reactants + reaction.products:
                var_index = state_variables.index(variable)
                odes[var_index] += sp.Add(*
                                          self._get_reaction_terms(reaction,
                                                                   variable))
        # TODO include the fact that the promoter variables sum to 1
        constraints = []

        return odes, free_variables, self.transition_rates, constraints

    def get_state_variables(self):
        return list(self.graph.nodes())

    def generate_rhs_func(self, njitted=False):
        odes, free_variables, transition_rates,\
            _ = self.generate_ode_system()

        # Create column vector of ode expressions
        odes = sp.Matrix([odes]).T
        params = list(transition_rates.keys())
        state_variables = [s for s in self.get_state_variables()
                           if s not in free_variables]

        free_variables = [sp.sympify(x) for x in free_variables]

        rhs_func = sp.lambdify((state_variables, params, free_variables), odes)

        if njitted:
            rhs_func = njit(rhs_func)

        return rhs_func

    def get_default_initial_conditions(self):

        state_vars = self.get_state_variables()
        ic_dict = {s: .0 for s in state_vars}

        ic_dict['G'] = 1.0

        for arf in self.arf_symbols:
            ic_dict[arf] = 1.0

        return ic_dict

    def get_default_parameters(self):
        param_dict = {v: .0 for v in self.transition_rates}

        param_dict['d__R'] = 1 / 240.0

        return param_dict

    def draw_graph(self, **kwargs):
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
            # Colour edges
            for _, _, d in graph.edges(data=True):
                cat = d['category'] if 'category' in d else 'None'
                d['category'] = cat
                e_cmap.append(e_palette[e_categories.index(cat)])

            kwargs['edge_color'] = e_cmap

        nx.draw(graph, **kwargs)

        network = Network('500px', '500px')
        network.from_nx(graph)
        network.show("graph.html", notebook=False)

