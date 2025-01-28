import logging

import networkx as nx
import seaborn as sns
import sympy as sp
from pyvis.network import Network

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
                         for s2 in arf_symbols + iaa_symbols]

        for dimer in dimer_symbols:
            self.add_state(dimer, category='dimer')

        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]

        for rna in rna_symbols:
            self.add_state(rna, category='rna')

        self.aux_symbol = 'x'

        self.all_state_vars = sorted(arf_symbols + iaa_symbols +
                                     self.promoter_symbols + dimer_symbols
                                     + rna_symbols)

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_production = sp.sympify('arf_k')
        arf_decay = sp.sympify('arf_d')

        for _, arf in enumerate(arf_symbols + iaa_symbols):
            self.addreaction([], [arf], fwd_rates=[
                arf_production], bwd_rates=[arf_decay])

        for d in dimer_symbols:
            # TODO define parameter labels
            association_rate = sp.sympify(f"k__{d}")
            dissociation_rate = sp.sympify(f"d__{d}")

            self.add_transition_rate(association_rate)
            self.add_transition_rate(dissociation_rate)

            reactants = str(d).split('__')[1:]
            self.addreaction(reactants, [d], [association_rate], [
                dissociation_rate])

        for iaa in iaa_symbols:
            self.addreaction([iaa], [self.empty_set_node], [
                sp.sympify('d_a')], enzymes=[self.aux_symbol])

        for promoter in self.promoter_symbols:

            if promoter == 'G':
                continue

            reactants = promoter.split('__')[1:]

            self.add_transition_rate(f"k__{promoter}")
            self.add_transition_rate(f"k__{promoter}")

            self.addreaction(
                reactants + [sp.sympify('G')], [promoter],
                [sp.sympify('g_k')], [sp.sympify('g_d')])

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        rna_production_rate = sp.sympify('k__R')
        self.add_transition_rate(rna_decay_rate)
        self.add_transition_rate(rna_production_rate)
        for rna_symbol in rna_symbols:
            self.addreaction([self.empty_set_node], [
                rna_symbol], [], [rna_decay_rate])
            iaa = rna_symbol.split('__')[1]
            self.addreaction([], [iaa],
                             fwd_rates=[rna_production_rate],
                             enzymes=[rna_production_rate])

        for promoter in self.promoter_symbols:
            if promoter == 'G':
                continue

            rna_symbol = promoter.split('__')[1]

            self.addreaction([], [rna_symbol], [sp.sympify('k')], [
                sp.sympify('d')], enzymes=['promoter'])

        for iaa in iaa_symbols:
            production_rate = sp.sympify(f"k__{iaa}")
            decay_rate = sp.sympify(f"d__{iaa}")

            self.add_transition_rate(production_rate)
            self.add_transition_rate(decay_rate)

            self.addreaction([self.empty_set_node], [iaa],
                             [production_rate],
                             [decay_rate],
                             enzymes=[rna_symbol])

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

    def addreaction(self, reactants=[], products=[], fwd_rates=[],
                    bwd_rates=[], enzymes=[], add_states=True, strict=None):

        reactants = tuple([str(r) for r in reactants])
        products = tuple([str(p) for p in products])
        fwd_rates = tuple([str(r) for r in reactants])
        bwd_rates = tuple([str(p) for p in products])
        print(enzymes)
        enzymes = tuple(enzymes)

        reaction = Reaction(reactants, products,
                            fwd_rates, bwd_rates, enzymes)
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
                    raise Exception("")
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

    @classmethod
    def _get_reaction_terms(cls, reaction, variable):
        terms = [sp.sympify('0')]
        if variable not in reaction.reactants + reaction.products:
            raise Exception("")

        if variable in reaction.reactants and variable in reaction.products:
            raise Exception("")

        if reaction.reaction_type == 'linear':
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
                else:
                    terms.append(sp.prod([sp.sympify(prod)
                                          for prod in reaction.products
                                          + reaction.enzymes]) * fwd_rate)
            elif variable in reaction.products and reaction.bwd_rates:
                if not reaction.enzymes:
                    terms.append(-sp.prod([sp.sympify(product)
                                           for product in reaction.products])
                                 * fwd_rate)
                    terms.append(sp.prod([sp.sympify(reactant)
                                          for reactant in reaction.reactants])
                                 * bwd_rate)
                else:
                    terms.append(-sp.prod([sp.sympify(prod)
                                           for prod in reaction.products
                                           + reaction.enzymes]) * fwd_rate)
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

        return odes, constraints, free_variables, self.transition_rates

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

        print(graph.nodes)
        if 'cmap' not in kwargs:
            cmap = [categories.index(graph.nodes[node]['category'])
                    for node in graph.nodes]
            cmap = [palette[index] for index in cmap]
            kwargs['node_color'] = cmap

        for node in graph.nodes:
            graph.nodes[node]['fillcolor'] = palette[categories.index(
                graph[node]['category']) if 'category' in graph[node] else 0]
            graph.nodes[node]['category'] = str(graph.nodes[node]['category'])

        network = Network('500px', '500px')
        network.from_nx(graph)
        network.show("graph.html", notebook=False)

        nx.draw(graph, **kwargs)
