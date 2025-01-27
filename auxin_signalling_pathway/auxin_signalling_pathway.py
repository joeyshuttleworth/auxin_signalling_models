import logging

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import sympy as sp
from pyvis.network import Network


class auxin_signalling_pathway():
    def __init__(self, nARFs=1, nIAAs=1):

        self.nARFs = nARFs
        self.nIAAs = nIAAs

        self.graph = nx.Graph()

        self._reactions_list = []

        self.empty_set_node = '__phi'

        self.add_state(self.empty_set_node, category='None')

        self.strict = False

        #  Initialise parameter symbols
        arf_symbols = [sp.sympify(f"ARF_{i + 1}") for i in range(nARFs)]

        for arf in arf_symbols:
            self.add_state(arf, category='ARF')

        iaa_symbols = [sp.sympify(f"iaa_{i + 1}") for i in range(nIAAs)]

        for iaa in iaa_symbols:
            self.add_state(iaa, category='IAA')

        # Dictionary defining each transition rate
        self.transition_rates = {}

        promoter_symbols = [f"G__{arf}__{arf1}"
                            for arf in arf_symbols for arf1 in arf_symbols]\
            + [f"G__{arf}" for arf in arf_symbols]\
            + [f"G__{arf}__{iaa}" for arf in arf_symbols
               for iaa in iaa_symbols]

        for promoter in promoter_symbols:
            self.add_state(promoter, category='promoter')

        dimer_symbols = [sp.sympify(f"D__{s1}__{s2}")
                         for s1 in arf_symbols + iaa_symbols
                         for s2 in arf_symbols + iaa_symbols]

        for dimer in dimer_symbols:
            self.add_state(dimer, category='dimer')

        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]

        for rna in rna_symbols:
            self.add_state(rna, category='rna')

        self.aux_symbol = 'x'

        self.all_state_vars = arf_symbols + iaa_symbols + \
            promoter_symbols + dimer_symbols + rna_symbols

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_production = sp.sympify('arf_k')
        arf_decay = sp.sympify('arf_d')

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
            self.add_reaction([iaa], [self.empty_set_node], [
                              sp.sympify('d_a')], enzyme=self.aux_symbol)

        for promoter in promoter_symbols:
            reactants = promoter.split('__')[1:]

            self.add_transition_rate(f"k__{promoter}")
            self.add_transition_rate(f"k__{promoter}")

            self.add_reaction(
                reactants + [sp.sympify('G')], [promoter],
                [sp.sympify('g_k')], [sp.sympify('g_d')])

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
                              enzyme=[rna_production_rate])

        for promoter in promoter_symbols:
            self.add_reaction([], [rna_symbol], [sp.sympify('k')], [
                              sp.sympify('d')], enzyme='promoter')
        for iaa in iaa_symbols:
            production_rate = sp.sympify(f"k__{iaa}")
            decay_rate = sp.sympify(f"d__{iaa}")

            self.add_transition_rate(production_rate)
            self.add_transition_rate(decay_rate)

            self.add_reaction([self.empty_set_node], [iaa],
                              [production_rate],
                              [decay_rate],
                              enzyme=rna_symbol)

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
                     bwd_rates=[], enzyme=None, add_states=True, strict=None):

        reactants = [str(r) for r in reactants]
        products = [str(p) for p in products]
        fwd_rates = [str(r) for r in reactants]
        bwd_rates = [str(p) for p in products]

        self._reactions_list.append(
            (reactants, products, fwd_rates, bwd_rates, enzyme))

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
            for fwd_rate, product in zip(fwd_rates, products):
                self.graph.add_edge(self.empty_set_node,
                                    product, weight=fwd_rate, enzyme=enzyme)

            for bwd_rate, reactant in zip(bwd_rates, reactants):
                self.graph.add_edge(
                    reactant, self.empty_set_node, weight=bwd_rate,
                    enzyme=None)

        for fwd_rate, reactant in zip(fwd_rates, reactants):
            for product in products:
                self.graph.add_edge(reactant, product, weight=fwd_rate)

        for bwd_rate, reactant in zip(bwd_rates, reactants):
            for product in products:
                self.graph.add_edge(product, reactant, weight=bwd_rate)

    def generate_ode_system(self):

        free_variables = self.aux_symbol

        odes = []
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
        print(graph.nodes())
        network.from_nx(graph)
        network.show("graph.html", notebook=False)

        nx.draw(graph, **kwargs)


if __name__ == '__main__':

    asp_model = auxin_signalling_pathway(2, 2)
    print(nx.write_network_text(asp_model.graph))
    print(asp_model.transition_rates)
    print(asp_model._reactions_list)
    print(len(asp_model.transition_rates))

    asp_model.draw_graph()
    plt.show()
