import sympy as sp
import networkx as nx
import logging

import matplotlib.pyplot as plt

class auxin_signalling_pathway():
    def __init__(self, nARFs=1, nIAAs=1):

        self.nARFs = nARFs
        self.nIAAs = nIAAs

        self.graph = nx.Graph()

        self.empty_set_node = r'\phi'
        self.graph.add_node(self.empty_set_node)

        self.strict = False

        # Initialise parameter symbols
        arf_symbols = [sp.sympify(f"ARF_{i + 1}") for i in range(nARFs)]
        iaa_symbols = [sp.sympify(f"iaa_{i + 1}") for i in range(nIAAs)]

        # Dictionary defining each transition rate
        self.transition_rates = {}

        promoter_symbols = [f"G__{arf}__{arf1}" for arf in arf_symbols for arf1 in arf_symbols]\
            + [f"G__{arf}" for arf in arf_symbols]\
            + [f"G__{arf}__{iaa}" for arf in arf_symbols for iaa in iaa_symbols]
        dimer_symbols = [sp.sympify(f"D__{s1}__{s2}") for s1 in arf_symbols + iaa_symbols for s2 in arf_symbols + iaa_symbols]
        rna_symbols = [f"R__{iaa}" for iaa in iaa_symbols]
        self.aux_symbol = 'x'

        self.all_state_vars = arf_symbols + iaa_symbols + promoter_symbols + dimer_symbols + rna_symbols

        state_var_eqns = [sp.sympify('0') for _ in self.all_state_vars]

        arf_production = sp.sympify('arf_k')
        arf_decay = sp.sympify('arf_d')

        for _, arf in enumerate(arf_symbols + iaa_symbols):
            self.add_reaction([], [arf], fwd_rates=[arf_production], bwd_rates=[arf_decay])

        for d in dimer_symbols:
            # TODO define parameter labels
            association_rate = sp.sympify(f"k__{d}")
            dissociation_rate = sp.sympify(f"d__{d}")

            self.add_transition_rate(association_rate)
            self.add_transition_rate(dissociation_rate)

            reactants = str(d).split('__')[1:]
            self.add_reaction(reactants, [d], [association_rate], [dissociation_rate])

        for iaa in iaa_symbols:
            self.add_reaction([iaa], [self.empty_set_node], [sp.sympify('d_a')], enzyme=self.aux_symbol)

        for promoter in promoter_symbols:
            reactants = promoter.split('__')[1:]

            self.add_transition_rate(f"k__{promoter}")
            self.add_transition_rate(f"k__{promoter}")

            self.add_reaction(reactants + [sp.sympify('G')], [promoter], [sp.sympify('g_k')], [sp.sympify('g_d')])

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        self.add_transition_rate(rna_decay_rate)
        for rna_symbol in rna_symbols:
            self.add_reaction([self.empty_set_node], [rna_symbol], [], [rna_decay_rate])

            for promoter in promoter_symbols:
                self.add_reaction([], [rna_symbol], [sp.sympify('k')], [sp.sympify('d')], enzyme='promoter')
            for iaa in iaa_symbols:
                production_rate = sp.sympify(f"k__{iaa}")
                decay_rate = sp.sympify(f"d__{iaa}")

                self.add_transition_rate(production_rate)
                self.add_transition_rate(decay_rate)

                self.add_reaction([self.empty_set_node], [iaa], [production_rate], [decay_rate], enzyme=rna_symbol)

        # TODO convert network to ODE
        self.state_var_eqns = state_var_eqns

    def add_transition_rate(self, rate, strict=None, description=''):

        rate = sp.sympify(rate)

        logging.debug(f"Adding rate {rate}")

        if strict is None:
            strict = self.strict

        if strict and rate in self.transition_rates:
            raise Exception("")

        self.transition_rates[rate] = description

    def add_reaction(self, reactants=[], products=[], fwd_rates=[],
                     bwd_rates=[], enzyme=None, add_states=True, strict=None):

        if strict is None:
            strict = self.strict

        logging.debug(f"adding reaction from {reactants} to {products} with rates {fwd_rates} and {bwd_rates}")

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
                if state not in self.graph:
                    self.graph.add_node(state)

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
                self.graph.add_edge(self.empty_set_node, product, weight=fwd_rate, enzyme=enzyme)

            for bwd_rate, reactant in zip(bwd_rates, reactants):
                self.graph.add_edge(reactant, self.empty_set_node, weight=bwd_rate, enzyme=None)

        for fwd_rate, reactant in zip(fwd_rates, reactants):
            for product in products:
                self.graph.add_edge(reactant, product, weight=fwd_rate)

        for bwd_rate, reactant in zip(bwd_rates, reactants):
            for product in products:
                self.graph.add_edge(product, reactant, weight=bwd_rate)


if __name__ == '__main__':
    # Temporary code for development
    # logging.basicConfig(level=logging.DEBUG)

    asp_model = auxin_signalling_pathway(2, 2)
    print(nx.write_network_text(asp_model.graph))
    print(asp_model.transition_rates)
    print(len(asp_model.transition_rates))

    # logging.basicConfig(level=logging.ERROR)
    nx.draw(asp_model.graph, pos=nx.circular_layout(asp_model.graph), with_labels=True)
    plt.show()
