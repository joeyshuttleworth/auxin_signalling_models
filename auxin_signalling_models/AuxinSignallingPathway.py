import logging
from copy import copy
from typing import Callable, List

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from pyvis.network import Network
from scipy.integrate import solve_ivp
from sympy.printing.julia import julia_code

try:
    import myokit
except ImportError:
    myokit = None

from .Parameter import Parameter
from .Reaction import Reaction
from .StateVariable import StateVariable

_reserved_variable_names = ["_t"]


class AuxinSignallingPathway:

    def __init__(self, nARFs: int = 1, nIAAs: int = 1, include_arf_transcription: bool = False,
                 aux_func: Callable[[float], float] = None):
        """A graph-based representation of an auxin-signalling pathway in terms
        of chemical reactions and chemical species.

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

        # Stored as a set of Variable objects
        self._parameters = set()

        self.nARFs = nARFs
        self.nIAAs = nIAAs

        self.graph = nx.Graph()

        self._reactions_set = set()

        empty_set_label = '__phi'

        self.empty_set_node = self.add_state(empty_set_label,
                                             category='None')

        self.strict = True

        self.parameter_sub_dict = {}

        arf_variable_labels = [f"ARF_{i + 1}" for i in range(self.nARFs)]
        iaa_variable_labels = [f"iaa_{i + 1}" for i in range(self.nIAAs)]

        #  Initialise parameter symbols
        self.arf_variables = [self.add_state(a, category="ARF",
                                             initial_condition=1.0)
                              for a in arf_variable_labels]

        self.iaa_variables = [self.add_state(i, category="IAA",
                                             initial_condition=0.1)
                              for i in iaa_variable_labels]

        promoter_labels = ["G"] + [f"G__{arf1}__{arf2}"
                                   for arf1 in arf_variable_labels
                                   for arf2 in arf_variable_labels
                                   if arf2 >= arf1]\
            + [f"G__{arf}" for arf in arf_variable_labels]\
            + [f"G__{arf}__{iaa}" for arf in arf_variable_labels
               for iaa in iaa_variable_labels]

        unbound_promoter = self.add_state("G", description="Unbound promoter state",
                                          category="promoter", initial_condition=1.0)
        promoter_variables = [unbound_promoter]

        for promoter in [p for p in promoter_labels if p != "G"]:
            components = promoter.split("__")[1:]

            if len(components) == 2:
                components = [f"D__{'__'.join(components)}"]

            _promoter_var = self.add_state(promoter, category="promoter",
                                           components=components,
                                           initial_condition=0.0)
            promoter_variables.append(_promoter_var)

        self.promoter_variables = promoter_variables

        dimer_variable_labels = [f"D__{s1}__{s2}"
                                 for s1 in arf_variable_labels + iaa_variable_labels
                                 for s2 in arf_variable_labels + iaa_variable_labels if s2 >= s1]
        self.dimer_variable_labels = dimer_variable_labels

        dimer_variables = []
        for dimer in dimer_variable_labels:
            components = dimer.split("__")[1:]
            _dimer_var = self.add_state(dimer, category='dimer',
                                        components=components,
                                        description=f"Dimer of {components[0]} protein "
                                        f"and {components[1]} protein",
                                        initial_condition=0.1)
            dimer_variables.append(_dimer_var)

        self.dimer_variables = dimer_variables

        rna_var_labels = [f"R__{iaa}" for iaa in iaa_variable_labels]

        if include_arf_transcription:
            rna_var_labels += [f"R__{arf}" for arf in arf_variable_labels]

        rna_variables = []

        for r in rna_var_labels:
            _rna_var = self.add_state(r, category="rna",
                                      description=f"RNA for protein {r.split('__')[1]}",
                                      initial_condition=0.0)
            rna_variables.append(_rna_var)

        self.rna_variables = rna_variables

        self.auxin_variable_label = 'x'
        self.auxin_variable = self.add_state(
            self.auxin_variable_label, category='None',
            initial_condition=1.0
        )

        self.all_state_variables = self.arf_variables + self.iaa_variables \
            + self.promoter_variables + dimer_variables \
            + rna_variables

        arf_decay = sp.sympify('d__arf__all')
        self.add_model_parameter(arf_decay, 1.0,
                                 "decay rate of ARF proteins")

        for _, arf in enumerate(self.arf_variables):
            self.add_reaction([], [arf.name], fwd_rates=[],
                              bwd_rates=[arf_decay])

        if not include_arf_transcription:
            arf_production = sp.sympify('k__arf__all')
            self.add_model_parameter(arf_production, 1.0,
                                     "production rate of ARF proteins")

            for _, arf in enumerate(self.arf_variables):
                self.add_reaction([], [arf.name], fwd_rates=[arf_production],
                                  bwd_rates=[])

        # iaa_production = sp.sympify('k__iaa__all')
        iaa_decay = sp.sympify('d__iaa__all')

        self.add_model_parameter(iaa_decay, 1.0,
                                 "decay rate of Aux/IAAs")

        for _, iaa in enumerate(self.iaa_variables):
            self.add_reaction([], [iaa.name], bwd_rates=[iaa_decay])

        for d in dimer_variables:
            association_rate = sp.sympify(f"k__{d.name}")
            dissociation_rate = sp.sympify(f"d__{d.name}")

            self.add_model_parameter(association_rate,
                                     0.001,
                                     f"Association rate of dimer {d.name}")
            self.add_model_parameter(dissociation_rate,
                                     0.1,
                                     f"Dissociation rate of dimer {d.name}")

            reactants = d.components
            self.add_reaction([str(r) for r in reactants], [d.name], [association_rate],
                              [dissociation_rate])

        for iaa in self.iaa_variables:
            iaa_decay_rate = f"d__{iaa.name}__{self.auxin_variable_label}"
            self.add_model_parameter(iaa_decay_rate, 0.1,
                                     "rate of auxin-mediated decay of Aux/IAA")
            self.add_reaction([iaa.name], [self.empty_set_node.name],
                              [iaa_decay_rate],
                              enzymes=[self.auxin_variable_label])

        for promoter in self.promoter_variables:
            if promoter.name == 'G':
                assert len(promoter.components) == 0
                continue

            assert len(
                promoter.components) == 1, f"Components are {promoter.components}"
            reactant = list(promoter.components)[0]

            promoter_binding_rate = f"k__{promoter.name}"
            promoter_unbinding_rate = f"d__{promoter.name}"

            self.add_model_parameter(promoter_binding_rate,
                                     1.0,
                                     f"binding rate for promoter {promoter.name}")
            self.add_model_parameter(promoter_unbinding_rate,
                                     1.0,
                                     "dissociation rate for promoter"
                                     f"{promoter}")

            self.add_reaction(
                [unbound_promoter.name], [promoter.name],
                [promoter_binding_rate], [],
                enzymes=[reactant]
            )

            self.add_reaction(
                [unbound_promoter.name], [promoter.name],
                [], [promoter_unbinding_rate],
                enzymes=[]
            )

        for promoter in self.promoter_variables:
            for rna in rna_variables:
                rate = f"k__{promoter.name}__{rna.name}"

                self.add_model_parameter(rate,
                                         1.0,
                                         f"Rate of transcription "
                                         f"for {rna.name} for promoter "
                                         f"{promoter.name}"
                                         )

                self.add_reaction(
                    [], [rna.name], [rate],
                    [], [promoter.name], reaction_type="linear"
                )

        for promoter in self.promoter_variables:
            if promoter.name == 'G':
                continue

            reactants = promoter.components

            if len(promoter.components) > 1:
                raise ValueError(
                    "Number of promoter-state components cannot exceed 2")

            bound_complex = list(promoter.components)[0]

            if bound_complex == "dimer":

                reactants = reactants[0].components

                # TODO use different transition rates
                partially_bound_promoter = f"G__{reactants[0].name}"

                if partially_bound_promoter in self.promoter_variables:
                    promoter_binding_rate = f"k__{partially_bound_promoter}__bind__{reactants[1]}"
                    promoter_unbinding_rate = f"d__{promoter}__unbind__{reactants[1].name}"

                    parameter_names = [p.name for p in self._parameters]
                    if promoter_binding_rate not in parameter_names:
                        self.add_model_parameter(promoter_binding_rate)
                        self.add_model_parameter(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter.name], [promoter.name],
                            [promoter_binding_rate], [],
                            enzymes=[reactants[1]])

                        self.add_reaction(
                            [partially_bound_promoter], [promoter],
                            [], [promoter_unbinding_rate]
                        )

                partially_bound_promoter = f"G__{reactants[1]}"
                if partially_bound_promoter in [p.name for p in self.promoter_variables]:
                    promoter_binding_rate = f"k__{partially_bound_promoter}__bind__{reactants[0]}"
                    promoter_unbinding_rate = f"d__{promoter}__unbind__{reactants[0].name}"

                    parameter_names = [p.name for p in self._parameters]
                    if promoter_binding_rate not in parameter_names:
                        self.add_model_parameter(promoter_binding_rate)
                        self.add_model_parameter(promoter_unbinding_rate)
                        self.add_reaction(
                            [partially_bound_promoter.name], [promoter.name],
                            [promoter_binding_rate], [],
                            enzymes=[reactants[0]])

                        self.add_reaction(
                            [partially_bound_promoter.name], [promoter.name],
                            [promoter_unbinding_rate]
                        )

        # Assume constant decay rate for all mRNA
        rna_decay_rate = sp.sympify('d__R')
        self.add_model_parameter(rna_decay_rate,
                                 0.01,
                                 "RNA decay rate")

        for rna in rna_variables:
            self.add_reaction([self.empty_set_node.name], [
                rna.name], [], [rna_decay_rate])
            prot = rna.name.split('__')[1]

            production_rate = f"k__{rna.name}__{prot}"
            rate_default_val = 1.0

            self.add_model_parameter(production_rate, rate_default_val,
                                     f"Rate of production of {prot} from"
                                     f"RNA {rna.name}")

            self.add_reaction([], [prot],
                              fwd_rates=[production_rate],
                              enzymes=[rna.name])

        self.rhs_func = None
        self.cfunc_rhs = None

        self.jac_func = None

    def add_model_parameter(self, label: str, default_value: float = 0.0,
                            description: str = "", strict: bool = True,
                            make_unique: bool = False):
        """Add a new parameter to the model.

        :param label: Label to use for the new parameter. Should be a
            valid Sympy symbol
        :param default_value: Default value for the newly added
            parameter
        :param description: Short description of the new model parameter
        :param strict: If False, overwrite an existing parameter of the
            same name

        :returns: A copy of the created Variable object

        :raises ValueError: When strict is True and a parameter with the
            same label already exists
        :raises ValueError: When label isn't a valid Python variable
            identifier
        """

        label = str(label)
        logging.debug(f"Adding label {label}")

        # Check if label can be converted to a sympy Symbol
        if not label.isidentifier():
            raise ValueError(
                f"label:{label} is not a valid variable identifier")

        if label in _reserved_variable_names:
            raise ValueError(
                f"label:{label} is a reserved variable identifier")

        if strict is None:
            strict = self.strict

        if strict and label in [p.name for p in self._parameters]:
            raise ValueError(f"label {label} already present "
                             f"in model: {self._parameters}")

        variable = Parameter(label, description=description, default_value=default_value,
                             make_unique=make_unique)

        if variable in self.graph.nodes():
            raise ValueError(
                f"Tried to add parameter {label} but already present in state variables")

        self._parameters.add(variable)

        return variable

    def add_state(self, label, strict: bool = True,
                  **kwargs):
        """Add a state variable to the model e.g. a monomer, dimer or other
        chemical species.

        :param label: Label for the new state variable
        :param category: Optional category field for the new state
            variable e.g ARF, Aux/IAA, or promoter
        :param strict: If True an exception is thrown when a state
            variable of the same name already exists in the model

        :return: Copy of the added Varaible object

        :raises ValueError: When strict is True and a duplicate state-
            variable is added
        """

        if str(label) in _reserved_variable_names:
            raise ValueError(
                f"Tried to add variable, {label}, with reserved variable identifier"
            )

        variable = StateVariable(label, **kwargs)

        if strict and label in self.graph.nodes():
            raise ValueError(
                f"Tried to add state {label} to the model but it already exists (strict=True)"
            )

        self.graph.add_node(variable)

        return copy(variable)

    def _get_state_variable_by_name(self, name):
        for var in self.get_state_variables(exclude_auxin=False, exclude_empty=False):
            if var.name == name:
                return var

        raise ValueError(f"State variable with name {name} not found in model")

    def add_reaction(self, reactants: List[str] = [], products: List[str] = [],
                     fwd_rates: List[str] = [],
                     bwd_rates: List[str] = [], enzymes: List[str] = [],
                     add_states: bool = False, strict: bool = None,
                     reaction_type='linear'):
        """Add a chemical reaction to the model.

        Reactions are stored in self.graph

        :param reactants: the reactants of the reaction
        :param products: the products of the reaction (cannot be empty
            or None)
        :param fwd_rates: parameters associated with the forwards
            reaction (reactants -> products)
        :param bwd_rates: parameters associated with the reverse
            reaction (products->reactants)
        :param enzymes: other state variables that contribute to the
            reaction but aren't consumed
        :param add_states: boolean flag, if true missing states are
            added to the model
        :param strict: Boolean flag, if true, an exception is thrown if
            any of the state variables are missing from the model
        """

        reactants = tuple(reactants)
        products = tuple(products)
        fwd_rates = tuple([str(x) for x in fwd_rates])
        bwd_rates = tuple([str(x) for x in bwd_rates])
        enzymes = tuple(enzymes)

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
                states = self.get_state_variable_names()\
                    + [self.empty_set_node.name, self.auxin_variable_label]
                if state not in states:
                    raise Exception(f"Tried to add reaction with state {state} "
                                    f"but {state} does not exist in the model "
                                    f"(strict=True). States are {states}")

        for rate in bwd_rates + fwd_rates:
            if rate not in self.get_parameter_names():
                if strict:
                    raise Exception(f"rate {rate} not found in model. "
                                    f"model has:\n{self._parameters}")
                else:
                    self.add_model_parameter(rate)

        if len(fwd_rates) == 1:
            bwd_rates = bwd_rates * len(reactants)

        if len(bwd_rates) == 1:
            fwd_rates = fwd_rates * len(products)

        bwd_rates = [str(sym) for sym in bwd_rates]
        fwd_rates = [str(sym) for sym in fwd_rates]

        if len(reactants) == 0:
            reactants = (self.empty_set_node.name,)

        for fwd_rate, reactant in zip(fwd_rates, reactants):
            reactant = self._get_state_variable_by_name(reactant)
            for product in products:
                product = self._get_state_variable_by_name(product)
                self.graph.add_edge(
                    reactant, product, weight=fwd_rate,
                    reaction=reaction.__dict__)

        for bwd_rate, reactant in zip(bwd_rates, reactants):
            reactant = self._get_state_variable_by_name(reactant)
            for product in products:
                product = self._get_state_variable_by_name(product)
                self.graph.add_edge(
                    product, reactant, weight=bwd_rate,
                    reaction=reaction.__dict__)

        for enzyme in enzymes:
            enzyme = self._get_state_variable_by_name(enzyme)
            if fwd_rates:
                for product in products:
                    product = self._get_state_variable_by_name(product)
                    self.graph.add_edge(
                        enzyme, product, weight=fwd_rates[0],
                        reaction=reaction.__dict__,
                        color='grey',
                        category='enzymatic'
                    )

            if bwd_rates:
                for reactant in reactants:
                    reactant = self._get_state_variable_by_name(reactant)
                    self.graph.add_edge(
                        enzyme, reactant, weight=bwd_rates[0],
                        reaction=reaction.__dict__,
                        color='grey',
                        category='enzymatic'
                    )

    def _get_reaction_terms(self, reaction: Reaction, variable: str):
        """Get a term for the ODE formulation of a model from a particular
        reaction.

        This function is used to generate an ODE system describing the
        time-evoluation of the model

        :param reaction: a Reaction describing the reaction of interest
        :param variable: the state variable for which terms are being
            generated
        :returns: a list of sympy expressions resulting from the
            reaction
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
        """Get a list of all of the chemical reactions defined in the model.

        Chemical reactions are stored as
        auxin_signalling_models.Reaction objects

        :returns: reaction_set
        """

        return self._reactions_set

    def generate_ode_system(self, parameter_subs_dict: dict = None,
                            state_variables: List[str] = None):
        """Generate a list of sympy expressions comprising an ODE system.

        :params parameter_subs_dict: a dictionary of parameter
            substitutions
        :param state_variables: Optional list of state variables
            defining the ordering of state variables in the model
        :returns: A tuple: the list of ODEs, a list of "free variables"
            e.g. the auxin concentration, a dictionary of parameters,
            and a list of constraints (sympy Expressions)
        """

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        free_variables = [self.auxin_variable_label]

        if state_variables is None:
            state_variables = self.get_state_variable_names()

        odes = [sp.sympify('0') for _ in state_variables]

        for reaction in self._reactions_set:
            for variable in reaction.reactants + reaction.products:
                if variable not in state_variables:
                    continue

                var_index = state_variables.index(variable)
                new_terms = self._get_reaction_terms(reaction, variable)

                odes[var_index] += sp.Add(*new_terms)

        sub_dict = {sp.sympify(self.empty_set_node.name): 1.0}
        odes = sp.Matrix([odes]).T.subs(sub_dict)

        # TODO include the fact that the promoter variables sum to 1
        constraints = []

        if parameter_subs_dict:
            odes = odes.subs(parameter_subs_dict)

        parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

        constraints = [c.subs(parameter_subs_dict) for c in constraints]

        return odes, free_variables, parameters, constraints

    def get_state_variable_names(self, *args, exclude_auxin: bool = True,
                                 exclude_empty: bool = True, **kwargs):
        """Construct a list of the names of the model's state variables.

        :param exclude_auxin: When True, omit the auxin state variable

        :param exclude_empty: When True, omit the empty state

        :returns: a list of the model's state variable names, sorted in
            alphabetical order, optionally excluding the empty-set node, and
            the auxin concentration
        """

        state_vars = self.get_state_variables(*args, **kwargs)
        state_var_labels = [s.name for s in state_vars]

        return sorted(state_var_labels)

    def get_state_variables(self, exclude_auxin: bool = True,
                            exclude_empty: bool = True):
        """ Construct a List of StateVariable objects found in the model

        :param exclude_auxin: When True, omit the auxin state variable
        :param exclude_empty: When True, omit the empty state

        :returns: The constructed list of (copied) StateVariable objects
        """

        return list(sorted([s for s in self.graph.nodes()
                            if (s.name != self.empty_set_node.name or not exclude_empty)
                            and (s.name != self.auxin_variable_label or not exclude_auxin)]))

    def get_all_variables(self):
        """ Constrict a list of string labels for variables found in the model (parameters and state variables)

        :returns: List of strings describing each variable in the model
        """

        return self.get_state_variable_names() + list(self._parameters)

    def make_forward_solver(self, parameter_subs_dict: dict = None):
        """Generate a function which simulates the model.

        :param parameter_subs_dict: An optional dictionary of parameter substitutions

        :returns: a function returning model output under different
            times, parameter values, initial conditions and auxin
            concentrations
        """

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

            rhs_func = self.generate_rhs_func(
                parameter_subs_dict=parameter_subs_dict)

            def deriv_func(t, y, params, aux):
                return rhs_func(y, t, params, aux).flatten()

            jac_func = self.generate_jacobian_func(
                parameter_subs_dict=parameter_subs_dict)
            sol = solve_ivp(
                deriv_func, ts[[0, -1]],
                ic.copy(),
                t_eval=ts,
                method="LSODA",
                args=[params, [aux]],
                jac=lambda t, y, p, aux: jac_func(y, t, p, aux)
            )

            success = True
            return sol.y.T, success

        return solver

    def generate_jacobian_func(self, parameter_subs_dict=None):
        odes, free_variables, _, \
            _ = self.generate_ode_system()

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict

        #  Create column vector of ode expressions
        odes = sp.Matrix([odes])

        state_var_syms = [sp.sympify(n)
                          for n in self.get_state_variable_names()]
        odes = odes.subs(parameter_subs_dict)
        jacobian = odes.jacobian(state_var_syms)

        ic_dict = self.get_default_initial_conditions()

        state_variables = sp.Matrix([s for s in ic_dict])
        params_dict = self.get_default_parameters(parameter_subs_dict)
        param_names = sp.Matrix(list(params_dict.keys()))

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])

        t_sym = sp.sympify("_t")

        _jac_func = sp.lambdify((state_variables, t_sym, param_names,
                                 free_variables), jacobian)

        def jac_func(y, t, p, x):
            ret = _jac_func(y, t, p, x)
            return ret

        return jac_func

    def generate_rhs_func(self, parameter_subs_dict: dict = None):
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

        t_sym = sp.sympify("_t")

        free_variables = sp.Matrix([sp.sympify(x) for x in free_variables])
        rhs_func = sp.lambdify((state_variables, t_sym, param_names,
                                free_variables), odes)

        return rhs_func

    def get_default_initial_conditions(self):
        """Create a dictionary of state variables and corresponding initial
        conditions.

        :returns: a dictionary with a key for each state variable
            corresponding to various default initial conidtions
        """

        state_variables = self.get_state_variables()

        ic_dict = {p.name: p.initial_condition for p in state_variables}

        return _sorted_dict(ic_dict)

    def get_default_parameters(self, parameter_subs_dict: dict = None,
                               ignore_unused: bool = False):
        """Return a default parameter set.

        :param parameter_subs_dict: Dictionary of parameter substitutions to apply
        :param ignore_unused: If True, remove parameters that don't appear in model equations
        :return: a dictionary of parameter values
        """

        if ignore_unused:
            _, _, parameters, _ = self.generate_ode_system(parameter_subs_dict)
            return parameters

        if parameter_subs_dict is None:
            parameter_subs_dict = self.parameter_subs_dict.copy()

        param_dict = {parameter_subs_dict[k.name] if k.name in parameter_subs_dict
                      and len(sp.sympify(parameter_subs_dict[k.name]).free_symbols) == 1
                      else k.name: k.default_value
                      for k in self._parameters}

        # sort parameters
        param_dict = _sorted_dict(param_dict)

        return param_dict

    def draw_graph(self, output_path: str = None, **kwargs):
        """Output a graph of the model using networkx and pygraphvis.

        :param output_path: an optional path for outputting the diagram
        """

        if 'pos' not in kwargs:
            kwargs['pos'] = nx.circular_layout(self.graph)

        if 'with_labels' not in kwargs:
            kwargs['with_labels'] = True

        graph = self.graph.copy()

        categories = sorted([str(val.category)
                            for val in graph.nodes() if val.category != "None"])

        categories = np.unique(categories)

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
        """Construct a LaTeX string describing the model mathematically.

        :returns: A string containg a LaTeX equations for each of the
            models state variables
        """

        odes, free_variables, parameters, \
            constraints = self.generate_ode_system(**kwargs)

        subs_dict = {}

        letters = ['U', 'V', 'W', 'X', 'Y', 'Z',]
        for arf, lab in zip(self.arf_variables, letters):
            subs_dict[arf.name] = lab

        letters = ['u', 'v', 'w', 'x', 'y', 'z',]
        for iaa, lab in zip(self.iaa_variables, letters):
            subs_dict[iaa.name] = lab

        for d in self.dimer_variables:
            components = d.components
            letter1, letter2 = [subs_dict[c] for c in components]
            subs_dict[d.name] = f'D_{letter1}_{letter2}'

        for i, promoter in enumerate(self.promoter_variables):
            if promoter.components:
                subs_dict[promoter.name] = r"G_" + str(i + 1)

        for i, p in enumerate(parameters):
            subs_dict[p] = f"p_{i}"

        odes = odes.subs(subs_dict)
        return sp.latex(odes)

    def output_julia(self, aux_func_name: str = "", aux_func_params: List[str] = [],
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

        state_vars = self.get_state_variable_names()
        default_parameters = self.get_default_parameters(
            parameter_subs_dict=parameter_subs_dict)

        code_str = f"function {self.__class__.__name__}_f_deriv(z, p, t=0.0)\n"

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

        code_str += "\treturn " + "[" + str(julia_code(odes))[1:-1] + "]\n\n"
        code_str += "end"

        return code_str

    def get_jacobian_expr(self, parameter_subs_dict: dict = None):
        """Symbolically the jacobian of the model's derivative function.

        Not implemented

        :param parameter_subs_dict: Dictionary of parameter
            substitutions
        :raises NotImplementedError:
        """
        raise NotImplementedError()

    def remove_parameter(self, rate_name: str, strict: bool = True):
        """Remove a parameter from the model.

        This method removes the parameter from the dictionary,
        self._parameters, and from the default parameter substitution
        dictionary (if it's there). If strict is True, a RuntimeError is
        raised if there is an attempt to remove a non-existant
        parameter.

        :param rate_name: string identifying parameter to be removed
        :param strict: If True, do not allow non-existant parameters to
            be removed
        """

        matching_rates = [r for r in self._parameters if r.name == rate_name]

        assert len(matching_rates) <= 1

        if len(matching_rates) == 0:
            if strict:
                raise RuntimeError(
                    f"Tried to remove parameter {rate_name} but it doesn't exist in the model")
            return

        rate = matching_rates[0]

        if rate in self.parameter_subs_dict:
            self.parameter_subs_dict.remove(rate)

        if rate in self._parameters:
            self._parameters.remove(rate)

    def remove_state(self, state: StateVariable, strict: bool = True):
        """Remove a state from the model.

        Removes some chemical species (i.e. a protein, dimer, or
        promoter state) from the model. If strict is True, an error is
        thrown if there is an attempt to remove a non-existant state.

        :param state: The state to remove
        :param strict: If True, the
        """

        node_list = list(self.graph.nodes())

        if state not in node_list:
            if strict:
                raise RuntimeError(
                    f"Tried to remove state {state} but it doesn't exist in the model")
            else:
                return

        if state in node_list:
            self.graph.remove_node(state)

        if state in self.promoter_variables:
            self.promoter_variables.remove(state)

        if state in self.arf_variables:
            self.arf_variables.remove(state)

        if state in self.iaa_variables:
            self.iaa_variables.remove(state)

        if state in self.rna_variables:
            self.rna_variables.remove(state)

        reaction_set_list = []

        # Remove any reaction that includes this state, also
        for reaction in self._reactions_set:
            if state.name not in list(reaction.products) + list(reaction.reactants)\
               + list(reaction.enzymes):
                reaction_set_list.append(reaction)

        self._reactions_set = set(reaction_set_list)

    def remove_state_variable_by_name(self, name, *args, strict=True, **kwargs):

        matching_states = [
            s for s in self.get_state_variables() if s.name == name]

        if strict and len(matching_states) != 1:
            raise RuntimeError(
                f"Tried removing state {name} for found {len(matching_states)} with this name: {matching_states}")

        state = matching_states[0]

        self.remove_state(state, *args, strict=strict, **kwargs)

    def generate_myokit_model(self, name: str = "",
                              eliminate_promoter_state: str = ""):
        """Generate myokit code for the model.

        This method is unimplemented

        :param name: A label for the created myokit model
            :eliminate_promoter_state: The state to be eliminated from
            the promoter component of the model (str)
        :raises: NotImplementedError
        :raises: ImportError if myokit not installed
        """
        if not myokit:
            raise ImportError("Install extra: pip install mypackage[myokit]")

        if not name:
            name = self.__class__.__name__

        odes, free_variables, _, _ = self.generate_ode_system()

        state_variables = self.get_state_variable_names()
        param_dict = self.get_default_parameters()

        model = myokit.Model(name)

        #  Nuclear Auxin Pathway
        comp_name = 'NAP'
        model.add_component(comp_name)
        comp = model[comp_name]

        if eliminate_promoter_state == '':
            eliminate_promoter_state = sorted(
                [p.name for p in self.promoter_variables])[0]

        # Set up time variable
        model.add_component('engine')
        model['engine'].add_variable('time')
        model['engine']['time'].set_binding('time')
        model['engine']['time'].set_rhs(0.0)

        #  Setup auxin symbol
        comp.add_variable(self.auxin_variable_label)
        comp[self.auxin_variable_label].set_rhs(0.0)
        comp[self.auxin_variable_label].set_binding('pace')

        for parameter, ic in param_dict.items():
            comp.add_variable(parameter)
            comp[parameter].set_rhs(ic)

        for state in state_variables:
            comp.add_variable(state)

        promoter_variables = [sp.sympify(p.name)
                              for p in self.promoter_variables]
        if eliminate_promoter_state:
            mk_expr = 1 - sum(sp.Matrix([s for s in promoter_variables if str(s)
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

    def get_variable_info(self):
        rows = []
        for p in self._parameters:
            row = {"name": p.name, "description": p.description,
                   "default_value": p.default_value}
            rows.append(row)

        parameter_df = pd.DataFrame(rows)

        rows = []
        for state in self.get_state_variables():
            row = {"name": state.name, "description": state.description,
                   "initial_condition": state.initial_condition, "components": state.components,
                   "category": state.category}
            rows.append(row)

        states_df = pd.DataFrame(rows)

        return parameter_df, states_df

    def get_model_name(self):
        """Compute a name for the model which includes the parameters used to
        create it.

        :returns: A string describing the model
        """

        name = f"{self.__class__.__name__}_{self.nARFs}_ARFs_{self.nIAAs}_IAAs"
        if self.include_arf_transcription:
            name += "_with_arf_transcription"
        return name

    def set_parameter_substitution_dict(self, param_subs_dict: dict, strict: bool = False):
        """Define a dictionary of parameter substitutions to use in the model.

        :param parameter_subs_dict:
        :param strict: If true, don't allow parameters to be substituted
            with undefined symbols
        :raises ValueError: When a parameter found in the substitution
            dict isn't found in the model and strict is True
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

    def get_parameters(self):
        return self._parameters.copy()

    def get_promoter_parameters(self):
        """Construct a list of model parameters which relate to the protein-
        promoter dynamics in the model.

        :returns: A list of parameters
        """

        pp_reactions = []
        promoter_variables = [s for s in self.get_state_variable_names()
                              if s[0] == 'G']

        for r in self._reactions_set:
            if np.all(np.isin(r.reactants + r.products, promoter_variables)):
                pp_reactions.append(r)

        pp_params = []

        for r in pp_reactions:
            pp_params += [p for p in r.fwd_rates + r.bwd_rates]

        return pp_params

    def get_parameter_substitution_dict(self):
        """Return the models dictionary of parameter substitutions.

        :returns: a sorted dictionary of parameter substitutions
        """

        return _sorted_dict(self.parameter_subs_dict)

    def pretty_print_parameter_dict(self):
        """Print all model parameters in a LaTeX-maths format.

        :returns: Dictionary of parameter labels in a LaTeX maths format
        """

        ret_dict = {p: self.pretty_print_parameter(p)
                    for p in sorted(list(self.get_default_parameters().keys()))}

        return ret_dict

    def get_parameter_names(self):
        return [p.name for p in self._parameters]

    def generate_state_var_graph(self):
        state_variables = self.get_state_variables()

        graph = nx.graph()

        for s in state_variables:
            graph.add_node(s.name)

        for s in state_variables:
            for c in s.components:
                graph.add_edge(s, c)

        return graph

    @classmethod
    def pretty_print_parameter(cls, par: str):
        """Print a parameter in a LaTeX math format.

        :param par: The parameter label to convert
        :returns: A LaTeX-maths string
        """
        parts = par.split("__")

        if len(parts) < 3:
            return par

        head = parts[0]
        top = parts[1]
        rest = parts[2:]

        return f"{head}_{{{top}_{{{', '.join(rest)}}}}}"


def _sorted_dict(_dict: dict):
    """Returns a dictionary where the keys are sorted (i.e. alphabetically).

    Note that this is only guaranteed to work for Python>=3.7 where
    insertion order is respected

    :param _dict: A dictionary to be sorted
    :returns: A dictionary with the same keys and values but sorted by
        key
    """

    #  Insertion order is respected in Python>=3.7
    ret_dict = {k: _dict[k] for k in sorted(_dict)}
    return ret_dict
