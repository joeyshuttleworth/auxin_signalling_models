from dataclasses import dataclass


@dataclass(eq=True, order=True, frozen=True)
class Reaction:
    """ Each reaction is stored as a Reaction dataclass object, which lists the
     reactants and products, as well as relevant model parameters, any other
     state variables which are necessary for the reaction (enzymes). Each
     reaction may be of a different type. Three reaction types are implemented:

     - linear: using the law of mass action (simple production and linear decay
     are possible by using the empty-set node as either a reactant or product)
    """

    reactants: tuple[str, ...]
    products: tuple[str, ...]
    fwd_rates: tuple[str, ...]
    bwd_rates: tuple[str, ...]
    enzymes: tuple[str, ...]
    reaction_type: str = 'linear'

    def __str__(self):
        return f"Reaction: reactants={self.reactants},"\
            f"products={self.products}, "\
            f"fwd_rates={self.fwd_rates}, bwd_rates={self.bwd_rates}, "\
            f"enzymes={self.enzymes}, "\
            f"reaction_type={self.reaction_type}"

    def reversed(self):
        """Swap reactants and products, and the corresponding parameters.
        Providing reversibilty, the resulting reaction should behave
        identically."""
        return self.__class__(products=self.reactants,
                              reactants=self.products,
                              fwd_rates=self.bwd_rates,
                              bwd_rates=self.fwd_rates, enzymes=self.enzymes,
                              reaction_type=self.reaction_type)
