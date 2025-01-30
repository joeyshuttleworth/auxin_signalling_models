from dataclasses import dataclass


@dataclass(eq=True, order=True, frozen=True)
class Reaction:
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
            f"enzymes={self.enzymes}"\
            f"reaction_type={self.reaction_type}"

    def reversed(self):
        return self.__class__(products=self.reactants, reactants=self.products,
                              fwd_rates=self.bwd_rates, bwd_rates=self.fwd_rates,
                              enzymes=self.enzymes, reaction_type=self.reaction_type)
