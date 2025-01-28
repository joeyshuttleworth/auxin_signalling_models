from dataclasses import dataclass

@dataclass(eq=True, order=True, frozen=True)
class Reaction:
    products: tuple[str, ...]
    reactants: tuple[str, ...]
    fwd_rates: tuple[str, ...]
    bwd_rates: tuple[str, ...]
    enzymes: tuple[str, ...]
    reaction_type: str = 'linear'

    def __str__(self):
        return f"Reaction: products={self.products}, reactant={self.reactants}, "\
            f", fwd_rates={self.fwd_rates}, bwd_rates={self.bwd_rates}, "\
            f"enzymes={self.enzymes}"

