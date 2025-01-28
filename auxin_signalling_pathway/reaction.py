from dataclasses import dataclass

@dataclass(eq=True, order=True, frozen=True)
class reaction:
    products: tuple[str, ...]
    reactans: tuple[str, ...]
    fwd_rates: tuple[str, ...]
    bwd_rates: tuple[str, ...]
    enzymes: tuple[str, ...]
    type: str = 'linear'
