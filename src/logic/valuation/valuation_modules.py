from typing import Sequence, Dict, Any
from abc import ABC

import torch as th
from torch import nn
from nsfr.fol.language import Language
from nsfr.fol.logic import Atom, Term


class ValuationModule(nn.Module, ABC):
    """Turns logic state representations into valuated atom probabilities according to
    the environment-specific valuation functions."""

    lang: Language
    device: th.device
    layers: Sequence[nn.Module]  # list of valuation functions
    vfs: Dict[str, nn.Module]  # predicate names to corresponding valuation fn
    attrs: Dict[Any, th.Tensor]  # attribute terms to corresponding one-hot encoding
    dataset: str

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.pretrained = pretrained
        self.layers, self.vfs = self.init_valuation_functions()

    def init_valuation_functions(self) -> (Sequence[nn.Module], Dict[str, nn.Module]):
        raise NotImplementedError()

    def forward(self, zs: th.Tensor, atom: Atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representation (the output of the YOLO model).
                atom (atom): The target atom to compute its probability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        # term: logical term
        # arg: vector representation of the term
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        val_fn = self.vfs[atom.pred.name]
        return val_fn(*args)

    def ground_to_tensor(self, term: Term, zs: th.Tensor):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        raise NotImplementedError()
