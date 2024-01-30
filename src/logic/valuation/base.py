from typing import Sequence, Dict, Any
from abc import ABC

import torch as th
from torch import nn
from nsfr.fol.language import Language
from nsfr.fol.logic import Atom, Term


class ValuationFunction(nn.Module, ABC):
    """Base class for valuation functions used inside valuation modules."""

    def __init__(self, pred_name: str):
        super().__init__()
        self.pred_name = pred_name

    def forward(self, **kwargs) -> th.Tensor:
        raise NotImplementedError()

    def bool2probs(self, bool_tensor: th.Tensor) -> th.Tensor:
        """Converts a Boolean tensor into a probability tensor by assigning
        probability 0.99 for True
        probability 0.01 for False."""
        return th.where(bool_tensor, 0.99, 0.01)


class ValuationModule(nn.Module, ABC):
    """Turns logic state representations into valuated atom probabilities according to
    the environment-specific valuation functions."""

    lang: Language
    device: th.device
    val_fns: Dict[str, ValuationFunction]  # predicate names to corresponding valuation fn
    attrs: Dict[Any, th.Tensor]  # attribute terms to corresponding one-hot encoding
    dataset: str

    def __init__(self, lang, device, pretrained=True):
        super().__init__()
        self.lang = lang
        self.device = device
        self.pretrained = pretrained

        # Prepare valuation functions
        val_fns = self.init_valuation_functions()
        pred_name_to_val_fn = {}
        for val_fn in val_fns:
            pred_name_to_val_fn[val_fn.pred_name] = val_fn
        self.val_fns: Dict[str, ValuationFunction] = pred_name_to_val_fn

    def init_valuation_functions(self) -> Sequence[ValuationFunction]:
        raise NotImplementedError()

    def forward(self, zs: th.Tensor, atom: Atom):
        """Convert the object-centric representation to a valuation tensor.

            Args:
                zs (tensor): The object-centric representation (the output of the YOLO model).
                atom (atom): The target atom to compute its probability.

            Returns:
                A batch of the probabilities of the target atom.
        """
        val_fn = self.val_fns[atom.pred.name]
        # term: logical term
        # args: the vectorized input evaluated by the value function
        args = [self.ground_to_tensor(term, zs) for term in atom.terms]
        return val_fn(*args)

    def ground_to_tensor(self, term: Term, zs: th.Tensor):
        """Ground terms into tensor representations.

            Args:
                term (term): The term to be grounded.
                zs (tensor): The object-centric representation.
        """
        raise NotImplementedError()
