from typing import Optional, Tuple
import torch as th
from gymnasium import spaces
from nsfr.nsfr import NSFReasoner
from nsfr.facts_converter import FactsConverter
from nsfr.logic_utils import build_infer_module, get_lang
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution, Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from logic.valuation_modules import SeaquestValuationModule, FreewayValuationModule

LARK_PATH = "in/logic/lark/exp.lark"
LANG_PATH = "in/logic/data/lang"

VALUATION_MODULES = {
    "seaquest": SeaquestValuationModule,
    "freeway": FreewayValuationModule,
}


class NudgePolicy(ActorCriticPolicy):
    """Wrapper class for NUDGE. Enables to use NUDGE while preserving the standard SB3 AC interface.
    Takes logic state tensors as input. Hence, the logic state representation needs to be done before
    and outside of this class."""

    def __init__(self,
                 env_name: str,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 net_arch,
                 device="cpu",
                 **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, **kwargs)

        lang, clauses, bk, atoms = get_lang(LARK_PATH, LANG_PATH, "", env_name)

        # Collect predicates (actions)
        predicates = []
        for clause in clauses:
            if clause.head.pred.name not in predicates:
                predicates.append(clause.head.pred.name)
        self.predicates = predicates

        if env_name not in VALUATION_MODULES.keys():
            raise NotImplementedError(f"No valuation module implemented for env '{env_name}'.")

        valuation_module = VALUATION_MODULES[env_name](lang=lang, device=device)
        facts_converter = FactsConverter(lang=lang, valuation_module=valuation_module, device=device)
        infer_module = build_infer_module(
            clauses,
            atoms,
            lang,
            m=len(predicates),
            infer_step=2,
            train=True,
            device=device
        )
        self.actor = NSFReasoner(facts_converter=facts_converter,
                                 infer_module=infer_module,
                                 atoms=atoms,
                                 bk=bk,
                                 clauses=clauses,
                                 train=True)

        self.n_actions = len(self.predicates)
        self.uniform = Categorical(th.ones(size=(self.n_actions,), device=device) / self.n_actions)

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        action_probs = self.actor(obs)
        dist = CategoricalDistribution(action_probs)
        return dist

    def forward(
            self,
            obs: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        dist = self.get_distribution(obs)
        actions = dist.get_actions(deterministic)
        log_probs = dist.log_prob(actions)
        values = self.predict_values(obs)
        return actions, values, log_probs

    def evaluate_actions(
            self,
            obs: th.Tensor,
            actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        values = self.predict_values(obs)
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

# TODO: check if NUDGE policy parameters are learned (they should change after a training step)
