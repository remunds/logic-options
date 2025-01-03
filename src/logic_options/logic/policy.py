from typing import Optional, Tuple, Type
import torch as th
from gymnasium import spaces
from nsfr.fol.logic import Const
from nsfr.nsfr import NSFReasoner
from nsfr.facts_converter import FactsConverter
from nsfr.valuation import ValuationModule
# from nsfr.valuation import YOLOValuationModule
# from nsfr.logic_utils import build_infer_module, get_lang
from nsfr.utils.logic import build_infer_module, get_lang
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from logic_options.envs.meeting_room import PLAYER_VIEW_SIZE
from logic_options.logic.base import LARK_PATH, LANG_PATH
from logic_options.options.meta_policy import MetaPolicy

# class NudgePolicy(ActorCriticPolicy):
class NudgePolicy(MetaPolicy):
    """Wrapper class for NUDGE. Enables to use NUDGE while preserving the standard SB3 AC interface.
    Takes logic state tensors as input. Hence, the logic state representation needs to be done before
    and outside of this class."""
    # TODO: replace parent class ActorCriticPolicy with BasePolicy and implement own value net

    optimizer_class: Type[th.optim.Adam]
    termination_regularizer: float = 0.00
    termination_mode: str = "raban"

    def __init__(self,
                 env_name: str,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Schedule,
                 net_arch,
                 uses_options: bool = False,
                 device="cuda",
                 **kwargs):

        lang_subdir = "options" if uses_options else "flat"
        dataset = env_name + f"/{lang_subdir}/"
        lang, clauses, bk, atoms = get_lang(LARK_PATH, LANG_PATH, dataset)

        predicates = self._get_predicates(clauses)
        n_predicates = len(predicates)

        if action_space is None:
            action_space = spaces.Discrete(n_predicates)
        else:
            assert isinstance(action_space, spaces.Discrete)
            assert n_predicates == action_space.n

        super().__init__(observation_space, action_space, lr_schedule, net_arch, **kwargs)

        self.predicates = predicates
        self.n_predicates = n_predicates

        val_fn_path = f"src/logic_options/logic/valuation/{env_name}.py"
        valuation_module = ValuationModuleExtended(val_fn_path, lang, device=device)

        facts_converter = FactsConverter(lang=lang, valuation_module=valuation_module, device=device)
        infer_module = build_infer_module(
            clauses,
            atoms,
            lang,
            m=len(self.predicates),
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

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.categorical = CategoricalDistribution(action_dim=action_space.n)

    def _get_predicates(self, clauses):
        """Reads out all predicates and determines which predicate belongs to which action."""
        predicates = []
        for clause in clauses:
            if clause.head.pred.name not in predicates:
                predicate_name = clause.head.pred.name
                predicates.append(predicate_name)
        return predicates

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        pred_probs = self.actor(obs)
        pred_logits = th.log(pred_probs)
        dist = self.categorical.proba_distribution(pred_logits)
        return dist

    # def forward(
    #         self,
    #         obs: th.Tensor,
    #         deterministic: bool = False
    # ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     pred_dist = self.get_distribution(obs)
    #     predicates = pred_dist.get_actions(deterministic)
    #     log_probs = pred_dist.log_prob(predicates)
    #     values = self.predict_values(obs)
    #     return predicates, values, log_probs

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
    
    def get_terminations(self, obs, deterministic=False):
        # global_values, option_distribution = self.meta_policy(obs, deterministic)
        option_distribution = self.get_distribution(obs)
        return self._get_terminations(len(obs), option_distribution, deterministic)

    def _get_terminations(self, n_envs, option_distribution: Distribution, deterministic: bool = False) -> th.Tensor:
        """
        Computes what option should terminate based on the log probabilities of all options of the meta-policy.
        """
        xi = self.termination_regularizer
        sampled_option = option_distribution.get_actions(deterministic)
        if not isinstance(option_distribution, CategoricalDistribution):
            raise NotImplementedError("Only discrete action spaces are supported for now.")

        # get log_probs of all options/actions
        # distribution expects shape (n_envs, n_actions)
        n_actions = option_distribution.action_dim
        log_probs = th.zeros((n_envs, n_actions)).to(self.device)
        action_tensor = th.arange(n_actions).to(self.device)
        for a in range(n_actions):
            log_probs[:, a] = option_distribution.log_prob(action_tensor[a])

        env_indices = th.arange(log_probs.shape[0]).to(self.device)
        probs_regularized = th.exp(log_probs.T) + xi
        if self.termination_mode == "maggi":
            # Maggis way: max(0, 2p(w*|s) / p(w*|s)+p(w|s) - 1)
            # terminations = th.max(th.tensor(0), 2 * th.exp(log_probs[env_indices, sampled_option]) / (probs_regularized + th.exp(log_probs[env_indices, sampled_option])) - 1).T
            terminations = th.clamp((2 * th.exp(log_probs[env_indices, sampled_option]) / (probs_regularized + th.exp(log_probs[env_indices, sampled_option])) - 1).T, 0, 1)
        else:
            # My way: max(0, 1-((p(w|s)+xi)/p(w*|s)))
            # terminations = th.max(th.tensor(0), 1 - (probs_regularized / th.exp(log_probs[env_indices, sampled_option])).T)
            terminations = th.clamp((1 - (probs_regularized / th.exp(input=log_probs[env_indices, sampled_option])).T), 0, 1)

        assert th.all((terminations >= 0) & (terminations <= 1)), "terminations contains values outside the range [0, 1]"

        # sample from bernoulli distribution with p=termination
        if deterministic:
            terminations_sample = terminations > 0.5
        else:
            terminations_sample = th.bernoulli(terminations).bool()
        return terminations_sample, terminations


class ValuationModuleExtended(ValuationModule):
    """Allows to modify the ground_to_tensor() method for custom environments."""

    def ground_to_tensor(self, const: Const, zs: th.Tensor):
        """Ground constant (term) into tensor representations.

            Args:
                const (const): The term to be grounded.
                zs (tensor): The object-centric state representation.
        """
        if const.dtype.name == "local_view":  # MeetingRoom
            # Extract local view
            local_view_shape = [-1, PLAYER_VIEW_SIZE, PLAYER_VIEW_SIZE]
            local_view = zs[..., 9:]
            return th.reshape(local_view, local_view_shape)
        else:
            return super().ground_to_tensor(const, zs)
