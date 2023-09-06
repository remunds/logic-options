from __future__ import annotations

from typing import Callable

import numpy as np

from utils.common import is_sorted


class ParamScheduler:
    """Represents a piecewise-defined mathematical function used for parameter scheduling."""

    def __init__(self,
                 init_value: float,
                 decay_mode: str = 'const',
                 half_life_period: int = None,
                 warmup_length: int = 0,
                 minimum: float = 0.0,
                 milestones: dict[float, float] = None):
        """
        :param init_value: Start value
        :param decay_mode: The function type used to decrease the value. Choose from:
            'const' for a constant function
            'step' for a step function
            'lin' for a piecewise linear function
            'exp' for an exponential function
        :param half_life_period: In case of an exponential function the period length after which
            the value is half compared to before.
        :param warmup_length: Inserts a segment of this length where the value increases linearly
            from zero to init_value. May be useful for learning rate. If specified, shifts milestones
            to the right by warmup_length.
        :param minimum: Minimum value which is never undercut over the course of practice
        :param milestones: Dictionary of the form {x1: y1, x2: y2, ...} defining some
            specific function steps.
        """
        # Check user inputs
        assert init_value >= minimum, "Initial value must lie above minimum value."

        if decay_mode == "lin":
            assert milestones is not None, "Linear decay mode requires milestones."
        elif decay_mode == "exp":
            assert half_life_period is not None
        elif decay_mode == "step":
            assert milestones is not None
        elif decay_mode != "const":
            raise ValueError("Invalid decay mode given:", decay_mode)

        self.init_value = init_value
        self.dynamic = decay_mode != "const"
        self.decay_mode = decay_mode

        self.warmup_length = warmup_length

        self.half_life_period = half_life_period
        self.decay_rate = 0.5 ** (1 / half_life_period) if half_life_period is not None else None

        self.minimum = minimum

        self.milestones = milestones
        if self.milestones is not None:
            self.milestones_x = np.array(list(milestones.keys())) + self.warmup_length
            assert is_sorted(self.milestones_x)
            self.milestone_y = np.array(list(milestones.values()))

            assert np.all(self.milestone_y >= self.minimum), \
                "Milestone values are required to lie above the global minimum."

    def get_value(self, current_trans_no):
        if current_trans_no < self.warmup_length:
            return self.init_value * current_trans_no / self.warmup_length
        elif self.dynamic:
            decay_transitions = current_trans_no - self.warmup_length
            if self.decay_mode == "exp":
                return max(self.init_value * self.decay_rate ** decay_transitions, self.minimum)
            else:
                milestone_cnt = np.sum(self.milestones_x <= current_trans_no)
                if milestone_cnt == len(self.milestones_x) and self.milestone_y is not None:
                    return self.milestone_y[-1]
                elif self.decay_mode == "lin":
                    # Linear interpolation
                    if milestone_cnt == 0:
                        segment_start = self.warmup_length
                        start_value = self.init_value
                    else:
                        segment_start = self.milestones_x[milestone_cnt - 1]
                        start_value = self.milestone_y[milestone_cnt - 1]

                    end_value = self.milestone_y[milestone_cnt]
                    segment_end = self.milestones_x[milestone_cnt]

                    segment_progress = (current_trans_no - segment_start) / (segment_end - segment_start)
                    return start_value * (1 - segment_progress) + end_value * segment_progress
                elif self.decay_mode == "step":
                    if milestone_cnt == 0:
                        return self.init_value
                    else:
                        return self.milestone_y[milestone_cnt - 1]
        else:
            return self.init_value

    def get_config(self):
        config = {"init_value": self.init_value,
                  "decay_mode": self.decay_mode}
        if self.warmup_length > 0:
            config.update({"warmup_length": self.warmup_length})
        if self.dynamic:
            if self.decay_mode == "exp":
                config.update({"half_life_period": self.half_life_period,
                               "minimum": self.minimum})
            elif self.decay_mode in ["step", "lin"]:
                config.update({"milestones": self.milestones})
        return config


def get_linear_schedule(initial_value: float) -> Callable[[float], float]:
    def linear(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return linear


def get_exponential_schedule(initial_value: float, half_life_period: float = 0.25) -> Callable[[float], float]:
    """It holds exponential(half_life_period) = 0.5. If half_life_period == 0.25, then
    exponential(0) ~= 0.06"""
    assert 0 < half_life_period < 1

    def exponential(progress_remaining: float) -> float:
        return initial_value * np.exp((1 - progress_remaining) * np.log(0.5) / half_life_period)

    return exponential


def maybe_make_schedule(args):
    if isinstance(args, (int, float)):
        return args
    elif isinstance(args, dict):
        schedule_type = args.pop("schedule_type")
        if schedule_type == "linear":
            return get_linear_schedule(**args)
        if schedule_type == "exponential":
            return get_exponential_schedule(**args)
        elif schedule_type is None:
            return args["initial_value"]
        else:
            ValueError(f"Unrecognized schedule type {schedule_type} provided.")
    else:
        raise ValueError("Invalid parameter schedule specification.")
