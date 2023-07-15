import logging
import os
import sys
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import sec2hhmmss, num2text


class Logger:
    def __init__(self, log_dir):
        self.target_dir = log_dir
        self.tf_writer = None
        self.start_time = time.time()
        self.episode_cnt = 0

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        self.writer = SummaryWriter(self.target_dir)

        # Setup logger
        self.logger = logging.getLogger("practice")
        self.logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(self.target_dir + '/logger.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Output logging stream to stdout instead of stderr to avoid red output color in PyCharm
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    def log_episode(self, steps, ret, option_lengths, ep_steps, epsilon):
        self.episode_cnt += 1

        self.logger.info(f"> ep {self.episode_cnt} done. "
                         f"total_steps={num2text(steps)} "
                         f"| return={ret} "
                         f"| episode_steps={ep_steps} "
                         f"| time={sec2hhmmss(self.get_passed_time())} "
                         f"| epsilon={epsilon:.3f}")

        self.writer.add_scalar(tag="returns", scalar_value=ret, global_step=self.episode_cnt)
        self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=self.episode_cnt)

        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            self.writer.add_scalar(tag=f"option_{option}_avg_length",
                                   scalar_value=np.mean(lens) if len(lens) > 0 else 0, global_step=self.episode_cnt)
            self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens) / ep_steps,
                                   global_step=self.episode_cnt)

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon", scalar_value=epsilon, global_step=step)

    def get_passed_time(self):
        """Returns the passed time since initialization, in seconds."""
        return time.time() - self.start_time
