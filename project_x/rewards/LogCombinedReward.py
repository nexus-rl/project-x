from typing import Any, Optional, Tuple, overload, Union
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.combined_reward import CombinedReward
import numpy as np

class LogCombinedReward(CombinedReward):
    """
    A reward composed of multiple rewards.
    """

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward.
        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__(reward_functions, reward_weights)

        self.extra_logger = None
        self.reward_names = [ type(rew_fn).__name__ for rew_fn in self.reward_functions ]

    def _inject_extra_logger(self, extra_logger):
        self.extra_logger = extra_logger

        # Pass the extra logger into sub-reward functions to gather their data
        for reward_fn in self.reward_functions:
            if hasattr(reward_fn, "_inject_extra_logger") and callable(reward_fn._inject_extra_logger):
                reward_fn._inject_extra_logger(extra_logger)

    def get_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.
        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.
        :return: The combined rewards for the player on the state.
        """
        weighted_rewards = []
        for func, weight in zip(self.reward_functions, self.reward_weights):
            reward = weight * func.get_reward(player, state, previous_action)

            weighted_rewards += [reward]
        
        if self.extra_logger is not None:
            multi_log = {}
            for w_r, reward_name in zip(weighted_rewards, self.reward_names):
                # print(reward_name, w_r)
                multi_log[reward_name] = w_r

            self.extra_logger.log_multi(multi_log)

        return np.sum(weighted_rewards)

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        return self.get_reward(player, state, previous_action)