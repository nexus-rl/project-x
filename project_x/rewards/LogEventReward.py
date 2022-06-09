from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
import numpy as np


class LogEventReward(EventReward):
    def __init__(self, **kwargs):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__(**kwargs)

        self.extra_logger = None
        self.names = ["goal", "team_goal", "concede", "touch", "shot", "save", "demo", "boost_pickup"]

    def _inject_extra_logger(self, extra_logger):
        self.extra_logger = extra_logger

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        if self.extra_logger is not None:
            new_events = np.where(diff_values > 0)[0]
            for i_event in new_events:
                event_name = self.names[i_event]
                self.extra_logger.log(event_name, 1)

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward