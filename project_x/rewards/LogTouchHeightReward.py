from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.common_values import CEILING_Z

import numpy as np

class LogTouchHeightReward(RewardFunction):
    """
    a ball height touch reward (removed touch ground condition to encourage wall touches)
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """
    
    def __init__(self, min_height=92, exp=2):
        self.min_height = min_height
        self.exp = exp
        self.max_height = CEILING_Z - self.min_height

        self.extra_logger = None

    def _inject_extra_logger(self, extra_logger):
        self.extra_logger = extra_logger
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and state.ball.position[2] >= self.min_height:
            clipped_height = np.clip(state.ball.position[2] - self.min_height, 0, self.max_height)

            norm_height = clipped_height / self.max_height
            if self.extra_logger is not None:
                multi_log = {
                    "TouchHeight": 1,
                    "TouchHeight_norm_height": norm_height,
                }
                self.extra_logger.log_multi(multi_log)
    
            return np.power(norm_height, self.exp)
    
        return 0
