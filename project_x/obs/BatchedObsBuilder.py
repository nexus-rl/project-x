from typing import Any
import numpy as np
from rlgym.utils import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState

from project_x.utils.gamestate_encoding import encode_gamestate


class BatchedObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.current_state = None
        self.current_obs = None

    def batched_build_obs(self, encoded_states: np.ndarray) -> Any:
        raise NotImplementedError

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        # Modify current obs to include action
        # player_index=None means actions for all players should be provided
        raise NotImplementedError

    def _reset(self, initial_state: GameState):
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        self.current_state = False
        self.current_obs = None
        self._reset(initial_state)

    def pre_step(self, state: GameState):
        if state != self.current_state:
            self.current_obs = self.batched_build_obs(
                np.expand_dims(encode_gamestate(state), axis=0)
            )
            self.current_state = state

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        for i, p in enumerate(state.players):
            if p == player:
                self.add_actions(self.current_obs, previous_action, i)
                return self.current_obs[i]
