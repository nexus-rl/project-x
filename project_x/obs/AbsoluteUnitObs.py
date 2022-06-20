import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from .DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder
# (if __main__)
# from DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder

from rlgym.utils.common_values import BLUE_GOAL_BACK, ORANGE_GOAL_BACK


class AbsoluteUnitObs(DefaultWithTimeoutsObsBuilder):
    def __init__(self,
            tick_skip=8,
            pad_teams_to=3,
    ):
        super().__init__(tick_skip=tick_skip)

        self.pad_teams_to = pad_teams_to

        self.current_state = None
        self.current_obs = None

        self.LIN_NORM = 1./2300
        self.ANG_NORM = 1./np.pi

        self.BLUE_GOAL_NORM = np.array(BLUE_GOAL_BACK) * self.LIN_NORM
        self.ORANGE_GOAL_NORM = np.array(ORANGE_GOAL_BACK) * self.LIN_NORM

        self.extra_logger = None

    def _inject_extra_logger(self, extra_logger):
        self.extra_logger = extra_logger


    @staticmethod
    def _angle_between(vec1, vec2):
        """Angle from v1 -> v2 in radians"""
        x1, y1 = vec1[..., 0], vec1[..., 1]
        x2, y2 = vec2[..., 0], vec2[..., 1]
        return np.arctan2( x1*y2 - y1*x2, x1*x2 + y1*y2)
    
    @staticmethod
    def _norm_unit(vecs):
        """ Assumed some structured array where the last dimension is coordinates """
        s = vecs.shape
        vecs_norm = np.linalg.norm(vecs, axis=-1)
        vecs_norm_safe = np.copy(vecs_norm)
        vecs_norm_safe[ vecs_norm_safe == 0 ] = 1
        vecs_unit = vecs / vecs_norm_safe.repeat(s[-1]).reshape(s)
        return vecs_norm, vecs_unit


    def _step_state(self, state: GameState):
        self._update_boost_timers(state.boost_pads, self._state.boost_pads)
        self._update_demo_timers(state.players)


        ### Assign state to big matrices
        # (could calculate inverses myself, but they're already calculated once elsewhere, so assigning slightly faster?)

        # Orientations
        # (players), inv, (forward, right, up), (xyz)
        orients = np.zeros((self.n_players, 2, 3, 3))

        # (ball, n_players), (invert), (pos, vel, ang_vel), (x,y,z)
        vecs = np.zeros((self.n_players+1, 2, 3, 3))

        vecs[0, 0] = [
            state.ball.position,
            state.ball.linear_velocity,
            state.ball.angular_velocity
        ]
        vecs[0, 1] = [
            state.inverted_ball.position,
            state.inverted_ball.linear_velocity,
            state.inverted_ball.angular_velocity
        ]

        # Extra player states
        player_states = []

        teams = np.zeros(self.n_players, dtype=int)
        for i_p, p in enumerate(state.players):
            if p.team_num == common_values.ORANGE_TEAM:
                teams[i_p] = 1
            
            player_states.append([
                p.boost_amount,
                int(p.on_ground),
                int(p.has_flip),
                int(p.is_demoed),
                self.demo_timers[i_p]
            ])

            vecs[i_p, 0, ...] = [
                p.car_data.position,
                p.car_data.linear_velocity,
                p.car_data.angular_velocity
            ]
            vecs[i_p, 1, ...] = [
                p.inverted_car_data.position,
                p.inverted_car_data.linear_velocity,
                p.inverted_car_data.angular_velocity
            ]

            # forward [:, 0], right [:, 1], up [:, 2]
            # Transpose for my use, so forward [0, :], ...
            orients[i_p, 0, ...] = p.car_data.rotation_mtx().T
            orients[i_p, 1, ...] = p.inverted_car_data.rotation_mtx().T
        
        # Coord normalization
        vecs[:, :, 0:2, :] *= self.LIN_NORM
        vecs[:, :, 2, :] *= self.ANG_NORM

        # Get magnitude and unit vectors for all vecs so far
        # (ball+players), (inv), (dist, speed, ang speed)
        # (ball+players), (inv), (unit dist, vel, ang vel), (xyz)
        vec_norm, vec_unit = self._norm_unit(vecs)


        ### Ball math

        # (1)
        # (2, 3)
        ball_speed = vec_norm[0, 0, 1]
        ball_vel_unit = vec_unit[0, :, 1, :]
        ball_ang_speed = vec_norm[0, 0, 2]
        ball_ang_vel_unit = vec_unit[0, :, 2, :]

        # (attacc, protecc), (invert), (xyz)
        ball_to_goals = np.zeros((2, 2, 3))
        ball_to_goals[:, 0, :] = [
            self.ORANGE_GOAL_NORM - vecs[0, 0, 0],    # attacc
            self.BLUE_GOAL_NORM - vecs[0, 0, 0],      # protecc
        ]
        ball_to_goals[:, 1, :] = [
            self.BLUE_GOAL_NORM - vecs[0, 1, 0],      # attacc
            self.ORANGE_GOAL_NORM - vecs[0, 1, 0]     # protecc
        ]

        # (2, 2)
        # (2, 2, 3)
        ball_to_goals_r, ball_to_goals_unit = self._norm_unit(ball_to_goals)

        # Two copies of ball vels for attacc/protecc comparison
        ball_vel_tile = np.broadcast_to(vecs[0, :, 1, :], (2, 2, 3))
        # Just a bunch of dot products, but np.dot gets weird above 1D...
        # (2, 2): (a, p), (inv), 
        ball_speed_to_goals = np.sum(ball_vel_tile * ball_to_goals_unit, axis=2)


        ### Compare everything to everything
        # ( 15usec for big subtraction, 31usec for constructing unit vectors )

        # player_1 - player_2 = ptp_pos[2, 3, inv, xyz]
        cmp_tile = np.broadcast_to(vecs, (1+self.n_players, 1+self.n_players, 2, 3, 3))
        # ref (ball, players), cmp (ball, players), (inv), (pos, vel, ang_vel), (xyz)
        vec_cmp = cmp_tile - np.transpose(cmp_tile, [1, 0, 2, 3, 4])

        # Magnitude and unit vectors of all comparisons
        vec_cmp_norm, vec_cmp_unit = self._norm_unit(vec_cmp)


        ### Calculate a couple alignments

        ## The angle between forward and the line to the ball, projected on ground 
        # (ball), (players), (inv), (pos), (xy) -> (players), (inv), (2)
        ptb_xy = vec_cmp[0, 1:, :, 0, :2]
        #  (players), (inv), (forward), (xy) 
        forward_xy = orients[..., 0, :2]
        # -> (players), (inv)
        theta_facing = self._angle_between(forward_xy, ptb_xy)
        # (cos,sin), (players), (inv)
        face_ball = np.stack([np.cos(theta_facing), np.sin(theta_facing)])
        
        # Player->ball vs attacc and protecc
        # vs (attacc, protecc), (invert), (xyz) -> (1,2,3) bcast -> (players, 2, 2)
        attacc_xy = np.broadcast_to(ball_to_goals[0, :, :2], (self.n_players, 2, 2))
        protecc_xy = np.broadcast_to(ball_to_goals[1, :, :2], (self.n_players, 2, 2))
        # -> (players, 2)
        attacc_theta = self._angle_between(ptb_xy, attacc_xy)
        protecc_theta = self._angle_between(ptb_xy, protecc_xy)
        # -> (ab), (cos,sin), (players), (inv)
        attacc_protecc_align = np.stack([
            [np.cos(attacc_theta), np.sin(attacc_theta)],
            [np.cos(protecc_theta), np.sin(protecc_theta)],
        ])

        ### Rotate coordinates to player->ball orientation for each player

        # Angle of player->ball vector (should be the same for inverse)
        # (n_players)
        theta_ptb = np.arctan2(vec_cmp[0, 1:, 0, 0, 0], vec_cmp[0, 1:, 0, 0, 1])

        # Create orientation matrices for every player rotated to every (player->ball) vector
        # (players), (players), (inv), (rot), (xyz)
        ptb_orients = np.tile(orients, (self.n_players, 1, 1, 1, 1))
        # (n_players) ptb frame, (n_players) player, (inv), (rot))
        shape = (self.n_players, self.n_players, 2, 3)
        xs = ptb_orients[..., 0]
        ys = ptb_orients[..., 1]
        # rotation calc remixed from Nexto
        ct = np.cos(theta_ptb).repeat( np.prod(shape[1:]) ).reshape(shape)
        st = np.sin(theta_ptb).repeat( np.prod(shape[1:]) ).reshape(shape)
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        ptb_orients[..., 0] = nx
        ptb_orients[..., 1] = ny

        # Rotate player->(object) unit-vecs to player->ball
        # (players), (ball, players), (inv), (dist, vel diff, ang-vel diff)
        shape = (self.n_players, 1+self.n_players, 2, 3)
        xs = vec_cmp_unit[1:, ..., 0]
        ys = vec_cmp_unit[1:, ..., 1]
        ct = np.cos(theta_ptb).repeat( np.prod(shape[1:]) ).reshape(shape)
        st = np.sin(theta_ptb).repeat( np.prod(shape[1:]) ).reshape(shape)
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        vec_cmp_unit[1:, ..., 0] = nx
        vec_cmp_unit[1:, ..., 1] = ny


        ### 
        # # The angle between forward and the line to the ball, projected on ground 
        # theta_facing = self._angle_between(p.forward[:2], p_to_ball[:2])
        # # sine(angle) upwards (or downwards) along the line to the ball


        ### Build up obs
        team_states = [ 
            [
                state.boost_pads,            # 34
                self.boost_pad_timers,       # 34
            ],
            [
                state.inverted_boost_pads,
                self.inverted_boost_pad_timers,
            ]
        ]

        for i_inv in [0,1]:
            team_states[i_inv] += [
                vecs[0, i_inv, 0],                     # 3  (ball abs position)
                [ball_speed],                            # 1
                ball_vel_unit[i_inv],                  # 3
                [ball_ang_speed],                        # 1
                ball_ang_vel_unit[i_inv],              # 3
                ball_to_goals_r[:, i_inv],                   # 2
                ball_to_goals_unit[:, i_inv, :].flatten(),   # 6
                ball_speed_to_goals[:, i_inv],               # 2
            ]                                          #    = 21

        # sort players by distance to ball
        r_to_ball = vec_cmp_norm[0, 1:, 0, 0]
        r_player_sort = np.argsort(r_to_ball)

        # obs length for each other agent (hand-counted)
        obs_len_other = 38

        self.player_obs = {}
        multi_logs = []
        for i_p, (p, i_inv) in enumerate(zip(state.players, teams)):
            # vec player ind
            i_vp = i_p + 1

            # (boost pads, and ball), player states
            # 68 + 21 + 5 = 94  (+ prev action (8) = 102)
            cur_obs = team_states[i_inv] + [ player_states[i_p] ]

            # Abs player coords
            # 20
            cur_obs += [
                vecs[i_vp, i_inv, 0, :],        # 3 absolute position
                [ vec_norm[i_vp, i_inv, 1] ],   # 4 velocity mag and unit
                vec_unit[i_vp, i_inv, 1, :],  
                [ vec_norm[i_vp, i_inv, 2] ],   # 4 ang-vel mag and unit
                vec_unit[i_vp, i_inv, 2, :],  
                orients[i_p, i_inv].flatten(),  # 9
            ]

            # Player -> ball comparisons
            # 39
            cur_obs += [
                vec_cmp_norm[0, i_vp, i_inv],             # 3  (distance, speed-diff, ang-speed diff)
                vec_cmp_unit[0, i_vp, i_inv].flatten(),   # 9 (p->b), vel-diff, ang-vel diff unit vecs
                vec_cmp_norm[i_vp, 0, i_inv],             # 3 (rotated to ptb) b->p distance, speed-diff, ang-speed diff
                vec_cmp_unit[i_vp, 0, i_inv].flatten(),   # 9 (rotated to ptb) (b->p), vel-diff, ang-vel diff, unit vecs
                ptb_orients[i_p, i_p, i_inv].flatten(),   # 9
                face_ball[:, i_p, i_inv],                 # 2 (cos, sin) (player->ball)(xy) -> forward(xy)
                attacc_protecc_align[:, :, i_p, i_inv].flatten()    # 4 (cos, sin) for attacc protecc
            ]  

            allies, opp = [], []
            n_allies, n_opp = 0, 0
            for i_o in r_player_sort:
                if i_o == i_p:
                    continue

                if teams[i_o] == i_inv:
                    team_obs = allies
                    n_allies += 1
                    if n_allies > (self.pad_teams_to - 1):
                        # Ignore agents over tean pad size
                        continue
                else:
                    team_obs = opp
                    n_opp += 1
                    if n_opp > self.pad_teams_to:
                        # Ignore agents over tean pad size
                        continue
                
                i_vo = i_o + 1
                other_obs = [
                    player_states[i_o],                         # 5
                    # Mag and unit vecs, other -> ball 
                    vec_cmp_norm[0, i_vo, i_inv],               # 3
                    vec_cmp_unit[0, i_vo, i_inv].flatten(),     # 9
                    # Mag and unit, other -> player  (rotated ptb frame)
                    vec_cmp_norm[i_vp, i_vo, i_inv],            # 3
                    vec_cmp_unit[i_vp, i_vo, i_inv].flatten(),  # 9
                    ptb_orients[i_p, i_o, i_inv].flatten()      # 9
                ] # 38

                team_obs += other_obs
            
            ###  Add all the other players to obs and pad teams

            cur_obs += allies
            ally_deficit = self.pad_teams_to - n_allies - 1
            if ally_deficit > 0:
                cur_obs += [ [0.0] * obs_len_other * ally_deficit ] 

            cur_obs += opp
            opp_deficit = self.pad_teams_to - n_opp
            if opp_deficit > 0:
                cur_obs.extend( [ [0.0] * obs_len_other * opp_deficit ] )
        
            self.player_obs[p.car_id] = np.concatenate(cur_obs)
            

            if self.extra_logger is not None:
                multi_logs += [ {
                    "car_speed": vec_norm[i_vp, i_inv, 0]/self.LIN_NORM,
                    "car_height": p.car_data.position[2],
                    "touch_grass": float(p.on_ground),
                    "boost_held": float(p.boost_amount),
                    "dist_to_ball": vec_cmp_norm[0, i_vp, i_inv, 0]/self.LIN_NORM,
                    "speed_to_ball": vec_cmp_norm[0, i_vp, i_inv, 1]/self.LIN_NORM,
                    "face_ball": face_ball[0, i_p, i_inv],
                    "ball_goal_align": attacc_protecc_align[0, 0, i_p, i_inv] + attacc_protecc_align[1, 0, i_p, i_inv]
                } ]
        
        if self.extra_logger is not None:
            # Ugly one-liner to take average of all values so far
            multi_log = { k: np.mean([ ml[k] for ml in multi_logs ]) for k in multi_logs[0].keys()}

            multi_log["ball_speed"] = np.linalg.norm(state.ball.linear_velocity)
            multi_log["ball_height"] = state.ball.position[2]

            self.extra_logger.log_multi(multi_log)

        self._state = state


    def reset(self, initial_state: GameState):
        self.current_state = False
        self.current_obs = None
        DefaultWithTimeoutsObsBuilder.reset(self, initial_state)

    def pre_step(self, state: GameState):
        if state != self.current_state:
            self._step_state(state)
            self.current_state = state

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        return np.concatenate((previous_action, self.player_obs[player.car_id]))


if __name__ == '__main__':
    # Also stolen from Rolv
    import rlgym

    env = rlgym.make(use_injector=True, self_play=True, team_size=3,
                     obs_builder=AbsoluteUnitObs())

    states = []
    actions = [[np.zeros(8)] for _ in range(6)]
    done = False
    obs, info = env.reset(return_info=True)
    obss = [[o] for o in obs]
    states.append(info["state"])
    while not done:
        act = [env.action_space.sample() for _ in range(6)]
        for a, arr in zip(act, actions):
            arr.append(a)
        obs, reward, done, info = env.step(act)
        for os, o in zip(obss, obs):
            os.append(o)
        states.append(info["state"])