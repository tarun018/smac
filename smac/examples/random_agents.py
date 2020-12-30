from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np


def main():
    env_args = {
            "map_name": "2_corridors",
            "difficulty": "7",
            "move_amount": 2,
            "step_mul": 8,
            "reward_sparse": False,
            "reward_only_positive": False,
            "reward_negative_scale": 0.5,
            "reward_death_value": 10,
            "reward_scale": True,
            "reward_scale_rate": 20,
            "reward_win": 200,
            "reward_defeat": 0,
            "state_last_action": True,
            "obs_instead_of_state": False,
            "obs_own_health": True,
            "obs_all_health": True,
            "continuing_episode": False,
            "save_replay_prefix": "",
            "heuristic": False,
            "restrict_actions": True,
            "obs_pathing_grid": False,
            "obs_terrain_height": False,
            "obs_last_action": False,
            "bunker_enter_range": 5,
            "seed": 5452
        }
    env = StarCraft2Env(env_args=env_args)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
