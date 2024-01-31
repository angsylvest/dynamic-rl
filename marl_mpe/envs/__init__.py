from gymnasium.envs.registration import register

register(
     id="envs/navigation.py",
     entry_point="env.navigation.py:GridWorldEnv",
     max_episode_steps=300,
)
