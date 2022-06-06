from argparse import Action
from project_x.obs.DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder
from distrib_rl.Environments.Custom.RocketLeague import ActionParserFactory, \
    ObsBuilderFactory, RewardFunctionFactory, StateSetterFactory, \
    TerminalConditionsFactory

def register_custom_action_parsers():
    # example action parser registration:
    # ActionParserFactory.register_action_parser("custom_action_parser", CustomActionParser)
    pass

def register_custom_obs_builders():
    ObsBuilderFactory.register_obs_builder("default_with_timeouts", DefaultWithTimeoutsObsBuilder)

def register_custom_reward_functions():
    # example reward function registration:
    # RewardFunctionFactory.register_reward_function("custom_reward_function", CustomRewardFunction)
    pass

def register_custom_state_setters():
    # example state setter registration:
    # StateSetterFactory.register_state_setter("custom_state_setter", CustomStateSetter)
    pass

def register_custom_terminal_conditions():
    # example terminal condition registration:
    # TerminalConditionsFactory.register_terminal_condition("custom_terminal_condition", CustomTerminalCondition)
    pass

def register_customizations():
    register_custom_action_parsers()
    register_custom_obs_builders()
    register_custom_state_setters()
    register_custom_reward_functions()
    register_custom_terminal_conditions()
