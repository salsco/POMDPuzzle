import numpy as np

class Dynamics:
  def __init__(self,reward_state_action_prob,state_state_action_prob,observation_state_action_prob,STATES,ACTIONS,OBSERVATIONS,REWARDS):
    self.reward_state_action_prob=reward_state_action_prob
    self.state_state_action_prob=state_state_action_prob
    self.observation_state_action_prob=observation_state_action_prob
    self.STATES=STATES
    self.ACTIONS=ACTIONS
    self.OBSERVATIONS=OBSERVATIONS
    self.REWARDS=REWARDS
    self.expected_rewards=self.get_expected_rewards()

  def get_expected_rewards(self):
    # R(s,a)
    expected_rewards=np.zeros((self.reward_state_action_prob.shape[1],self.reward_state_action_prob.shape[2]))
    for reward_idx in range(self.reward_state_action_prob.shape[0]):
      expected_rewards+=reward_state_action_prob[reward_idx]*REWARDS[reward_idx]

    return expected_rewards
