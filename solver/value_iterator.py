import numpy as np

class ValueIterator:

    def __init__(self, dynamics):
        self.dynamics=dynamics

    def value_iteration(self,gamma=1,eps=0.1,max_iter=100):
        dynamics=self.dynamics
        belief_rewards=self.build_immediate_reward(dynamics)
        obs_given_ba=self.observation_given_belief_action_prob(dynamics)
        maximum_belief_rewards=self.maximum_vfunct(belief_rewards)
        max_value_horizon_cur=maximum_belief_rewards

        t=0
        delta=1000
        new_horizon=None
        while(delta>=eps):
            if(t>=max_iter):
                break
            print("Iteration number "+str(t+1))
            transformed_vfunct=self.all_values_transform(dynamics,max_value_horizon_cur)
            gamma_terms=self.characterize_gamma_terms(dynamics,obs_given_ba)
            new_horizon=self.finalize_sum(dynamics,gamma_terms,belief_rewards,gamma)
            new_max_value_horizon=self.maximum_vfunct(new_horizon)
            max_value_horizon_cur=new_max_value_horizon
            t+=1
            print("Delta: "+str(delta))

        partition_dict=None
        if(new_horizon):
            print("Getting Max Partition")
            partition_dict=self.get_partitioned_max_space(new_horizon)
        
        print("Iterations Performed: "+str(t+1))
        print("Delta: "+str(delta))

        return max_value_horizon_cur,parition_dict
