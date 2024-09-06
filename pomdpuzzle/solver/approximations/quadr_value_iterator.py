import numpy as np

from pomdpuzzle.solver.value_iterator import *
from pomdpuzzle.utils.curves import *

class QuadraticValueIterator(ValueIterator):

    def build_immediate_reward(self,log=False):
        dynamics=self.dynamics
        rewards_linears=[]
        # r(b,a)=b1*r(1,a)+b2*r(2,a)+(1-b1-b2)*r(3,a)=b1*(r(1,a)-r(3,a))+b2*(r(2,a)-r(3,a))+r(3,a)
        for a_idx,a in enumerate(dynamics.ACTIONS):
            reward_a=[]
            for s_idx,s in enumerate(dynamics.STATES[:-1]):
                reward_a.append(dynamics.expected_rewards[s_idx][a_idx]-dynamics.expected_rewards[-1][a_idx])
            reward_a.append(dynamics.expected_rewards[-1][a_idx])
            reward_lin=Linear(reward_a,label=a) # Labels identify the action
            rewards_linears.append(reward_lin)

        linear_belief_rewards=LinearSet(rewards_linears)

        return linear_belief_rewards


    def observation_given_belief_action_prob(self,log=False):
        dynamics=self.dynamics
        linears_for_obs_actions=[None for i in range(len(dynamics.OBSERVATIONS))]
        ones_vec=np.ones(len(dynamics.STATES)-1)
        for o_idx,o in enumerate(dynamics.OBSERVATIONS):
            # P(o|b,a)=O(o|s1,a)*( T(s1|s1,a)*b1 + T(s1|s2,a)*b2 + T(s1|s3,a)*(1-b1-b2)  )
            linears=[]
            for a_idx,a in enumerate(dynamics.ACTIONS):
                vec=np.zeros(len(dynamics.STATES))
                for s_prime_idx, s_prime in enumerate(dynamics.STATES):
                    vec[:-1]=dynamics.observation_state_action_prob[o_idx,s_prime_idx,a_idx]*(dynamics.state_state_action_prob[s_prime_idx,:-1,a_idx]-ones_vec)
                    vec[-1]=dynamics.observation_state_action_prob[o_idx,s_prime_idx,a_idx]*dynamics.state_state_action_prob[s_prime_idx,-1,a_idx]

                linears+=[Linear(vec,label=a)]
            linears_for_obs_actions[o_idx]=LinearSet(linears)

        return linears_for_obs_actions


# To approximate with derivative, we must compute the gradients evaluated in 0

    def gradients_bias(self,o_idx,a_idx,s_idx):
        dynamics=self.dynamics
        ones_vec=np.ones(len(dynamics.STATES)-1)
        zero_valued_num=dynamics.observation_state_action_prob[o_idx,s_idx,a_idx]*dynamics.state_state_action_prob[s_idx,-1,a_idx]
        zero_valued_den=np.dot(dynamics.observation_state_action_prob[o_idx,:,a_idx].transpose(),dynamics.state_state_action_prob[:,-1,a_idx])
        grad_num_0=dynamics.state_state_action_prob[s_idx,:-1,a_idx]-np.dot(dynamics.state_state_action_prob[s_idx,-1,a_idx],ones_vec.transpose())
        grad_num=dynamics.observation_state_action_prob[o_idx,s_idx,a_idx]*grad_num_0
        grad_den_0=dynamics.state_state_action_prob[:,:-1,a_idx]-np.matmul(dynamics.state_state_action_prob[:,-1,a_idx][None].transpose(),ones_vec[None])
        grad_den=np.dot(dynamics.observation_state_action_prob[o_idx,:,a_idx].transpose(),grad_den_0)
        overall_grad=(grad_num*zero_valued_den-zero_valued_num*grad_den)/(zero_valued_den*zero_valued_den)

        return -overall_grad,zero_valued_num/zero_valued_den


    def transform_belief_value(self,maximum_quadratic,a_idx,o_idx,log=False):
        dynamics=self.dynamics
        transform_matrix=np.zeros((len(dynamics.STATES),len(dynamics.STATES)))
        for s_prime_idx,s_prime in enumerate(dynamics.STATES):
            gradient, bias=self.gradients_bias(o_idx,a_idx,s_prime_idx)
            transform_matrix[s_prime_idx,:-1]=gradient
            transform_matrix[s_prime_idx,-1]=bias
        maximum_quadratic.label=dynamics.ACTIONS[a_idx]
        transformed_quadratic=maximum_quadratic.linear_transformation(transform_matrix)
        return transformed_quadratic


    def all_values_transform(self,max_value_horizon_prev,log=False):
        dynamics=self.dynamics
        quadratics_for_transformed_belief_value=[None for i in range(len(dynamics.ACTIONS))]
        for a_idx,a in enumerate(dynamics.ACTIONS):
            transformed_quadratics=[]
            for o_idx,o in enumerate(dynamics.OBSERVATIONS):
                transformed_quadratic=self.transform_belief_value(max_value_horizon_prev,a_idx,o_idx) # Quadratic
                transformed_quadratics+=[transformed_quadratic]
            quadratics_for_transformed_belief_value[a_idx]=transformed_quadratics
        return quadratics_for_transformed_belief_value


    def characterize_gamma_terms(self,linears_for_obs_actions,linears_for_transformed_belief_value,log=False):
        dynamics=self.dynamics
        gamma_terms=[]
        for a_idx,a_transformation in enumerate(linears_for_transformed_belief_value):
            action_cubic=None
            for o_idx,obs_linears_prob in enumerate(linears_for_obs_actions):
                observation_cubic=a_transformation[o_idx].multiply_to_linear(obs_linears_prob.linears[a_idx]) # Cubic
                if(not action_cubic):
                    action_cubic=observation_cubic
                else:
                    action_cubic=observation_cubic.sum_cubic(observation_cubic)

            # We go back to Quadratic
            quadratized_action_cubic=action_cubic.to_quadratic()
            gamma_terms.append(quadratized_action_cubic)

        return QuadraticSet(gamma_terms)



    def finalize_sum(self,gamma_terms_quadratics, belief_rewards, gamma=1,log=False):
        dynamics=self.dynamics
        gamma_mul_curves=gamma_terms_quadratics.scalar_multiply(gamma)
        # Belief Rewards: [lin_a0,lin_a1,...,lin_an]
        # gamma_mul_curves: [lin_a0,lin_a0,...,lin_a0,lin_a1,...,lin_a1,...,lin_an,...,lin_an]
        summed_quadratics=[]
        last_processed=0
        for l1 in belief_rewards.linears:
            #print("BELIEF "+l1.label)
            for q2_idx,q2 in enumerate(gamma_mul_curves.quadratics[last_processed:]):
                if(l1.label==q2.label):
                    #print("GAMMA CURVE "+l2.label)
                    summed_quadratics.append(q2.sum_to_linear(l1))
                else:
                    last_processed+=q2_idx
                    break
        final_term=QuadraticSet(summed_quadratics)
        return final_term


    def maximum_vfunct(self,funct):
        maximum_eval_funct=lambda x: funct.maximum_evaluation(x)[0]
        max_vfunct_quadratic=Quadratic.linear_regr_quadratic(maximum_eval_funct, len(self.dynamics.STATES)-1,"No Label")
        return max_vfunct_quadratic

    def delta_function(self,max_value_horizon_cur,new_max_value_horizon):
        delta=np.sqrt(np.sum((max_value_horizon_cur.quadr_matr-new_max_value_horizon.quadr_matr)**2)) # Frobenius distance
        return delta


    def get_partitioned_max_space(self,horizon):
        # Current code works if horizon is a QuadraticSet
        # x^T * Q1 * x >= x^T * Q2 * x ===> x^T(Q1-Q2)*x >=0
        hyperquadratics=horizon.quadratics
        partitions_dict={} # Label1: [[A and B and C] OR [D and E and F]]
        h1_idx=-1
        while (h1_idx+1<len(hyperquadratics)):
            h1_idx+=1
            h1=hyperquadratics[h1_idx]
            h1_part=[]
            h2_idx=-1
            while (h2_idx+1<len(hyperquadratics)):
                h2_idx+=1
                h2=hyperquadratics[h2_idx]
                #print("CONFRONTING "+h1.label+" "+str(h1.a)+" "+h2.label+" "+str(h2.a))
                dec_boundary=h1.quadr_matr-h2.quadr_matr
                #print(dec_boundary)
                if(h1_idx==h2_idx or h1.label==h2.label):
                    continue
                h1_part.append(Quadratic(dec_boundary,h1.label))
            if(not h1_part):
                continue
            if(h1.label not in partitions_dict):
                partitions_dict[h1.label]=[h1_part]
            else:
                partitions_dict[h1.label]+=[h1_part]

        #return partitions,partitions_dict
        return partitions_dict


    def to_act_value_dict(self,horizon):
        act_value_dict={}
        for quadr in horizon.quadratics:
            act_value_dict[quadr.label]=quadr
        return act_value_dict
