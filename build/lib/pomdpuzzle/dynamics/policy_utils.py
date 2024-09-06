
class ActionValueFunction:

    def __init__(self,action_vfunct_dict):
        self.action_vfunct_dict=action_vfunct_dict

    def best_action(self,belief_point):
        max_value=None
        best_action=None
        for action in self.action_vfunct_dict.keys():
            evaluated=self.action_vfunct_dict[action].evaluate(belief_point)
            if(not max_value and max_value!=0):
                max_value=evaluated
                best_action=action
                continue
            if(evaluated>max_value):
                max_value=evaluated
                best_action=action
        
        return max_value,best_action
            
class Policy:

    def __init__(self,action_dependent_vfunct):
        self.action_dependent_vfunct=action_dependent_vfunct

    def retrieve_action(self,belief_point):
        action_name=self.action_dependent_vfunct.best_action(belief_point)[1]
        return action_name

