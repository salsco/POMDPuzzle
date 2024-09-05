import numpy as np

class BoundariesViewer:

    def print_to_desmos(self,boundary_list, two_dimensional=False):
        for b_idx,boundary in enumerate(boundary_list):
            extended_form=self.extend_boundary_form(boundary)
            if(two_dimensional):
                extended_form=extended_form.replace("x1","x")
                extended_form=extended_form.replace("x2","y")
            if(b_idx==0):
                print(extended_form+">0", end=" ")
            else:
                print("",end="\\left\\{")
                print(extended_form+">0", end="\\right\\}")


class QuadraticBoundariesViewer(BoundariesViewer):

    def extend_boundary_form(self,boundary):
        return boundary.quadraticform_string()

