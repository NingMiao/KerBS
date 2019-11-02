# KerBS
Code for &lt;Kernelized Bayesian Softmax for Text Generation>

This is a implemetation of KerBS in hvd and tensorflow platform.

-Use KerBS_top.py to build the top layers of generation model.

-Use KerBS_hook.py to dynamically train KerBS.

-Reallocation strategy is saved in reallocate.py and you can DIY better strategies.
