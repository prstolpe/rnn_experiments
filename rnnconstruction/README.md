# RNN construction

This directory deals with ways to construct properties in recurrent neural networks
through other ways than optimizing for a task. As such it aims to optimize directly for
properties e.g. fixed point behavior.  
Thereby, the goal is to increase the understanding of how such properties emerge. 

## Optimizing recurrent weight matrices

In the file [rnnconstruction_example](rnnconstruction_example.py) it is made use of the 
[RNNconstructor](rnnconcstruct.py) which can optimize recurrent weights. For instance one could
optimize such a matrix to imitate behavior around fixed points that another network had previously 
been optimized for. Later on one can employ the [PretrainableFlipFlopper](../fixedpointfinder/three_bit_flip_flop.py)
to keep the recurrent kernel fixed and solely optimize input and output mappings.