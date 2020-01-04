README

I'm just gonna write up some of the differences between what the paper specifies and what is actually going on

According to the CW definition, we should take the table of params and memory usage as fact.
The memory figures are consistent with the size of the layers decreasing ONLY due to the max
pooling applied.
This means that we need to set padding and stride such that the layers do not reduce naturally

Maintaining the 3x3 kernel since that is consistent with parameter figures, I have set
a (1,1) padding and reduced the stride to (1,1). This maintains height and width across layers.

In order to get the 15.9 mill params value between Conv4 layer and the 1024 fully connected I have done
the following

1. Added a max pool after conv4 layer (reducing height and width by half rounding up)
2. The resulting 11x22 shape with 64 kernels can be flattened to a 15488
3. This 15488 is fully connected to the 1024 fc1 layer (yielding 15.85 mill params)
4. The 1024 layer then connects down to 10 for the classes as declared in paper


Update 4/1/20
Just moved the code over from mlmcnet that I worked on as an attempt without the
foundations of train_cifar.py
