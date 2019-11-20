# Kernelized Bayesian Softmax for Text Generation (KerBS)

KerBS is a powerful substitute for Softmax. Please refer to our [paper](https://arxiv.org/abs/1911.00274) or [poster](https://github.com/NingMiao/KerBS/blob/master/poster/poster_KerBS.pdf) for details.

## Requirements
- python
  - `==3.4`

- python packages
  - TensorFlow `== 1.4.0` (Other versions are not tested.)
  - numpy
  - pickle
  - horovod (Running without horovod needs some slight modifications.)
  
## Running
- To replace Softmax with KerBS, change the output layer to `KerBS_top.kerbs_top`, for example, 
  ```python
  logits = KerBS_top.kerbs_top(top_features=h, bayes_component=3, top_dimension=10000, dtype=tf.float32)
  ```

- To dynamically allocate senses, add `HvdReallocateHook`.

- To customize reallocation strategies, change  `get_new_sense_allocate` in `reallocate.py`
