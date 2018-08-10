# Tensorflow Dataset API
Tensorflow Dataset is a new but very powerful api for feedding data into a computation graph.

## How to create a Dataset
### Create from generator
Sometimes, you prepared data in other form, which can not be used directly used by tf and you want to preprocess it. So you can create a generator with this api.
```python
import tensorflow as tf
import itertools
from tensorflow.data import Dataset
# help(tf.data.Dataset)
# from_generator(generator, output_types, output_shapes=None, args=None)
    
def gen():
  for i in itertools.count(1):
  yield (i, [1] * i)

ds = Dataset.from_generator(gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
value = ds.make_one_shot_iterator().get_next()
with tf.Session() as sess:
  print(sess.run(value))  # (1, array([1]))
  print(sess.run(value))  # (2, array([1, 1]))
```

### Create from a tensor
You can create a Dataset from a tensor, constructed from a numpy array. But if the array is very large, this method will waste a lot of memory or even cause error.
```
import tensorflow as tf
import itertools
from tensorflow.data import Dataset
import numpy as np
x = np.random.randint(1, 255, (28, 28))
x_tensor = tf.convert_to_tensor(x)
x_ds = Dataset.from_tensors([x_tensor])
print(x_ds)
```

### Create from tensor slice
This is an important method. The real function is to create a dataset which slice the tensor according to the first dimension. For example, if there is a tensor shaped (5, 6), the method will create a dataset which contains 5 samples, and each sample contains 6 numbers.
```
x_ds = Dataset.from_tensor_slices(np.random.uniform(size=(5, 6)))
print(x_ds) # <TensorSliceDataset shapes: (6,), types: tf.float64>
x_ds2 = dataset = tf.data.Dataset.from_tensor_slices(
  (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2)))
)
print(x_ds2)
x_ds_iter = x_ds2.make_one_shot_iterator()
one_element = x_ds_iter.get_next()
print(sess.run(one_element))
```

### Create from file pattern
```python
# Here ignore imports. You should add them.
x_ds = Dataset.list_files('./train/*.jpg') # Containing file names
# now you can test it.
x_ds_iter = x_ds.make_one_shot_iterator()
one_element = x_ds_iter.get_next()
print(sess.run(one_element))
```

### Create from a range of integers.
```python
# Here ignore imports. You should add them.
x_ds = Dataset.range(5, 1, -2) # The same as range of python: start, stop, step
# now you can test it.
x_ds_iter = x_ds.make_one_shot_iterator()
one_element = x_ds_iter.get_next()
print(sess.run(one_element))
```

