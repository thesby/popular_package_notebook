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

## Dataset transformation
` map(), batch(), shuffle() and repeat() `
map: mapping elements
batch: pack data in the form of batch. Very useful when training or testing.
shuffle: randomize the data.
repeat: repeat the dataset in specified times, such as twice.
```
dataset = tf.data.Dataset.from_tensor_slices((np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.random.uniform(size=(5, 2))))
dataset2 = dataset.map(lambda x, y: (x, y+1)) # the random array will add 1 elementwise.
sess.run(dataset.make_one_shot_iterator().get_next())
sess.run(dataset2.make_one_shot_iterator().get_next())
dataset3 = dataset2.batch(2)
sess.run(dataset3.make_one_shot_iterator().get_next())
```

## A simple toy to use Dataset
```
# read image and resize
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label
 
# get file list
filenames = tf.constant(['./train/%05d.jpg'%p for p in range(100)])
labels = tf.constant(np.random.randint(0,2, (100,))  # you can also read from a label file
 
# dataset: each element is (filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
 
# image now can be read by map ==> (image_resized, label)
dataset = dataset.map(_parse_function)
 
# packing into batches ==> (image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)
```

# Tensorflow FixedLengthRecordDataset, TFRecordDataset, TextLineDataset
1. TextLineDataset: input a list of text file list, and tensorflow will read them line by line.
2. FixedLengthRecordDataset: read data from file with fixed length.
3. TFRecordDataset: this is very simple, just read TFRecord files

# Iterator
we have used `make_one_shot_iterator()` before, but there are some other iterators.
* initializable iterator
* reinitializable iterator
* feedable iterator

### initializable iterator
It must be initialized by `sess.run()`.
```python
# copy from somewhere
limit = tf.placeholder(dtype=tf.int32, shape=[])
dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 10})
    for i in range(10):
      value = sess.run(next_element)
      assert i == value
```
There is an example to use iterator to read very large data from disk, and avoid load them into your computation graph.
```python
# Read very large feature file from disk.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]
 
features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
 
dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

```

### reinitializable iterator
It's to create many datasets with the same pattern, such as training dataset and testing dataset.
```python
# Reinitializable iterator to switch between Datasets
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        sess.run([features, labels])
    sess.run(test_init_op) # switch to val dataset.
    print(sess.run([features, labels]))
```

# Real example to use dataset in neural network.
```python
EPOCHS = 10
BATCH_SIZE = 16
# using two numpy arrays
features, labels = (np.array([np.random.sample((100,2))]), 
                    np.array([np.random.sample((100,1))]))
dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
```
# Some useful tutorials
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset
* https://blog.csdn.net/dqcfkyqdxym3f8rb0/article/details/79342369
