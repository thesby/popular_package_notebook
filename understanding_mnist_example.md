Tensorflow official models give an example how to use datasets and estimators to build a full model with training, validation and testing. Here is the notes.
The most import entry is `run_mnist`, so I start here.

* `model_helpers.apply_clean(flags_obj)` is to clean model_dir if you set clean in command.
* `model_function = model_fn` is the model function which you should return the output tensorflow of your model. In the example, a model built with keras is demonstrated. Actually, you can use any supported tools, such as low api of tensorflow, `tf.nn` and `tf.layers`.
* we ignore all about multi-gpu, this is not important for playing.
* `data_format = flags_obj.data_format`. It's to choose input data format which used when building keras model.
* Next, build the estimator. `model_fn` is your model function that has fixed input parameter form or function signation: features, labels, mode, params. params will be passed into `model_fn` as parameter params.
```
mnist_classifier = tf.estimator.Estimator(
      model_fn=model_function,
      model_dir=flags_obj.model_dir,
      params={
          'data_format': data_format,
          'multi_gpu': multi_gpu
      })
```

OK, all training, validation and testing are disappeared, which are embedded in the estimator. The states are set in `tf.estimator.ModeKeys`: TRAIN, EVAL, PREDICT.

Let's see the `model_fn`, which includes many important functions.
* `model = create_model(params['data_format'])` is to create a model. Let's try `tf.layers` to take place of it. I just hard code it.
```
image = tf.reshape(image, (-1, 28, 28, 1))
model = tf.layers.conv2d(image, 32, [5,5], padding='same', activation=tf.nn.relu)
model = tf.layers.max_pooling2d(model, (2, 2), (2, 2), padding='same')
model = tf.layers.conv2d(model, 64, [5,5], padding='same', activation=tf.nn.relu)
model = tf.layers.max_pooling2d(model, (2, 2), (2, 2), padding='same')
model = tf.layers.flatten(model)
model = tf.layers.dense(model, units=1024)
model = tf.layers.dropout(model, 0.4)
logits = tf.layers.dense(model, units=10)
```
* `if mode == tf.estimator.ModeKeys.PREDICT`, `if mode == tf.estimator.ModeKeys.TRAIN`, and `if mode == tf.estimator.ModeKeys.EVAL` are to deduce the state.
* Prediction is very simple: defines 3 parameters, `mode, predictions and export_outputs`. `predictions` is a dict that its values are tensors you want to output and its corresponding keys are their output names. In this mode, you must assign predictions.
```
    if mode == tf.estimator.ModeKeys.PREDICT:
        # logits = model(image, training=False)
        predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                        'classify': tf.estimator.export.PredictOutput(predictions)
                })
```
* `EVAL` This is for evaluating the model, you must assign the loss.
```
    if mode == tf.estimator.ModeKeys.EVAL:
        # logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                        'accuracy':
                                tf.metrics.accuracy(
                                        labels=labels, predictions=tf.argmax(logits, axis=1)),
                })
```
* `TRAIN` is the most complicated. Firstly, define the optimizer, loss, and accuracy tensors. `tf.identity` is to log the tensors, and output in your terminal. `tf.summary` is to write in summary. In this mode, `loss` and `train_op` are required.
```
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
```
That's all about the framework to train a model. You can just modify the model part and dataset part to apply in your applications.

Reference:
* https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py
