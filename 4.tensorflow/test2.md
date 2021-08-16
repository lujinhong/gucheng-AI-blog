```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers,metrics,optimizers
import pandas as pd
import numpy as np

def build_vgg16_model():
    weight_decay = 0.000
    num_classes = 10
    input_shape = (32, 32, 3)

    model = keras.models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                 input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))
    # model.add(layers.Activation('softmax'))
    
    return model
```


```python
model = build_vgg16_model()

(x_train, y_train),(x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# y预处理
def y_preprocess(y_train, y_test):
    # 删除y的一个纬度，从[b,1]变成[b,]，否则做onehot后纬度会变成[b,1,10]，而不是[b,10]
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    # onehot
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    return y_train, y_test

# x预处理
def x_preprocess(x_train, x_test):
    #数据标准化
    x_train = x_train/255.0
    x_test = x_test/255.0
    
    x_mean = x_train.mean()
    x_std = x_train.std()
    x_train = (x_train-x_mean)/x_std
    x_test = (x_test-x_mean)/x_std
    print(x_train.max(), x_train.min(), x_train.mean(), x_train.std())
    print(x_test.max(), x_test.min(), x_test.mean(), x_test.std())
    # 改成float32加快训练速度，避免使用float64
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    return x_train, x_test  


y_train, y_test = y_preprocess(y_train, y_test)
x_train, x_test = x_preprocess(x_train, x_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.CategoricalAccuracy()])

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#必须batch()，否则少了以为，shapeError。
ds_train = ds_train.shuffle(50000).batch(32).repeat(1)
ds_test = ds_test.shuffle(50000).batch(32).repeat(1)

model.fit(ds_train, validation_data=ds_test, epochs=20)

```

    (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
    2.0934103819959744 -1.8816433721538914 9.83429634023499e-15 1.0000000000000042
    2.0934103819959744 -1.8816433721538914 0.0128073056421839 0.9986109795368511
    (50000, 32, 32, 3) (50000, 10) (10000, 32, 32, 3) (10000, 10)
    Epoch 1/20
    1563/1563 [==============================] - 723s 459ms/step - loss: 0.4357 - categorical_accuracy: 0.1310 - val_loss: 0.3742 - val_categorical_accuracy: 0.1514
    Epoch 2/20
    1563/1563 [==============================] - 715s 457ms/step - loss: 0.0917 - categorical_accuracy: 0.2021 - val_loss: 0.1509 - val_categorical_accuracy: 0.1962
    Epoch 3/20
    1563/1563 [==============================] - 668s 427ms/step - loss: 0.0862 - categorical_accuracy: 0.2819 - val_loss: 0.0936 - val_categorical_accuracy: 0.2671
    Epoch 4/20
     240/1563 [===>..........................] - ETA: 7:25 - loss: 0.0818 - categorical_accuracy: 0.3503


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-6-65441a2c168d> in <module>
         48 ds_test = ds_test.shuffle(50000).batch(32).repeat(1)
         49 
    ---> 50 model.fit(ds_train, validation_data=ds_test, epochs=20)
    

    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1181                 _r=1):
       1182               callbacks.on_train_batch_begin(step)
    -> 1183               tmp_logs = self.train_function(iterator)
       1184               if data_handler.should_sync:
       1185                 context.async_wait()


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        887 
        888       with OptionalXlaContext(self._jit_compile):
    --> 889         result = self._call(*args, **kwds)
        890 
        891       new_tracing_count = self.experimental_get_tracing_count()


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        915       # In this case we have created variables on the first call, so we run the
        916       # defunned version which is guaranteed to never create variables.
    --> 917       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        918     elif self._stateful_fn is not None:
        919       # Release the lock early so that multiple threads can perform the call


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in __call__(self, *args, **kwargs)
       3021       (graph_function,
       3022        filtered_flat_args) = self._maybe_define_function(args, kwargs)
    -> 3023     return graph_function._call_flat(
       3024         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
       3025 


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1958         and executing_eagerly):
       1959       # No tape is watching; skip to running the function.
    -> 1960       return self._build_call_outputs(self._inference_function.call(
       1961           ctx, args, cancellation_manager=cancellation_manager))
       1962     forward_backward = self._select_forward_and_backward_functions(


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        589       with _InterpolateFunctionError(self):
        590         if cancellation_manager is None:
    --> 591           outputs = execute.execute(
        592               str(self.signature.name),
        593               num_outputs=self._num_outputs,


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         57   try:
         58     ctx.ensure_initialized()
    ---> 59     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         60                                         inputs, attrs, num_outputs)
         61   except core._NotOkStatusException as e:


    KeyboardInterrupt: 



```python
model = build_vgg16_model()

(x_train, y_train),(x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
model.fit(ds_train, validation_data=ds_test, epochs=20)

#print(x_train.dtype, y_train.dtype)
```

    (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-4-80a20a07c249> in <module>
         10 ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
         11 ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ---> 12 model.fit(ds_train, validation_data=ds_test, epochs=20)
         13 
         14 #print(x_train.dtype, y_train.dtype)


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1103     # Legacy graph support is contained in `training_v1.Model`.
       1104     version_utils.disallow_legacy_graph('Model', 'fit')
    -> 1105     self._assert_compile_was_called()
       1106     self._check_call_args('fit')
       1107     _disallow_inside_tf_function('fit')


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in _assert_compile_was_called(self)
       2691     # (i.e. whether the model is built and its inputs/outputs are set).
       2692     if not self._is_compiled:
    -> 2693       raise RuntimeError('You must compile your model before '
       2694                          'training/testing. '
       2695                          'Use `model.compile(optimizer, loss)`.')


    RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.



```python

x_mean = x_train.mean()
x_std = x_train.std()

def preprocess(x,y):
    y = tf.one_hot(y,depth=10)
    x = x/255
    x = (x-x_mean)/x_std

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x,y


ds_train = ds_train.shuffle(50000).batch(32).repeat(1)
ds_test = ds_test.shuffle(50000).batch(32).repeat(1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.CategoricalAccuracy()])

model.fit(ds_train, validation_data=ds_test, epochs=20)
```

    (50000, 32, 32, 3) (50000,) (10000, 32, 32, 3) (10000,)
    Epoch 1/20
      65/1563 [>.............................] - ETA: 11:10 - loss: 28.9155 - categorical_accuracy: 0.1077


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-3-bf356773524b> in <module>
         20 model.compile(optimizer=optimizer, loss='mse', metrics=[keras.metrics.CategoricalAccuracy()])
         21 
    ---> 22 model.fit(ds_train, validation_data=ds_test, epochs=20)
    

    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1181                 _r=1):
       1182               callbacks.on_train_batch_begin(step)
    -> 1183               tmp_logs = self.train_function(iterator)
       1184               if data_handler.should_sync:
       1185                 context.async_wait()


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
        887 
        888       with OptionalXlaContext(self._jit_compile):
    --> 889         result = self._call(*args, **kwds)
        890 
        891       new_tracing_count = self.experimental_get_tracing_count()


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py in _call(self, *args, **kwds)
        915       # In this case we have created variables on the first call, so we run the
        916       # defunned version which is guaranteed to never create variables.
    --> 917       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        918     elif self._stateful_fn is not None:
        919       # Release the lock early so that multiple threads can perform the call


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in __call__(self, *args, **kwargs)
       3021       (graph_function,
       3022        filtered_flat_args) = self._maybe_define_function(args, kwargs)
    -> 3023     return graph_function._call_flat(
       3024         filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access
       3025 


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1958         and executing_eagerly):
       1959       # No tape is watching; skip to running the function.
    -> 1960       return self._build_call_outputs(self._inference_function.call(
       1961           ctx, args, cancellation_manager=cancellation_manager))
       1962     forward_backward = self._select_forward_and_backward_functions(


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        589       with _InterpolateFunctionError(self):
        590         if cancellation_manager is None:
    --> 591           outputs = execute.execute(
        592               str(self.signature.name),
        593               num_outputs=self._num_outputs,


    ~/anaconda3/envs/ljh/lib/python3.8/site-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         57   try:
         58     ctx.ensure_initialized()
    ---> 59     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         60                                         inputs, attrs, num_outputs)
         61   except core._NotOkStatusException as e:


    KeyboardInterrupt: 



```python

```
