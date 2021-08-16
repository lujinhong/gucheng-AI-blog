在本文中，我们使用tensorflow2.x实现了lenet-5，用于mnist的识别。


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
```

# 数据预处理
我们先载入mnist数据


```python
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
```

我们把特征数据增加一个纬度，用于LeNet5的输入：


```python
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, y_train.shape)
```

    (60000, 28, 28) (60000,)
    (60000, 28, 28, 1) (60000,)


特征数据归一化：


```python
x_train = x_train/255.0
x_test = x_test/255.0
```

标签做onehot:


```python
y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))
```

# 构建模型
我们使用sequential构建LeNet-5模型：


```python
model = keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```

我们看一下模型的详细情况，包括每一层的输出大小，可训练参数数量，模型的总参数等。

总结
* 每个卷积层的大小是滤波器的大小（k*k) * 滤波器的数量 * 输入的通道数量（即上一层输出的通道数量）。
* 每个滤波器内的多个通道得到的最终结果，会按位相加得到最终一个输出矩阵，所以每一个卷积层输出filter个通道，这也是下一层的输入通道数量。


关于参数数量，我们做一个详细的解释：

* 第一个卷积层，每个滤波器的大小为5*5，总共有6个，加上偏置量，所以参数数量为5*5*6+6=156。如果输入图片为3通道图片，则参数个数为5*5*3*6+6=456。也就是说每个卷积核，他的厚度是与前一层的输入相同的。
* 第二个卷积层，每个滤波器大小也是5*5，总共16个，上一层的输出通道是6，所以总的参数数量是5*5*6*16+16=2416。
* 第三个卷积层，每个滤波器大小也是5*5，总共120个，上一层输出通道是16，所以总的参数数量是5*5*16*12+120=48120。


```python
model.summary()
```

    Model: "sequential_15"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_32 (Conv2D)           (None, 28, 28, 6)         156       
    _________________________________________________________________
    average_pooling2d_22 (Averag (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_33 (Conv2D)           (None, 10, 10, 16)        2416      
    _________________________________________________________________
    average_pooling2d_23 (Averag (None, 5, 5, 16)          0         
    _________________________________________________________________
    conv2d_34 (Conv2D)           (None, 1, 1, 120)         48120     
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 120)               0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 84)                10164     
    _________________________________________________________________
    dense_16 (Dense)             (None, 10)                850       
    =================================================================
    Total params: 61,706
    Trainable params: 61,706
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
```


```python
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

    (60000, 28, 28, 1) (60000, 10) (10000, 28, 28, 1) (10000, 10)


# 训练模型


```python
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
```

    Epoch 1/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0638 - acc: 0.9805 - val_loss: 0.0618 - val_acc: 0.9801
    Epoch 2/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0548 - acc: 0.9832 - val_loss: 0.0515 - val_acc: 0.9830
    Epoch 3/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0480 - acc: 0.9851 - val_loss: 0.0727 - val_acc: 0.9763
    Epoch 4/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0431 - acc: 0.9870 - val_loss: 0.0420 - val_acc: 0.9864
    Epoch 5/10
    1875/1875 [==============================] - 9s 5ms/step - loss: 0.0390 - acc: 0.9881 - val_loss: 0.0461 - val_acc: 0.9851
    Epoch 6/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0347 - acc: 0.9889 - val_loss: 0.0394 - val_acc: 0.9866
    Epoch 7/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0309 - acc: 0.9904 - val_loss: 0.0434 - val_acc: 0.9851
    Epoch 8/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0279 - acc: 0.9908 - val_loss: 0.0373 - val_acc: 0.9879
    Epoch 9/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0257 - acc: 0.9919 - val_loss: 0.0353 - val_acc: 0.9886
    Epoch 10/10
    1875/1875 [==============================] - 10s 5ms/step - loss: 0.0229 - acc: 0.9930 - val_loss: 0.0361 - val_acc: 0.9876


准确率可以打到98%以上。

# 保存模型


```python
model.save('mnist.h5')
```

# 加载模型&预测
我们使用上面的模型对手写数字进行预测


```python
import cv2
img = cv2.imread('3.png', 0)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7f602c0a07f0>




    
![png](tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B7%EF%BC%9ALeNet-5%E7%9A%84%E5%AE%9E%E7%8E%B0_files/tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B7%EF%BC%9ALeNet-5%E7%9A%84%E5%AE%9E%E7%8E%B0_21_1.png)
    



```python
img = cv2.resize(img, (28,28))
img = img.reshape(1, 28, 28, 1)
img = img/255.0
```


```python
my_model = tf.keras.models.load_model('mnist.h5')
predict = my_model.predict(img)
print(predict)
print(np.argmax(predict))
```

    [[4.9507696e-09 7.0097293e-08 1.7773251e-06 9.9997258e-01 3.6114369e-09
      1.9603556e-05 2.9246516e-10 1.3854858e-06 7.9077779e-07 3.7732302e-06]]
    3


# 手动训练模型
上述方式使用keras定义网络结构并compile()后，直接使用fit()来进行模型训练。

如果我们需要对训练过程有更细节的调整可以手动定义训练过程。

对于数据而言，我们可以使用传统的numpy/pandas的数据处理方式，但如果数据量太大时，可以使用tf.dataset的方式处理。下面我们先介绍numpy/pandas的处理方式，然后再介绍tf.dataset的处理方式。

## numpy/pandas方式


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses

(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
print('datasets:', x_train.shape, x_train.shape, x_train.min(), x_train.max())

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train/255.0
x_test = x_test/255.0

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))


# CNN不加L2
model = keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.build(input_shape=(None, 28, 28, 1))
model.summary()

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return x_train[idx], y_train[idx]

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
          end=end)

n_epochs = 5
batch_size = 32
n_steps = len(x_train) // batch_size
optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()

# 补充一个关于accuracy的说明：
# * 如果模型输出是一个数值，而标签也是一个数值，此时可以直接使用metrics=['accuracy']或者metrics= [keras.metrics.Accuracy()]
# * 如果模型输出是一个多分类的概率列表，标签是一个onehot后的列表，则此时使用category_accuracy或者keras.metrics.CategoricalAccuracy()
# * 如果模型输出是一个多分类的概率列表，标签是一个数值，则此时使用sparse_category_accuracy或者keras.metrics.SparseCategoricalAccuracy()

metrics = [keras.metrics.MeanAbsoluteError(), keras.metrics.CategoricalAccuracy(), keras.metrics.Precision()]

for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(x_train, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #如果model各层中有加入正则化：
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()
        

# 测试误差
print('Test metrics: ')
y_test_pred = model(x_test)
for metric in metrics:
    print(metric.name, metric(y_test, y_test_pred).numpy())
```

    datasets: (60000, 28, 28) (60000, 28, 28) 0 255
    Model: "sequential_54"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_143 (Conv2D)          (None, 28, 28, 6)         156       
    _________________________________________________________________
    average_pooling2d_94 (Averag (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_144 (Conv2D)          (None, 10, 10, 16)        2416      
    _________________________________________________________________
    average_pooling2d_95 (Averag (None, 5, 5, 16)          0         
    _________________________________________________________________
    conv2d_145 (Conv2D)          (None, 1, 1, 120)         48120     
    _________________________________________________________________
    flatten_47 (Flatten)         (None, 120)               0         
    _________________________________________________________________
    dense_114 (Dense)            (None, 84)                10164     
    _________________________________________________________________
    dense_115 (Dense)            (None, 10)                850       
    =================================================================
    Total params: 61,706
    Trainable params: 61,706
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/5
    60000/60000 - mean: 0.0155 - mean_absolute_error: 0.0320 - categorical_accuracy: 0.8792 - precision_36: 0.9509
    60000/60000 - mean: 0.0155 - mean_absolute_error: 0.0320 - categorical_accuracy: 0.8792 - precision_36: 0.9509
    Epoch 2/5
    60000/60000 - mean: 0.0044 - mean_absolute_error: 0.0083 - categorical_accuracy: 0.9716 - precision_36: 0.9751
    60000/60000 - mean: 0.0044 - mean_absolute_error: 0.0083 - categorical_accuracy: 0.9716 - precision_36: 0.9751
    Epoch 3/5
    60000/60000 - mean: 0.0037 - mean_absolute_error: 0.0063 - categorical_accuracy: 0.9771 - precision_36: 0.9792
    60000/60000 - mean: 0.0037 - mean_absolute_error: 0.0063 - categorical_accuracy: 0.9771 - precision_36: 0.9792
    Epoch 4/5
    60000/60000 - mean: 0.0031 - mean_absolute_error: 0.0051 - categorical_accuracy: 0.9804 - precision_36: 0.9820
    60000/60000 - mean: 0.0031 - mean_absolute_error: 0.0051 - categorical_accuracy: 0.9804 - precision_36: 0.9820
    Epoch 5/5
    60000/60000 - mean: 0.0029 - mean_absolute_error: 0.0047 - categorical_accuracy: 0.9815 - precision_36: 0.9834
    60000/60000 - mean: 0.0029 - mean_absolute_error: 0.0047 - categorical_accuracy: 0.9815 - precision_36: 0.9834
    mean_absolute_error 0.0048679295
    categorical_accuracy 0.9814
    precision_36 0.9836411



```python

```


```python

```
