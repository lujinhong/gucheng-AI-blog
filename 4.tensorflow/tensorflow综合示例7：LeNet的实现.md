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


```python
model.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_14 (Conv2D)           (None, 28, 28, 6)         156       
    _________________________________________________________________
    average_pooling2d_10 (Averag (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 10, 10, 16)        2416      
    _________________________________________________________________
    average_pooling2d_11 (Averag (None, 5, 5, 16)          0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 1, 1, 120)         48120     
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 120)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                850       
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




    
![png](tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B7%EF%BC%9ALeNet%E7%9A%84%E5%AE%9E%E7%8E%B0_files/tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B7%EF%BC%9ALeNet%E7%9A%84%E5%AE%9E%E7%8E%B0_21_1.png)
    



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



```python

```
