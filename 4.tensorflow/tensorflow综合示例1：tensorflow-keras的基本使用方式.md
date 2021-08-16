```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
import os

print(tf.__version__)
```

    2.4.1


# 1、基本模型构建

本文只介绍了最基本，也是最常用的API，除此以外可以使用函数式API和子类API构建模型，详见《机器学习实战》第10章。

## 1.1 准备数据集
在这里我们使用了fashion_mnist数据集，里面是70000张28*28的图片，图片分为衣服、鞋子等10类。


```python
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)
```

    (55000, 28, 28) (55000,)
    (5000, 28, 28) (5000,)
    (10000, 28, 28) (10000,)


我们看一下图片是什么样子的：


```python
def show_single_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()
    
show_single_image(x_train[0])
```


    
![png](tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_files/tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_5_0.png)
    



```python
def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col 
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary",
                       interpolation = 'nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']
show_imgs(3, 5, x_train, y_train, class_names)
```


    
![png](tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_files/tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_6_0.png)
    


## 1.2 构建模型
构建模型主要分成2部分：

（1）指定模型的各层节点数及其连接

（2）编译模型，指定损失函数、优化方法、metrics等


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) #将输入的二维数组展开成一维向量
model.add(keras.layers.Dense(300,activation='sigmoid'))
model.add(keras.layers.Dense(100,activation='sigmoid'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])
```

如果sigmoid改成relu的话，精度就会降低非常的多。因为使用relu需要将数据归一化处理，而sigmoid则不需要。

编译模型主要指定损失函数、优化器和衡量指标，完整列表可见：
https://keras.io/api/losses/ https://keras.io/api/optimizers https://keras.io/api/metrics

我们使用sparse_categorical_crossentropy损失，因为我们具有稀疏标签，（即对于每个实例，只有一个目标类索引，在这种情况下为0到9），并且这些类是互斥的。相反，如果每个实例的每个类都有一个目标概率（例如独热向量，[0.，0.，0.，1.，0.，0.，0.，0.，0.，0]代表类3），则我们需要使用"categorical_crossentropy"损失。如果我们正在执行二进制分类（带有一个或多个二进制标签），则在输出层中使用"sigmoid"（即逻辑）激活函数，而不是"softmax"激活函数，并且使用"binary_crossentropy"损失。
如果要将稀疏标签（即类索引）转换为独热向量标签，使用keras.utils.to_categorical（）函数。反之则使用np.argmax（）函数和axis=1。

关于优化器，"sgd"表示我们使用简单的随机梯度下降来训练模型。换句话说，Keras将执行先前所述的反向传播算法（即反向模式自动微分加梯度下降）。我们将在第11章中讨论更有效的优化器（它们改进梯度下降部分，而不是自动微分）。


这样我们的模型就构建完成了，我们看一下模型长什么样子的：


```python
model.layers
```




    [<tensorflow.python.keras.layers.core.Flatten at 0x7f83fa791d00>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f83fa791d30>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f83fa80f5b0>,
     <tensorflow.python.keras.layers.core.Dense at 0x7f83fa68d220>]




```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            (None, 784)               0         
    _________________________________________________________________
    dense (Dense)                (None, 300)               235500    
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               30100     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________


## 1.3 训练模型
训练模型fit()返回的是一个History对象，用于保存中间计算过程的数据。

如果训练集非常不平衡，其中某些类的代表过多，而其它类的代表不足，那么在调用fit()方法时设置class_weight参数会很有用，这给代表性不足的类更大的权重，给代表过多的类更小的权重。Keras在计算损失时将使用这些权重。如果你需要每个实例的权重，设置sample_weight参数（如果class_weight和sample_weight都提供了，Keras会把它们相乘）。如果某些实例由专家标记，而另一些实例使用众包平台标记，则按实例权重可能会有用：你可能希望为前者赋予更多权重。你还可以通过将其作为validation_data元组的第三项添加到验证集中来提供样本权重（但不提供类权重）。

fit（）方法返回一个History对象，其中包含训练参数（history.params）、经历的轮次列表（history.epoch），最重要的是包含在训练集和验证集（如果有）上的每个轮次结束时测得的损失和额外指标的字典（history.history）。如果使用此字典创建pandasDataFrame并调用其plot（）方法，则会获得如图学习曲线.




```python
history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))
```

    Epoch 1/10
    1719/1719 [==============================] - 7s 4ms/step - loss: 1.5234 - accuracy: 0.5958 - val_loss: 0.7285 - val_accuracy: 0.7686
    Epoch 2/10
    1719/1719 [==============================] - 6s 4ms/step - loss: 0.7017 - accuracy: 0.7717 - val_loss: 0.5901 - val_accuracy: 0.8062
    Epoch 3/10
    1719/1719 [==============================] - 7s 4ms/step - loss: 0.5912 - accuracy: 0.8001 - val_loss: 0.5643 - val_accuracy: 0.8136
    Epoch 4/10
    1719/1719 [==============================] - 6s 4ms/step - loss: 0.5614 - accuracy: 0.8092 - val_loss: 0.5472 - val_accuracy: 0.8146
    Epoch 5/10
    1719/1719 [==============================] - 6s 3ms/step - loss: 0.5440 - accuracy: 0.8122 - val_loss: 0.5352 - val_accuracy: 0.8212
    Epoch 6/10
    1719/1719 [==============================] - 6s 3ms/step - loss: 0.5428 - accuracy: 0.8109 - val_loss: 0.5608 - val_accuracy: 0.8158
    Epoch 7/10
    1719/1719 [==============================] - 6s 3ms/step - loss: 0.5468 - accuracy: 0.8121 - val_loss: 0.5384 - val_accuracy: 0.8196
    Epoch 8/10
    1719/1719 [==============================] - 6s 4ms/step - loss: 0.5405 - accuracy: 0.8102 - val_loss: 0.5467 - val_accuracy: 0.8032
    Epoch 9/10
    1719/1719 [==============================] - 7s 4ms/step - loss: 0.5495 - accuracy: 0.8055 - val_loss: 0.5529 - val_accuracy: 0.8198
    Epoch 10/10
    1719/1719 [==============================] - 5s 3ms/step - loss: 0.5522 - accuracy: 0.8046 - val_loss: 0.5286 - val_accuracy: 0.8214



```python
type(history)
history.history
```




    {'loss': [1.1346834897994995,
      0.6621189713478088,
      0.5879183411598206,
      0.5602594017982483,
      0.5486269593238831,
      0.5448580980300903,
      0.5459325909614563,
      0.5451844334602356,
      0.5476701855659485,
      0.546495795249939],
     'accuracy': [0.6913090944290161,
      0.7830908894538879,
      0.8013636469841003,
      0.8078363537788391,
      0.810981810092926,
      0.8124363422393799,
      0.812145471572876,
      0.8098727464675903,
      0.8069090843200684,
      0.8073999881744385],
     'val_loss': [0.7285007834434509,
      0.5901457071304321,
      0.564271867275238,
      0.5471994876861572,
      0.5351706743240356,
      0.5608181357383728,
      0.5383569002151489,
      0.5467274188995361,
      0.5528538227081299,
      0.5285636782646179],
     'val_accuracy': [0.7685999870300293,
      0.8062000274658203,
      0.8136000037193298,
      0.8145999908447266,
      0.8212000131607056,
      0.8158000111579895,
      0.819599986076355,
      0.8032000064849854,
      0.8198000192642212,
      0.821399986743927]}



我们把训练过程中的loss及accuracy打印出来:


```python
def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
print_learning_curves(history)
```


    
![png](tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_files/tensorflow%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B1%EF%BC%9Atensorflow-keras%E7%9A%84%E5%9F%BA%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F_16_0.png)
    


## 1.4 evaluate模型


```python
model.evaluate(x_test,y_test)
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.5633 - accuracy: 0.7997





    [0.5633445382118225, 0.7997000217437744]



### 1.5 使用模型预测
我们使用上述训练得到的模型进行预测：


```python
x_new = x_test[:3]
y_proba = model.predict(X_new)
print(y_proba)
```

    [[6.3497009e-04 2.9951176e-03 3.5227172e-03 1.4390906e-03 7.3460588e-04
      1.5983881e-01 6.2727387e-04 1.8396391e-01 1.0167611e-02 6.3607597e-01]
     [1.4601831e-02 2.4284667e-03 5.7923472e-01 7.1747215e-03 1.8146098e-01
      2.0480098e-03 2.0280096e-01 3.3682014e-04 9.4090607e-03 5.0444162e-04]
     [5.4534234e-04 9.9256706e-01 1.0021541e-03 3.8844990e-03 1.1454911e-03
      1.0074565e-04 2.7266973e-05 5.8435014e-04 4.7284644e-05 9.5837881e-05]]


对于每个实例，模型估计从0类到9类每个类的概率。例如，对于第一个图像，模型估计是第9类（脚踝靴）的概率为96％，第5类的概率（凉鞋）为3％，第7类（运动鞋）的概率为1％，其他类别的概率可忽略不计。换句话说，它“相信”第一个图像是鞋类，最有可能是脚踝靴，但也可能是凉鞋或运动鞋。如果你只关心估计概率最高的类（即使该概率非常低），则可以使用predict_classes（）方法：



```python
y_pred = model.predict_classes(x_new)
print(y_pred)
```

    [9 2 1]


    /Users/ljhn1829/opt/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/keras/engine/sequential.py:450: UserWarning: `model.predict_classes()` is deprecated and will be removed after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
      warnings.warn('`model.predict_classes()` is deprecated and '


# 1.5 完整代码


```python
import numpy as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)

def show_single_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()
    
show_single_image(x_train[0])

def show_imgs(n_rows, n_cols, x_data, y_data, class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize = (n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col 
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(x_data[index], cmap="binary",
                       interpolation = 'nearest')
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']
show_imgs(3, 5, x_train, y_train, class_names)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) #将输入的二维数组展开成一维向量
model.add(keras.layers.Dense(300,activation='sigmoid'))
model.add(keras.layers.Dense(100,activation='sigmoid'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

model.layers
model.summary()

history = model.fit(x_train,y_train,epochs=10,validation_data=(x_valid,y_valid))

type(history)
history.history

def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
print_learning_curves(history)

model.evaluate(x_test,y_test)
```

# 2、归一化


```python
print(np.max(x_train), np.min(x_train))
```

现有数据在0~255之间，下面我们对数据做归一化。

我们使用均值是0，方差为1的标准正则归一化（也叫Z-score归一化），即： x = (x-u)/std

还有一种常见的归一化方式：Min-max归一化：x*=(x-min)/(max-min),取值在[0,1]之间。


```python
#scaler = sklearn.preprocessing.StandardScaler()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaler = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaler = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaler = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
```

上述代码的几个说明：

（1）fit_transform/transform接受的是一个二维浮点数向量作为参数，所以需要先转成2维向量再转回三维。

（2）fit_transform()和transform()：fit_transform()除了transform的归一化功能外，还把数据集的均值和方差记录下来，供下面的验证集、测试集使用。

然后我们再训练时使用上面经过归一化的数据：


```python
history = model.fit(x_train_scaler,y_train,epochs=10,validation_data=(x_valid_scaler,y_valid))
model.evaluate(x_test_scaler,y_test)
```

## 3、回调函数：TensorBoard EarlyStopping ModelCheckpoint

Callbacks: utilities called at certain points during model training.

也就是说模型训练过程中在某些点会触发一些功能或者操作。

最常用的就是TensorBoard EarlyStopping ModelCheckpoint这3类，以下会分别介绍。完整的callback请参考官方文档的
tf.keras.callback：https://www.tensorflow.org/api_docs/python/tf/keras/callbacks?hl=zh-cn
    


```python
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]

history = model.fit(x_train_scaler, y_train, epochs=10, 
                   validation_data=(x_valid_scaler,y_valid),
                   callbacks = callbacks)
```

启动tensorborad的方式很简单：

tensorboard --logdir=callbacks

然后打开http://localhost:6006/ 即可。

![%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-20%20%E4%B8%8B%E5%8D%883.30.39.png](attachment:%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-03-20%20%E4%B8%8B%E5%8D%883.30.39.png)

## 4、深度神经网络

DNN也没什么特别，就是层数比较多：


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(10,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
```

DNN在前几个ecpochs的训练时，loss降低的很慢，导致这个问题的原因主要有：

（1）参数众多、训练不足

（2）梯度消失 多层符合函数的链式法则导致的。

## 5、批归一化、dropout、激活函数

### 5.1 批归一化

归一化是对训练、测试数据做了归一化，就是模型的输入数据做了归一化。

而批归一化是对每一层激活函数的输出（也就是下一层的输入）都做了归一化。


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(10,activation='relu'))
    model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(10,activation='softmax'))
```

### 5.2 selu
上述relu+批归一化也可以通过直接使用selu激活函数代替：



```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(10,activation='selu'))
model.add(keras.layers.Dense(10,activation='softmax'))
```

### 5.3 dropout


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(10,activation='selu'))
model.add(keras.layers.AlphaDropout(rate=0.5))# 只在最后一层添加了dropout
# AlphaDropout: 1. 均值和方差不变 2. 归一化性质也不变
# model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10,activation='softmax'))
```

## 6、完整代码


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sklearn
import os

#导入数据
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

#训练数据归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaler = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaler = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

#构建及compile模型
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation='selu'))
model.add(keras.layers.AlphaDropout(rate=0.5))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

#定义callback
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]

#训练模型
history = model.fit(x_train_scaler, y_train, epochs=10, 
                   validation_data=(x_valid_scaler,y_valid),
                   callbacks = callbacks)

#检查模型效果
def print_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(10,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
print_learning_curves(history)

model.evaluate(x_test_scaler,y_test)
```


