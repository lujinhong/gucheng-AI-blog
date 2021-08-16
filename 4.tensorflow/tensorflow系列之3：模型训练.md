本文主要介绍了模型训练中的各方面内容，包括模型构建、loss、优化器、metrics、正则化、学习率、激活函数、epochs、参数初始化、超参数搜索等。


```python
import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np
import pandas
import matplotlib as mpl
print(tf.__version__)
```

    2.5.0



## 5、超参数搜索

神经网络的灵活性也是它们的主要缺点之一：有许多需要调整的超参数。你不仅可以使用任何可以想象的网络结构，而且即使在简单的MLP中，你也可以更改层数、每层神经元数、每层要使用的激活函数的类型、权重初始化逻辑，以及更多。

一种选择是简单地尝试超参数的许多组合，然后查看哪种对验证集最有效（或使用K折交叉验证）。例如我们可以像第2章中一样使用GridSearchCV或RandomizedSearchCV来探索超参数空间。为此我们需要将Keras模型包装在模仿常规ScikitLearn回归器的对象中。

下面我们详细介绍在tensorflow中进行超参数搜索的方式。


我们先构建一个基本模型用于之后的超参数搜索。本次我们使用housing数据集。



```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_valid = scaler.transform(X_valid)
x_test = scaler.transform(X_test)

```



超参数搜索，一种选择是简单地尝试超参数的许多组合，然后查看哪种对验证集最有效（或使用K折交叉验证）。例如我们可以使用GridSearchCV或RandomizedSearchCV来探索超参数空间。为此我们需要将Keras模型包装在模仿常规ScikitLearn回归器的对象中。第一步是创建一个函数，该函数将在给定一组超参数的情况下构建并编译Keras模型：


```python
def build_model(n_hidden=1, n_neurons=10, learning_rate=0.0001, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics='mse')
    return model


```

我们简单看一下只运行一次模型的情况。

指定任何超参数，因此它将使用我们在build_model（）中定义的默认超参数。现在，我们可以像常规ScikitLearn回归器一样使用该对象：我们可以使用其fit（）方法进行训练，然后使用其score（）方法进行评估，然后使用predict()方法预测。

传递给fit（）方法的任何其他参数都将传递给内部的Keras模型。还要注意，该分数将与MSE相反，因为ScikitLearn希望获得分数，而不是损失（即分数越高越好）。


```python

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
keras_reg.fit(x_train, y_train, epochs=5,
             validation_data=(x_valid, y_valid),
             callbacks=[keras.callbacks.EarlyStopping(patience=5)])
mse_test = keras_reg.score(x_test, y_test)
y_pred = keras_reg.predict(x_test[:5])
print(mse_test)
print(y_pred)
```

    Epoch 1/5
    363/363 [==============================] - 1s 1ms/step - loss: 5.0193 - mse: 5.0193 - val_loss: 3.9611 - val_mse: 3.9611
    Epoch 2/5
    363/363 [==============================] - 0s 766us/step - loss: 3.5299 - mse: 3.5299 - val_loss: 3.1452 - val_mse: 3.1452
    Epoch 3/5
    363/363 [==============================] - 0s 879us/step - loss: 2.6217 - mse: 2.6217 - val_loss: 2.8003 - val_mse: 2.8003
    Epoch 4/5
    363/363 [==============================] - 0s 798us/step - loss: 2.0493 - mse: 2.0493 - val_loss: 2.5640 - val_mse: 2.5640
    Epoch 5/5
    363/363 [==============================] - 0s 749us/step - loss: 1.6769 - mse: 1.6769 - val_loss: 2.3456 - val_mse: 2.3456
    162/162 [==============================] - 0s 507us/step - loss: 1.5176 - mse: 1.5176
    -1.5176007747650146
    [0.76028025 0.8603613  1.949896   1.5127833  1.2329737 ]


下面我们开始超参数搜索。

我们不想训练和评估这样的单个模型，尽管我们想训练数百个变体，并查看哪种变体在验证集上表现最佳。由于存在许多超参数，因此最好使用随机搜索而不是网格搜索。让我们尝试探索隐藏层的数量、神经元的数量和学习率：



```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(10, 30),
    "learning_rate": reciprocal(3e-4, 3e-3),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(x_train, x_train, epochs=5,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=3)])
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.2812 - mse: 1.2812 - val_loss: 271016.0938 - val_mse: 271016.0938
    Epoch 2/5
    242/242 [==============================] - 0s 818us/step - loss: 0.9913 - mse: 0.9913 - val_loss: 110464.4844 - val_mse: 110464.4844
    Epoch 3/5
    242/242 [==============================] - 0s 988us/step - loss: 0.9683 - mse: 0.9683 - val_loss: 72900.2344 - val_mse: 72900.2344
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9603 - mse: 0.9603 - val_loss: 59062.6641 - val_mse: 59062.6641
    Epoch 5/5
    242/242 [==============================] - 0s 815us/step - loss: 0.9552 - mse: 0.9552 - val_loss: 55489.0234 - val_mse: 55489.0234
    121/121 [==============================] - 0s 804us/step - loss: 0.8883 - mse: 0.8883
    [CV] END learning_rate=0.0022068903028043606, n_hidden=0, n_neurons=14; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.2686 - mse: 1.2686 - val_loss: 2363.8618 - val_mse: 2363.8618
    Epoch 2/5
    242/242 [==============================] - 0s 840us/step - loss: 0.8948 - mse: 0.8948 - val_loss: 39679.1523 - val_mse: 39679.1523
    Epoch 3/5
    242/242 [==============================] - 0s 823us/step - loss: 0.8604 - mse: 0.8604 - val_loss: 58850.3477 - val_mse: 58850.3477
    Epoch 4/5
    242/242 [==============================] - 0s 839us/step - loss: 0.8461 - mse: 0.8461 - val_loss: 67334.5469 - val_mse: 67334.5469
    121/121 [==============================] - 0s 808us/step - loss: 1.1879 - mse: 1.1879
    [CV] END learning_rate=0.0022068903028043606, n_hidden=0, n_neurons=14; total time=   1.3s
    Epoch 1/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.3345 - mse: 1.3345 - val_loss: 384362.5938 - val_mse: 384362.5938
    Epoch 2/5
    242/242 [==============================] - 0s 856us/step - loss: 1.0359 - mse: 1.0359 - val_loss: 167893.2344 - val_mse: 167893.2344
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0112 - mse: 1.0112 - val_loss: 101112.2188 - val_mse: 101112.2188
    Epoch 4/5
    242/242 [==============================] - 0s 816us/step - loss: 1.0046 - mse: 1.0046 - val_loss: 76459.3359 - val_mse: 76459.3359
    Epoch 5/5
    242/242 [==============================] - 0s 964us/step - loss: 1.0021 - mse: 1.0021 - val_loss: 64804.9102 - val_mse: 64804.9102
    121/121 [==============================] - 0s 783us/step - loss: 0.7146 - mse: 0.7146
    [CV] END learning_rate=0.0022068903028043606, n_hidden=0, n_neurons=14; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.1029 - mse: 1.1029 - val_loss: 91277.7109 - val_mse: 91277.7109
    Epoch 2/5
    242/242 [==============================] - 0s 879us/step - loss: 1.0333 - mse: 1.0333 - val_loss: 89677.3672 - val_mse: 89677.3672
    Epoch 3/5
    242/242 [==============================] - 0s 882us/step - loss: 0.9360 - mse: 0.9360 - val_loss: 118919.1094 - val_mse: 118919.1094
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9293 - mse: 0.9293 - val_loss: 114945.0000 - val_mse: 114945.0000
    Epoch 5/5
    242/242 [==============================] - 0s 954us/step - loss: 0.9272 - mse: 0.9272 - val_loss: 125172.4297 - val_mse: 125172.4297
    121/121 [==============================] - 0s 885us/step - loss: 0.8630 - mse: 0.8630
    [CV] END learning_rate=0.0023308034965341855, n_hidden=1, n_neurons=21; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.9087 - mse: 0.9087 - val_loss: 94299.0078 - val_mse: 94299.0078
    Epoch 2/5
    242/242 [==============================] - 0s 892us/step - loss: 0.8234 - mse: 0.8234 - val_loss: 124009.0078 - val_mse: 124009.0078
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8087 - mse: 0.8087 - val_loss: 132739.4531 - val_mse: 132739.4531
    Epoch 4/5
    242/242 [==============================] - 0s 888us/step - loss: 0.8020 - mse: 0.8020 - val_loss: 129496.0859 - val_mse: 129496.0859
    121/121 [==============================] - 0s 815us/step - loss: 1.1818 - mse: 1.1818
    [CV] END learning_rate=0.0023308034965341855, n_hidden=1, n_neurons=21; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.1323 - mse: 1.1323 - val_loss: 29715.3086 - val_mse: 29715.3086
    Epoch 2/5
    242/242 [==============================] - 0s 888us/step - loss: 1.0290 - mse: 1.0290 - val_loss: 42294.3750 - val_mse: 42294.3750
    Epoch 3/5
    242/242 [==============================] - 0s 883us/step - loss: 1.0133 - mse: 1.0133 - val_loss: 58250.1016 - val_mse: 58250.1016
    Epoch 4/5
    242/242 [==============================] - 0s 902us/step - loss: 1.0085 - mse: 1.0085 - val_loss: 58030.9531 - val_mse: 58030.9531
    121/121 [==============================] - 0s 872us/step - loss: 0.7211 - mse: 0.7211
    [CV] END learning_rate=0.0023308034965341855, n_hidden=1, n_neurons=21; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0081 - mse: 1.0081 - val_loss: 2368.5986 - val_mse: 2368.5986
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9619 - mse: 0.9619 - val_loss: 127.0774 - val_mse: 127.0774
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9513 - mse: 0.9513 - val_loss: 372.0601 - val_mse: 372.0601
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9464 - mse: 0.9464 - val_loss: 960.9767 - val_mse: 960.9767
    Epoch 5/5
    242/242 [==============================] - 0s 963us/step - loss: 0.9433 - mse: 0.9433 - val_loss: 1815.6578 - val_mse: 1815.6578
    121/121 [==============================] - 0s 847us/step - loss: 0.8800 - mse: 0.8800
    [CV] END learning_rate=0.0007101921987678483, n_hidden=3, n_neurons=23; total time=   2.1s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.9271 - mse: 0.9271 - val_loss: 158.2235 - val_mse: 158.2235
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8492 - mse: 0.8492 - val_loss: 1503.8811 - val_mse: 1503.8811
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8308 - mse: 0.8308 - val_loss: 3984.1479 - val_mse: 3984.1479
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8205 - mse: 0.8205 - val_loss: 6707.6294 - val_mse: 6707.6294
    121/121 [==============================] - 0s 1ms/step - loss: 1.2342 - mse: 1.2342
    [CV] END learning_rate=0.0007101921987678483, n_hidden=3, n_neurons=23; total time=   1.9s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.1465 - mse: 1.1465 - val_loss: 15719.9639 - val_mse: 15719.9639
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0752 - mse: 1.0752 - val_loss: 33268.1328 - val_mse: 33268.1328
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0488 - mse: 1.0488 - val_loss: 46255.3398 - val_mse: 46255.3398
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0359 - mse: 1.0359 - val_loss: 58163.0586 - val_mse: 58163.0586
    121/121 [==============================] - 0s 858us/step - loss: 0.7434 - mse: 0.7434
    [CV] END learning_rate=0.0007101921987678483, n_hidden=3, n_neurons=23; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0054 - mse: 1.0054 - val_loss: 14119.4756 - val_mse: 14119.4756
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9505 - mse: 0.9505 - val_loss: 18180.2520 - val_mse: 18180.2520
    Epoch 3/5
    242/242 [==============================] - 0s 1000us/step - loss: 0.9394 - mse: 0.9394 - val_loss: 25748.5371 - val_mse: 25748.5371
    Epoch 4/5
    242/242 [==============================] - 0s 966us/step - loss: 0.9343 - mse: 0.9343 - val_loss: 32599.8477 - val_mse: 32599.8477
    121/121 [==============================] - 0s 1ms/step - loss: 0.8709 - mse: 0.8709
    [CV] END learning_rate=0.0011354062592168128, n_hidden=3, n_neurons=24; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.8495 - mse: 0.8495 - val_loss: 9506.7891 - val_mse: 9506.7891
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8130 - mse: 0.8130 - val_loss: 18660.5508 - val_mse: 18660.5508
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8002 - mse: 0.8002 - val_loss: 22676.0098 - val_mse: 22676.0098
    Epoch 4/5
    242/242 [==============================] - 0s 986us/step - loss: 0.7940 - mse: 0.7940 - val_loss: 26547.0371 - val_mse: 26547.0371
    121/121 [==============================] - 0s 1ms/step - loss: 1.2028 - mse: 1.2028
    [CV] END learning_rate=0.0011354062592168128, n_hidden=3, n_neurons=24; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.1086 - mse: 1.1086 - val_loss: 56416.4922 - val_mse: 56416.4922
    Epoch 2/5
    242/242 [==============================] - 0s 951us/step - loss: 1.0285 - mse: 1.0285 - val_loss: 50999.6055 - val_mse: 50999.6055
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0215 - mse: 1.0215 - val_loss: 51732.1289 - val_mse: 51732.1289
    Epoch 4/5
    242/242 [==============================] - 0s 977us/step - loss: 1.0186 - mse: 1.0186 - val_loss: 53576.6641 - val_mse: 53576.6641
    Epoch 5/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0167 - mse: 1.0167 - val_loss: 54969.4609 - val_mse: 54969.4609
    121/121 [==============================] - 0s 1ms/step - loss: 0.7302 - mse: 0.7302
    [CV] END learning_rate=0.0011354062592168128, n_hidden=3, n_neurons=24; total time=   2.2s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.9596 - mse: 0.9596 - val_loss: 16166.7402 - val_mse: 16166.7402
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9592 - mse: 0.9592 - val_loss: 40418.1172 - val_mse: 40418.1172
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9357 - mse: 0.9357 - val_loss: 32146.0391 - val_mse: 32146.0391
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9307 - mse: 0.9307 - val_loss: 42682.4766 - val_mse: 42682.4766
    121/121 [==============================] - 0s 1ms/step - loss: 0.8644 - mse: 0.8644
    [CV] END learning_rate=0.0028774200613215215, n_hidden=2, n_neurons=10; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0027 - mse: 1.0027 - val_loss: 701.9717 - val_mse: 701.9717
    Epoch 2/5
    242/242 [==============================] - 0s 884us/step - loss: 0.8337 - mse: 0.8337 - val_loss: 26590.4492 - val_mse: 26590.4492
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8141 - mse: 0.8141 - val_loss: 47854.2578 - val_mse: 47854.2578
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8082 - mse: 0.8082 - val_loss: 57246.3008 - val_mse: 57246.3008
    121/121 [==============================] - 0s 827us/step - loss: 1.1804 - mse: 1.1804
    [CV] END learning_rate=0.0028774200613215215, n_hidden=2, n_neurons=10; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 3ms/step - loss: 1.2153 - mse: 1.2153 - val_loss: 39862.7031 - val_mse: 39862.7031
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0863 - mse: 1.0863 - val_loss: 47315.9258 - val_mse: 47315.9258
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0627 - mse: 1.0627 - val_loss: 25198.9922 - val_mse: 25198.9922
    Epoch 4/5
    242/242 [==============================] - 0s 922us/step - loss: 1.0490 - mse: 1.0490 - val_loss: 23330.9707 - val_mse: 23330.9707
    Epoch 5/5
    242/242 [==============================] - 0s 883us/step - loss: 1.0406 - mse: 1.0406 - val_loss: 25656.3789 - val_mse: 25656.3789
    121/121 [==============================] - 0s 831us/step - loss: 0.7269 - mse: 0.7269
    [CV] END learning_rate=0.0028774200613215215, n_hidden=2, n_neurons=10; total time=   2.2s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.4290 - mse: 1.4290 - val_loss: 15890.3203 - val_mse: 15890.3203
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9684 - mse: 0.9684 - val_loss: 37877.1562 - val_mse: 37877.1562
    Epoch 3/5
    242/242 [==============================] - 0s 998us/step - loss: 0.9484 - mse: 0.9484 - val_loss: 47793.1992 - val_mse: 47793.1992
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9435 - mse: 0.9435 - val_loss: 53184.6406 - val_mse: 53184.6406
    121/121 [==============================] - 0s 1ms/step - loss: 0.8871 - mse: 0.8871
    [CV] END learning_rate=0.002056216541631671, n_hidden=0, n_neurons=13; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.2321 - mse: 1.2321 - val_loss: 31010.3359 - val_mse: 31010.3359
    Epoch 2/5
    242/242 [==============================] - 0s 972us/step - loss: 0.9201 - mse: 0.9201 - val_loss: 584.1401 - val_mse: 584.1401
    Epoch 3/5
    242/242 [==============================] - 0s 992us/step - loss: 0.8436 - mse: 0.8436 - val_loss: 19999.9570 - val_mse: 19999.9570
    Epoch 4/5
    242/242 [==============================] - 0s 817us/step - loss: 0.8167 - mse: 0.8167 - val_loss: 42793.5469 - val_mse: 42793.5469
    Epoch 5/5
    242/242 [==============================] - 0s 786us/step - loss: 0.8042 - mse: 0.8042 - val_loss: 56733.9883 - val_mse: 56733.9883
    121/121 [==============================] - 0s 776us/step - loss: 1.3703 - mse: 1.3703
    [CV] END learning_rate=0.002056216541631671, n_hidden=0, n_neurons=13; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.8375 - mse: 1.8375 - val_loss: 2623.2158 - val_mse: 2623.2158
    Epoch 2/5
    242/242 [==============================] - 0s 896us/step - loss: 1.0282 - mse: 1.0282 - val_loss: 28197.4180 - val_mse: 28197.4180
    Epoch 3/5
    242/242 [==============================] - 0s 885us/step - loss: 1.0017 - mse: 1.0017 - val_loss: 41335.2266 - val_mse: 41335.2266
    Epoch 4/5
    242/242 [==============================] - 0s 864us/step - loss: 0.9990 - mse: 0.9990 - val_loss: 47283.7617 - val_mse: 47283.7617
    121/121 [==============================] - 0s 786us/step - loss: 0.7114 - mse: 0.7114
    [CV] END learning_rate=0.002056216541631671, n_hidden=0, n_neurons=13; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.3317 - mse: 1.3317 - val_loss: 98436.5938 - val_mse: 98436.5938
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1113 - mse: 1.1113 - val_loss: 73269.6484 - val_mse: 73269.6484
    Epoch 3/5
    242/242 [==============================] - 0s 982us/step - loss: 1.0130 - mse: 1.0130 - val_loss: 60402.1953 - val_mse: 60402.1953
    Epoch 4/5
    242/242 [==============================] - 0s 824us/step - loss: 0.9664 - mse: 0.9664 - val_loss: 53842.9062 - val_mse: 53842.9062
    Epoch 5/5
    242/242 [==============================] - 0s 802us/step - loss: 0.9432 - mse: 0.9432 - val_loss: 50272.0234 - val_mse: 50272.0234
    121/121 [==============================] - 0s 1ms/step - loss: 0.8741 - mse: 0.8741
    [CV] END learning_rate=0.0006162164274253107, n_hidden=0, n_neurons=12; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.3801 - mse: 1.3801 - val_loss: 293260.4688 - val_mse: 293260.4688
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1640 - mse: 1.1640 - val_loss: 151182.0625 - val_mse: 151182.0625
    Epoch 3/5
    242/242 [==============================] - 0s 844us/step - loss: 1.0378 - mse: 1.0378 - val_loss: 70168.0156 - val_mse: 70168.0156
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9629 - mse: 0.9629 - val_loss: 27008.2852 - val_mse: 27008.2852
    Epoch 5/5
    242/242 [==============================] - 0s 845us/step - loss: 0.9169 - mse: 0.9169 - val_loss: 7222.2085 - val_mse: 7222.2085
    121/121 [==============================] - 0s 793us/step - loss: 1.3878 - mse: 1.3878
    [CV] END learning_rate=0.0006162164274253107, n_hidden=0, n_neurons=12; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 3.0256 - mse: 3.0256 - val_loss: 999730.5625 - val_mse: 999730.5625
    Epoch 2/5
    242/242 [==============================] - 0s 908us/step - loss: 1.8599 - mse: 1.8599 - val_loss: 428278.5312 - val_mse: 428278.5312
    Epoch 3/5
    242/242 [==============================] - 0s 842us/step - loss: 1.3814 - mse: 1.3814 - val_loss: 168287.2188 - val_mse: 168287.2188
    Epoch 4/5
    242/242 [==============================] - 0s 832us/step - loss: 1.1830 - mse: 1.1830 - val_loss: 57176.5195 - val_mse: 57176.5195
    Epoch 5/5
    242/242 [==============================] - 0s 826us/step - loss: 1.0936 - mse: 1.0936 - val_loss: 13803.8721 - val_mse: 13803.8721
    121/121 [==============================] - 0s 778us/step - loss: 0.7673 - mse: 0.7673
    [CV] END learning_rate=0.0006162164274253107, n_hidden=0, n_neurons=12; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.9654 - mse: 0.9654 - val_loss: 15823.8057 - val_mse: 15823.8057
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9557 - mse: 0.9557 - val_loss: 17473.8359 - val_mse: 17473.8359
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9490 - mse: 0.9490 - val_loss: 19702.0723 - val_mse: 19702.0723
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9439 - mse: 0.9439 - val_loss: 22999.4316 - val_mse: 22999.4316
    121/121 [==============================] - 0s 1ms/step - loss: 0.8738 - mse: 0.8738
    [CV] END learning_rate=0.0005041072464493266, n_hidden=3, n_neurons=28; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.8652 - mse: 0.8652 - val_loss: 7807.2402 - val_mse: 7807.2402
    Epoch 2/5
    242/242 [==============================] - 0s 993us/step - loss: 0.8311 - mse: 0.8311 - val_loss: 6267.4023 - val_mse: 6267.4023
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8194 - mse: 0.8194 - val_loss: 6582.5571 - val_mse: 6582.5571
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8130 - mse: 0.8130 - val_loss: 7624.7715 - val_mse: 7624.7715
    Epoch 5/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8089 - mse: 0.8089 - val_loss: 9024.7676 - val_mse: 9024.7676
    121/121 [==============================] - 0s 869us/step - loss: 1.1590 - mse: 1.1590
    [CV] END learning_rate=0.0005041072464493266, n_hidden=3, n_neurons=28; total time=   2.0s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0978 - mse: 1.0978 - val_loss: 3858.8838 - val_mse: 3858.8838
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0697 - mse: 1.0697 - val_loss: 5162.4463 - val_mse: 5162.4463
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0519 - mse: 1.0519 - val_loss: 5992.9204 - val_mse: 5992.9204
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0399 - mse: 1.0399 - val_loss: 6746.4946 - val_mse: 6746.4946
    121/121 [==============================] - 0s 833us/step - loss: 0.7439 - mse: 0.7439
    [CV] END learning_rate=0.0005041072464493266, n_hidden=3, n_neurons=28; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.2717 - mse: 1.2717 - val_loss: 23373.5664 - val_mse: 23373.5664
    Epoch 2/5
    242/242 [==============================] - 0s 987us/step - loss: 0.9831 - mse: 0.9831 - val_loss: 294.8194 - val_mse: 294.8194
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9454 - mse: 0.9454 - val_loss: 5932.8896 - val_mse: 5932.8896
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9330 - mse: 0.9330 - val_loss: 17998.1992 - val_mse: 17998.1992
    Epoch 5/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9277 - mse: 0.9277 - val_loss: 28753.9922 - val_mse: 28753.9922
    121/121 [==============================] - 0s 812us/step - loss: 0.8646 - mse: 0.8646
    [CV] END learning_rate=0.0013374759073922044, n_hidden=1, n_neurons=19; total time=   1.9s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.9918 - mse: 0.9918 - val_loss: 504126.4062 - val_mse: 504126.4062
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8420 - mse: 0.8420 - val_loss: 333093.1875 - val_mse: 333093.1875
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8218 - mse: 0.8218 - val_loss: 257948.6562 - val_mse: 257948.6562
    Epoch 4/5
    242/242 [==============================] - 0s 843us/step - loss: 0.8128 - mse: 0.8128 - val_loss: 213543.3125 - val_mse: 213543.3125
    Epoch 5/5
    242/242 [==============================] - 0s 857us/step - loss: 0.8075 - mse: 0.8075 - val_loss: 186322.9844 - val_mse: 186322.9844
    121/121 [==============================] - 0s 1ms/step - loss: 1.2765 - mse: 1.2765
    [CV] END learning_rate=0.0013374759073922044, n_hidden=1, n_neurons=19; total time=   2.1s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.4262 - mse: 1.4262 - val_loss: 9396.6846 - val_mse: 9396.6846
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1700 - mse: 1.1700 - val_loss: 17816.9316 - val_mse: 17816.9316
    Epoch 3/5
    242/242 [==============================] - 0s 863us/step - loss: 1.0911 - mse: 1.0911 - val_loss: 28112.8926 - val_mse: 28112.8926
    Epoch 4/5
    242/242 [==============================] - 0s 860us/step - loss: 1.0569 - mse: 1.0569 - val_loss: 38816.5234 - val_mse: 38816.5234
    121/121 [==============================] - 0s 1ms/step - loss: 0.7612 - mse: 0.7612
    [CV] END learning_rate=0.0013374759073922044, n_hidden=1, n_neurons=19; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.0433 - mse: 1.0433 - val_loss: 7120.2026 - val_mse: 7120.2026
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9736 - mse: 0.9736 - val_loss: 24620.0840 - val_mse: 24620.0840
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9568 - mse: 0.9568 - val_loss: 43995.9805 - val_mse: 43995.9805
    Epoch 4/5
    242/242 [==============================] - 0s 929us/step - loss: 0.9476 - mse: 0.9476 - val_loss: 63117.6211 - val_mse: 63117.6211
    121/121 [==============================] - 0s 1ms/step - loss: 0.8839 - mse: 0.8839
    [CV] END learning_rate=0.001557352779269722, n_hidden=3, n_neurons=12; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.8183 - mse: 0.8183 - val_loss: 2194.5620 - val_mse: 2194.5620
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8085 - mse: 0.8085 - val_loss: 4912.2158 - val_mse: 4912.2158
    Epoch 3/5
    242/242 [==============================] - 0s 949us/step - loss: 0.8048 - mse: 0.8048 - val_loss: 7725.8325 - val_mse: 7725.8325
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8026 - mse: 0.8026 - val_loss: 9980.3330 - val_mse: 9980.3330
    121/121 [==============================] - 0s 831us/step - loss: 1.1655 - mse: 1.1655
    [CV] END learning_rate=0.001557352779269722, n_hidden=3, n_neurons=12; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.0888 - mse: 1.0888 - val_loss: 144572.7188 - val_mse: 144572.7188
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0278 - mse: 1.0278 - val_loss: 116815.1797 - val_mse: 116815.1797
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0176 - mse: 1.0176 - val_loss: 106761.1797 - val_mse: 106761.1797
    Epoch 4/5
    242/242 [==============================] - 0s 960us/step - loss: 1.0139 - mse: 1.0139 - val_loss: 99824.1797 - val_mse: 99824.1797
    Epoch 5/5
    242/242 [==============================] - 0s 949us/step - loss: 1.0122 - mse: 1.0122 - val_loss: 95354.5938 - val_mse: 95354.5938
    121/121 [==============================] - 0s 1ms/step - loss: 0.7263 - mse: 0.7263
    [CV] END learning_rate=0.001557352779269722, n_hidden=3, n_neurons=12; total time=   1.9s
    Epoch 1/5
    363/363 [==============================] - 1s 1ms/step - loss: 0.9547 - mse: 0.9547 - val_loss: 29415.5410 - val_mse: 29415.5410
    Epoch 2/5
    363/363 [==============================] - 0s 836us/step - loss: 0.9210 - mse: 0.9210 - val_loss: 49913.9961 - val_mse: 49913.9961
    Epoch 3/5
    363/363 [==============================] - 0s 963us/step - loss: 0.9124 - mse: 0.9124 - val_loss: 52824.2383 - val_mse: 52824.2383
    Epoch 4/5
    363/363 [==============================] - 0s 770us/step - loss: 0.9077 - mse: 0.9077 - val_loss: 53914.6758 - val_mse: 53914.6758





    RandomizedSearchCV(cv=3,
                       estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x7fc2e648ab80>,
                       param_distributions={'learning_rate': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc2d86e72e0>,
                                            'n_hidden': [0, 1, 2, 3],
                                            'n_neurons': array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29])},
                       verbose=2)



搜索可能持续数小时，具体时间取决于硬件、数据集的大小、模型的复杂性以及n_iter和cv的值。当结束时，你可以访问找到的最佳参数、最佳分数和经过训练的Keras模型：


```python
print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
model = rnd_search_cv.best_estimator_.model
```

    {'learning_rate': 0.0023308034965341855, 'n_hidden': 1, 'n_neurons': 21}
    -0.9219454526901245


完整代码如下：


```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_valid = scaler.transform(X_valid)
x_test = scaler.transform(X_test) 

def build_model(n_hidden=1, n_neurons=10, learning_rate=0.0001, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics='mse')
    return model

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(10, 30),
    "learning_rate": reciprocal(3e-4, 3e-3),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(x_train, x_train, epochs=5,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=3)])
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.1251 - mse: 1.1251 - val_loss: 47374.3945 - val_mse: 47374.3945
    Epoch 2/5
    242/242 [==============================] - 0s 863us/step - loss: 1.0586 - mse: 1.0586 - val_loss: 81092.0469 - val_mse: 81092.0469
    Epoch 3/5
    242/242 [==============================] - 0s 876us/step - loss: 1.0360 - mse: 1.0360 - val_loss: 45819.7070 - val_mse: 45819.7070
    Epoch 4/5
    242/242 [==============================] - 0s 868us/step - loss: 1.0653 - mse: 1.0653 - val_loss: 93077.3594 - val_mse: 93077.3594
    Epoch 5/5
    242/242 [==============================] - 0s 894us/step - loss: 1.0592 - mse: 1.0592 - val_loss: 60477.1094 - val_mse: 60477.1094
    121/121 [==============================] - 0s 1ms/step - loss: 0.8770 - mse: 0.8770
    [CV] END learning_rate=0.0017487067052027632, n_hidden=1, n_neurons=18; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.9969 - mse: 0.9969 - val_loss: 96573.0547 - val_mse: 96573.0547
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8442 - mse: 0.8442 - val_loss: 41781.1484 - val_mse: 41781.1484
    Epoch 3/5
    242/242 [==============================] - 0s 916us/step - loss: 0.8226 - mse: 0.8226 - val_loss: 18573.9082 - val_mse: 18573.9082
    Epoch 4/5
    242/242 [==============================] - 0s 862us/step - loss: 0.8137 - mse: 0.8137 - val_loss: 7696.1504 - val_mse: 7696.1504
    Epoch 5/5
    242/242 [==============================] - 0s 872us/step - loss: 0.8084 - mse: 0.8084 - val_loss: 3060.2534 - val_mse: 3060.2534
    121/121 [==============================] - 0s 811us/step - loss: 1.1590 - mse: 1.1590
    [CV] END learning_rate=0.0017487067052027632, n_hidden=1, n_neurons=18; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.1137 - mse: 1.1137 - val_loss: 84434.2500 - val_mse: 84434.2500
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0437 - mse: 1.0437 - val_loss: 87475.7656 - val_mse: 87475.7656
    Epoch 3/5
    242/242 [==============================] - 0s 888us/step - loss: 1.0266 - mse: 1.0266 - val_loss: 88909.7422 - val_mse: 88909.7422
    Epoch 4/5
    242/242 [==============================] - 0s 877us/step - loss: 1.0171 - mse: 1.0171 - val_loss: 90662.4062 - val_mse: 90662.4062
    121/121 [==============================] - 0s 812us/step - loss: 0.7276 - mse: 0.7276
    [CV] END learning_rate=0.0017487067052027632, n_hidden=1, n_neurons=18; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.2036 - mse: 1.2036 - val_loss: 100045.9375 - val_mse: 100045.9375
    Epoch 2/5
    242/242 [==============================] - 0s 938us/step - loss: 1.1311 - mse: 1.1311 - val_loss: 97947.2812 - val_mse: 97947.2812
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0827 - mse: 1.0827 - val_loss: 98367.7031 - val_mse: 98367.7031
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0483 - mse: 1.0483 - val_loss: 94935.3281 - val_mse: 94935.3281
    Epoch 5/5
    242/242 [==============================] - 0s 903us/step - loss: 1.0230 - mse: 1.0230 - val_loss: 94672.2578 - val_mse: 94672.2578
    121/121 [==============================] - 0s 822us/step - loss: 0.9409 - mse: 0.9409
    [CV] END learning_rate=0.0003300029011245294, n_hidden=2, n_neurons=13; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0004 - mse: 1.0004 - val_loss: 37700.2227 - val_mse: 37700.2227
    Epoch 2/5
    242/242 [==============================] - 0s 908us/step - loss: 0.9283 - mse: 0.9283 - val_loss: 23583.4805 - val_mse: 23583.4805
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8947 - mse: 0.8947 - val_loss: 15595.6738 - val_mse: 15595.6738
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8754 - mse: 0.8754 - val_loss: 10595.4346 - val_mse: 10595.4346
    Epoch 5/5
    242/242 [==============================] - 0s 941us/step - loss: 0.8626 - mse: 0.8626 - val_loss: 7325.4795 - val_mse: 7325.4795
    121/121 [==============================] - 0s 1ms/step - loss: 1.3331 - mse: 1.3331
    [CV] END learning_rate=0.0003300029011245294, n_hidden=2, n_neurons=13; total time=   2.3s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.3150 - mse: 1.3150 - val_loss: 39398.5234 - val_mse: 39398.5234
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1184 - mse: 1.1184 - val_loss: 52931.5430 - val_mse: 52931.5430
    Epoch 3/5
    242/242 [==============================] - 0s 978us/step - loss: 1.0746 - mse: 1.0746 - val_loss: 58011.5898 - val_mse: 58011.5898
    Epoch 4/5
    242/242 [==============================] - 0s 971us/step - loss: 1.0539 - mse: 1.0539 - val_loss: 60266.1719 - val_mse: 60266.1719
    121/121 [==============================] - 0s 1ms/step - loss: 0.7500 - mse: 0.7500
    [CV] END learning_rate=0.0003300029011245294, n_hidden=2, n_neurons=13; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 3.2033 - mse: 3.2033 - val_loss: 533178.2500 - val_mse: 533178.2500
    Epoch 2/5
    242/242 [==============================] - 0s 836us/step - loss: 1.6398 - mse: 1.6398 - val_loss: 166993.0625 - val_mse: 166993.0625
    Epoch 3/5
    242/242 [==============================] - 0s 829us/step - loss: 1.1886 - mse: 1.1886 - val_loss: 42801.5859 - val_mse: 42801.5859
    Epoch 4/5
    242/242 [==============================] - 0s 837us/step - loss: 1.0360 - mse: 1.0360 - val_loss: 5650.2422 - val_mse: 5650.2422
    Epoch 5/5
    242/242 [==============================] - 0s 835us/step - loss: 0.9768 - mse: 0.9768 - val_loss: 207.2831 - val_mse: 207.2831
    121/121 [==============================] - 0s 800us/step - loss: 0.9104 - mse: 0.9104
    [CV] END learning_rate=0.0007929328213730603, n_hidden=0, n_neurons=29; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 2.4323 - mse: 2.4323 - val_loss: 664159.1875 - val_mse: 664159.1875
    Epoch 2/5
    242/242 [==============================] - 0s 851us/step - loss: 1.4502 - mse: 1.4502 - val_loss: 205217.5156 - val_mse: 205217.5156
    Epoch 3/5
    242/242 [==============================] - 0s 827us/step - loss: 1.0666 - mse: 1.0666 - val_loss: 46449.8125 - val_mse: 46449.8125
    Epoch 4/5
    242/242 [==============================] - 0s 840us/step - loss: 0.9095 - mse: 0.9095 - val_loss: 3605.0403 - val_mse: 3605.0403
    Epoch 5/5
    242/242 [==============================] - 0s 836us/step - loss: 0.8427 - mse: 0.8427 - val_loss: 1939.2106 - val_mse: 1939.2106
    121/121 [==============================] - 0s 787us/step - loss: 1.2751 - mse: 1.2751
    [CV] END learning_rate=0.0007929328213730603, n_hidden=0, n_neurons=29; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 2.2687 - mse: 2.2687 - val_loss: 88149.9141 - val_mse: 88149.9141
    Epoch 2/5
    242/242 [==============================] - 0s 908us/step - loss: 1.3341 - mse: 1.3341 - val_loss: 48710.5352 - val_mse: 48710.5352
    Epoch 3/5
    242/242 [==============================] - 0s 834us/step - loss: 1.1004 - mse: 1.1004 - val_loss: 36882.2617 - val_mse: 36882.2617
    Epoch 4/5
    242/242 [==============================] - 0s 1000us/step - loss: 1.0384 - mse: 1.0384 - val_loss: 34149.3242 - val_mse: 34149.3242
    Epoch 5/5
    242/242 [==============================] - 0s 924us/step - loss: 1.0199 - mse: 1.0199 - val_loss: 34745.8242 - val_mse: 34745.8242
    121/121 [==============================] - 0s 791us/step - loss: 0.7261 - mse: 0.7261
    [CV] END learning_rate=0.0007929328213730603, n_hidden=0, n_neurons=29; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.0575 - mse: 1.0575 - val_loss: 452.1569 - val_mse: 452.1569
    Epoch 2/5
    242/242 [==============================] - 0s 990us/step - loss: 1.0276 - mse: 1.0276 - val_loss: 82.5773 - val_mse: 82.5773
    Epoch 3/5
    242/242 [==============================] - 0s 938us/step - loss: 1.0122 - mse: 1.0122 - val_loss: 112.7742 - val_mse: 112.7742
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9972 - mse: 0.9972 - val_loss: 791.2552 - val_mse: 791.2552
    Epoch 5/5
    242/242 [==============================] - 0s 953us/step - loss: 0.9847 - mse: 0.9847 - val_loss: 2131.1604 - val_mse: 2131.1604
    121/121 [==============================] - 0s 828us/step - loss: 0.9225 - mse: 0.9225
    [CV] END learning_rate=0.0014381566219127105, n_hidden=3, n_neurons=16; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.8600 - mse: 0.8600 - val_loss: 12676.4814 - val_mse: 12676.4814
    Epoch 2/5
    242/242 [==============================] - 0s 957us/step - loss: 0.8294 - mse: 0.8294 - val_loss: 17479.8203 - val_mse: 17479.8203
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8146 - mse: 0.8146 - val_loss: 23118.3105 - val_mse: 23118.3105
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8073 - mse: 0.8073 - val_loss: 29235.3867 - val_mse: 29235.3867
    121/121 [==============================] - 0s 819us/step - loss: 1.2152 - mse: 1.2152
    [CV] END learning_rate=0.0014381566219127105, n_hidden=3, n_neurons=16; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0548 - mse: 1.0548 - val_loss: 8203.1064 - val_mse: 8203.1064
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0385 - mse: 1.0385 - val_loss: 10502.4805 - val_mse: 10502.4805
    Epoch 3/5
    242/242 [==============================] - 0s 961us/step - loss: 1.0316 - mse: 1.0316 - val_loss: 13889.0244 - val_mse: 13889.0244
    Epoch 4/5
    242/242 [==============================] - 0s 935us/step - loss: 1.0266 - mse: 1.0266 - val_loss: 16957.6973 - val_mse: 16957.6973
    121/121 [==============================] - 0s 817us/step - loss: 0.7375 - mse: 0.7375
    [CV] END learning_rate=0.0014381566219127105, n_hidden=3, n_neurons=16; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.2113 - mse: 1.2113 - val_loss: 4072.1184 - val_mse: 4072.1184
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0924 - mse: 1.0924 - val_loss: 7.4905 - val_mse: 7.4905
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0333 - mse: 1.0333 - val_loss: 2940.3550 - val_mse: 2940.3550
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0004 - mse: 1.0004 - val_loss: 9002.7754 - val_mse: 9002.7754
    Epoch 5/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9799 - mse: 0.9799 - val_loss: 15444.6045 - val_mse: 15444.6045
    121/121 [==============================] - 0s 876us/step - loss: 0.9061 - mse: 0.9061
    [CV] END learning_rate=0.00044519425836113695, n_hidden=3, n_neurons=27; total time=   2.1s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.8973 - mse: 0.8973 - val_loss: 22322.5645 - val_mse: 22322.5645
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8682 - mse: 0.8682 - val_loss: 15331.2178 - val_mse: 15331.2178
    Epoch 3/5
    242/242 [==============================] - 0s 926us/step - loss: 0.8492 - mse: 0.8492 - val_loss: 10439.3564 - val_mse: 10439.3564
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8358 - mse: 0.8358 - val_loss: 6745.4243 - val_mse: 6745.4243
    Epoch 5/5
    242/242 [==============================] - 0s 981us/step - loss: 0.8261 - mse: 0.8261 - val_loss: 4157.8188 - val_mse: 4157.8188
    121/121 [==============================] - 0s 851us/step - loss: 1.2005 - mse: 1.2005
    [CV] END learning_rate=0.00044519425836113695, n_hidden=3, n_neurons=27; total time=   2.0s
    Epoch 1/5
    242/242 [==============================] - 1s 3ms/step - loss: 1.2313 - mse: 1.2313 - val_loss: 4438.2280 - val_mse: 4438.2280
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1003 - mse: 1.1003 - val_loss: 8602.9102 - val_mse: 8602.9102
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0606 - mse: 1.0606 - val_loss: 11477.5586 - val_mse: 11477.5586
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0428 - mse: 1.0428 - val_loss: 13673.6533 - val_mse: 13673.6533
    121/121 [==============================] - 0s 834us/step - loss: 0.7441 - mse: 0.7441
    [CV] END learning_rate=0.00044519425836113695, n_hidden=3, n_neurons=27; total time=   2.1s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.0355 - mse: 1.0355 - val_loss: 13176.9150 - val_mse: 13176.9150
    Epoch 2/5
    242/242 [==============================] - 0s 866us/step - loss: 0.9398 - mse: 0.9398 - val_loss: 18983.1172 - val_mse: 18983.1172
    Epoch 3/5
    242/242 [==============================] - 0s 868us/step - loss: 0.9299 - mse: 0.9299 - val_loss: 28577.3691 - val_mse: 28577.3691
    Epoch 4/5
    242/242 [==============================] - 0s 881us/step - loss: 0.9265 - mse: 0.9265 - val_loss: 32331.8711 - val_mse: 32331.8711
    121/121 [==============================] - 0s 817us/step - loss: 0.8629 - mse: 0.8629
    [CV] END learning_rate=0.002096736214950621, n_hidden=1, n_neurons=24; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 0.9794 - mse: 0.9794 - val_loss: 192180.3750 - val_mse: 192180.3750
    Epoch 2/5
    242/242 [==============================] - 0s 939us/step - loss: 0.8244 - mse: 0.8244 - val_loss: 145389.7656 - val_mse: 145389.7656
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8130 - mse: 0.8130 - val_loss: 131413.3125 - val_mse: 131413.3125
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8073 - mse: 0.8073 - val_loss: 122946.6719 - val_mse: 122946.6719
    Epoch 5/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8034 - mse: 0.8034 - val_loss: 121542.7344 - val_mse: 121542.7344
    121/121 [==============================] - 0s 823us/step - loss: 1.1510 - mse: 1.1510
    [CV] END learning_rate=0.002096736214950621, n_hidden=1, n_neurons=24; total time=   1.9s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.2160 - mse: 1.2160 - val_loss: 2009.8264 - val_mse: 2009.8264
    Epoch 2/5
    242/242 [==============================] - 0s 928us/step - loss: 1.0531 - mse: 1.0531 - val_loss: 40550.7188 - val_mse: 40550.7188
    Epoch 3/5
    242/242 [==============================] - 0s 812us/step - loss: 1.0208 - mse: 1.0208 - val_loss: 66782.4297 - val_mse: 66782.4297
    Epoch 4/5
    242/242 [==============================] - 0s 821us/step - loss: 1.0144 - mse: 1.0144 - val_loss: 75719.2969 - val_mse: 75719.2969
    121/121 [==============================] - 0s 1ms/step - loss: 0.7250 - mse: 0.7250
    [CV] END learning_rate=0.002096736214950621, n_hidden=1, n_neurons=24; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.9910 - mse: 0.9910 - val_loss: 117426.2812 - val_mse: 117426.2812
    Epoch 2/5
    242/242 [==============================] - 0s 868us/step - loss: 0.9427 - mse: 0.9427 - val_loss: 98775.3047 - val_mse: 98775.3047
    Epoch 3/5
    242/242 [==============================] - 0s 854us/step - loss: 0.9317 - mse: 0.9317 - val_loss: 90234.4844 - val_mse: 90234.4844
    Epoch 4/5
    242/242 [==============================] - 0s 877us/step - loss: 0.9278 - mse: 0.9278 - val_loss: 85063.1875 - val_mse: 85063.1875
    Epoch 5/5
    242/242 [==============================] - 0s 874us/step - loss: 0.9260 - mse: 0.9260 - val_loss: 81102.0000 - val_mse: 81102.0000
    121/121 [==============================] - 0s 828us/step - loss: 0.8636 - mse: 0.8636
    [CV] END learning_rate=0.0015745689687095968, n_hidden=2, n_neurons=15; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.8722 - mse: 0.8722 - val_loss: 15908.1582 - val_mse: 15908.1582
    Epoch 2/5
    242/242 [==============================] - 0s 896us/step - loss: 0.8360 - mse: 0.8360 - val_loss: 1844.0704 - val_mse: 1844.0704
    Epoch 3/5
    242/242 [==============================] - 0s 906us/step - loss: 0.8200 - mse: 0.8200 - val_loss: 193.2919 - val_mse: 193.2919
    Epoch 4/5
    242/242 [==============================] - 0s 937us/step - loss: 0.8107 - mse: 0.8107 - val_loss: 2920.6826 - val_mse: 2920.6826
    Epoch 5/5
    242/242 [==============================] - 0s 894us/step - loss: 0.8047 - mse: 0.8047 - val_loss: 7379.7544 - val_mse: 7379.7544
    121/121 [==============================] - 0s 930us/step - loss: 1.2061 - mse: 1.2061
    [CV] END learning_rate=0.0015745689687095968, n_hidden=2, n_neurons=15; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0950 - mse: 1.0950 - val_loss: 1826.1127 - val_mse: 1826.1127
    Epoch 2/5
    242/242 [==============================] - 0s 889us/step - loss: 1.0450 - mse: 1.0450 - val_loss: 13793.1777 - val_mse: 13793.1777
    Epoch 3/5
    242/242 [==============================] - 0s 876us/step - loss: 1.0269 - mse: 1.0269 - val_loss: 25364.3750 - val_mse: 25364.3750
    Epoch 4/5
    242/242 [==============================] - 0s 873us/step - loss: 1.0180 - mse: 1.0180 - val_loss: 33199.7266 - val_mse: 33199.7266
    121/121 [==============================] - 0s 804us/step - loss: 0.7282 - mse: 0.7282
    [CV] END learning_rate=0.0015745689687095968, n_hidden=2, n_neurons=15; total time=   1.5s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 3.4423 - mse: 3.4423 - val_loss: 875100.8125 - val_mse: 875100.8125
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.6706 - mse: 1.6706 - val_loss: 361830.9688 - val_mse: 361830.9688
    Epoch 3/5
    242/242 [==============================] - 0s 793us/step - loss: 1.1918 - mse: 1.1918 - val_loss: 140969.6562 - val_mse: 140969.6562
    Epoch 4/5
    242/242 [==============================] - 0s 939us/step - loss: 1.0410 - mse: 1.0410 - val_loss: 47968.8477 - val_mse: 47968.8477
    Epoch 5/5
    242/242 [==============================] - 0s 864us/step - loss: 0.9812 - mse: 0.9812 - val_loss: 11359.3086 - val_mse: 11359.3086
    121/121 [==============================] - 0s 768us/step - loss: 0.9003 - mse: 0.9003
    [CV] END learning_rate=0.0008189085210535839, n_hidden=0, n_neurons=17; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 1.3801 - mse: 1.3801 - val_loss: 282048.3125 - val_mse: 282048.3125
    Epoch 2/5
    242/242 [==============================] - 0s 779us/step - loss: 1.1184 - mse: 1.1184 - val_loss: 159018.3750 - val_mse: 159018.3750
    Epoch 3/5
    242/242 [==============================] - 0s 801us/step - loss: 0.9749 - mse: 0.9749 - val_loss: 80309.6016 - val_mse: 80309.6016
    Epoch 4/5
    242/242 [==============================] - 0s 966us/step - loss: 0.8942 - mse: 0.8942 - val_loss: 34035.1328 - val_mse: 34035.1328
    Epoch 5/5
    242/242 [==============================] - 0s 796us/step - loss: 0.8477 - mse: 0.8477 - val_loss: 10549.7539 - val_mse: 10549.7539
    121/121 [==============================] - 0s 780us/step - loss: 1.1904 - mse: 1.1904
    [CV] END learning_rate=0.0008189085210535839, n_hidden=0, n_neurons=17; total time=   1.6s
    Epoch 1/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.7294 - mse: 1.7294 - val_loss: 1959.4244 - val_mse: 1959.4244
    Epoch 2/5
    242/242 [==============================] - 0s 798us/step - loss: 1.1897 - mse: 1.1897 - val_loss: 6279.4092 - val_mse: 6279.4092
    Epoch 3/5
    242/242 [==============================] - 0s 912us/step - loss: 1.0556 - mse: 1.0556 - val_loss: 11229.3398 - val_mse: 11229.3398
    Epoch 4/5
    242/242 [==============================] - 0s 811us/step - loss: 1.0167 - mse: 1.0167 - val_loss: 16178.8672 - val_mse: 16178.8672
    121/121 [==============================] - 0s 786us/step - loss: 0.7208 - mse: 0.7208
    [CV] END learning_rate=0.0008189085210535839, n_hidden=0, n_neurons=17; total time=   1.3s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.1348 - mse: 1.1348 - val_loss: 24491.6699 - val_mse: 24491.6699
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0662 - mse: 1.0662 - val_loss: 25519.4316 - val_mse: 25519.4316
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0273 - mse: 1.0273 - val_loss: 25877.6504 - val_mse: 25877.6504
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0025 - mse: 1.0025 - val_loss: 25736.9395 - val_mse: 25736.9395
    121/121 [==============================] - 0s 1ms/step - loss: 0.9315 - mse: 0.9315
    [CV] END learning_rate=0.0003103139254608936, n_hidden=2, n_neurons=15; total time=   2.2s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.3437 - mse: 1.3437 - val_loss: 2180.7332 - val_mse: 2180.7332
    Epoch 2/5
    242/242 [==============================] - 0s 913us/step - loss: 1.0281 - mse: 1.0281 - val_loss: 13328.1123 - val_mse: 13328.1123
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9355 - mse: 0.9355 - val_loss: 22830.6777 - val_mse: 22830.6777
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8970 - mse: 0.8970 - val_loss: 28979.9961 - val_mse: 28979.9961
    121/121 [==============================] - 0s 822us/step - loss: 1.3005 - mse: 1.3005
    [CV] END learning_rate=0.0003103139254608936, n_hidden=2, n_neurons=15; total time=   1.8s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.3553 - mse: 1.3553 - val_loss: 446430.2188 - val_mse: 446430.2188
    Epoch 2/5
    242/242 [==============================] - 0s 907us/step - loss: 1.1985 - mse: 1.1985 - val_loss: 329705.9062 - val_mse: 329705.9062
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1407 - mse: 1.1407 - val_loss: 271468.0000 - val_mse: 271468.0000
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.1128 - mse: 1.1128 - val_loss: 239324.0625 - val_mse: 239324.0625
    Epoch 5/5
    242/242 [==============================] - 0s 941us/step - loss: 1.0957 - mse: 1.0957 - val_loss: 218292.5469 - val_mse: 218292.5469
    121/121 [==============================] - 0s 885us/step - loss: 0.7913 - mse: 0.7913
    [CV] END learning_rate=0.0003103139254608936, n_hidden=2, n_neurons=15; total time=   1.9s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.9923 - mse: 0.9923 - val_loss: 119565.0000 - val_mse: 119565.0000
    Epoch 2/5
    242/242 [==============================] - 0s 958us/step - loss: 0.9512 - mse: 0.9512 - val_loss: 91749.7891 - val_mse: 91749.7891
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9425 - mse: 0.9425 - val_loss: 77752.0703 - val_mse: 77752.0703
    Epoch 4/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.9381 - mse: 0.9381 - val_loss: 76165.2656 - val_mse: 76165.2656
    Epoch 5/5
    242/242 [==============================] - 0s 979us/step - loss: 0.9356 - mse: 0.9356 - val_loss: 70646.8359 - val_mse: 70646.8359
    121/121 [==============================] - 0s 833us/step - loss: 0.8731 - mse: 0.8731
    [CV] END learning_rate=0.0029933570302045145, n_hidden=3, n_neurons=16; total time=   1.9s
    Epoch 1/5
    242/242 [==============================] - 1s 1ms/step - loss: 0.8700 - mse: 0.8700 - val_loss: 2790.7131 - val_mse: 2790.7131
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8255 - mse: 0.8255 - val_loss: 17141.0352 - val_mse: 17141.0352
    Epoch 3/5
    242/242 [==============================] - 0s 1ms/step - loss: 0.8092 - mse: 0.8092 - val_loss: 27963.0703 - val_mse: 27963.0703
    Epoch 4/5
    242/242 [==============================] - 0s 938us/step - loss: 0.8020 - mse: 0.8020 - val_loss: 29027.1738 - val_mse: 29027.1738
    121/121 [==============================] - 0s 1ms/step - loss: 1.3151 - mse: 1.3151
    [CV] END learning_rate=0.0029933570302045145, n_hidden=3, n_neurons=16; total time=   1.7s
    Epoch 1/5
    242/242 [==============================] - 1s 2ms/step - loss: 1.0804 - mse: 1.0804 - val_loss: 12018.9805 - val_mse: 12018.9805
    Epoch 2/5
    242/242 [==============================] - 0s 1ms/step - loss: 1.0250 - mse: 1.0250 - val_loss: 38152.5234 - val_mse: 38152.5234
    Epoch 3/5
    242/242 [==============================] - 0s 965us/step - loss: 1.0135 - mse: 1.0135 - val_loss: 42353.9375 - val_mse: 42353.9375
    Epoch 4/5
    242/242 [==============================] - 0s 969us/step - loss: 1.0095 - mse: 1.0095 - val_loss: 44230.9141 - val_mse: 44230.9141
    121/121 [==============================] - 0s 862us/step - loss: 0.7230 - mse: 0.7230
    [CV] END learning_rate=0.0029933570302045145, n_hidden=3, n_neurons=16; total time=   1.7s
    Epoch 1/5
    363/363 [==============================] - 1s 1ms/step - loss: 1.2226 - mse: 1.2226 - val_loss: 296414.3438 - val_mse: 296414.3438
    Epoch 2/5
    363/363 [==============================] - 0s 857us/step - loss: 1.0809 - mse: 1.0809 - val_loss: 80034.7266 - val_mse: 80034.7266
    Epoch 3/5
    363/363 [==============================] - 0s 802us/step - loss: 1.0241 - mse: 1.0241 - val_loss: 128797.2656 - val_mse: 128797.2656
    Epoch 4/5
    363/363 [==============================] - 0s 795us/step - loss: 0.9319 - mse: 0.9319 - val_loss: 83236.0000 - val_mse: 83236.0000
    Epoch 5/5
    363/363 [==============================] - 0s 791us/step - loss: 0.9092 - mse: 0.9092 - val_loss: 94351.1172 - val_mse: 94351.1172





    RandomizedSearchCV(cv=3,
                       estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x7fc2e648ab80>,
                       param_distributions={'learning_rate': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc2e403f850>,
                                            'n_hidden': [0, 1, 2, 3],
                                            'n_neurons': array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29])},
                       verbose=2)




```python

```
