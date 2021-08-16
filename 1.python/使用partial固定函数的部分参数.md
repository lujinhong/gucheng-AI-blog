functools模块这样为为高阶函数提供支持，partial 就是其中的一个，它的主要作用是：
**把一个函数的某些参数给固定住，返回一个新的函数。**

我们先看一个例子：


```python
def multiply(x, y=3):
    return x*y
print(multiply(5))
```

    15


我们可以使用上述默认参数的方式来减少需要提供的参数，但对于一些并非由你自己定义的函数，或者其默认值不适合你的需求，而你又不想频繁的填写参数，这时你就可以使用patial了：


```python
from functools import partial
double = partial(multiply, y=2)
triple = partial(multiply, y=3)

print(double(5))
print(triple(5))
```

    10
    15


比如在DNN中，经常有一些相同或者近似的层，此时就可以使用partial了：



```python
import tensorflow as tf
from tensorflow import keras

from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding='SAME')

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax')
])
```


```python

```
