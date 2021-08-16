本文介绍了加载数据后，对数据需要做的一些预处理，包括乱序、数据拆分、缺失值、onehot、embedding、特征缩放、批量归一化等等。

但很多数据预处理工作可以使用pandas, sklearn提供的函数，本文仅介绍了tensorflow专用的数据预处理工具，也就是针对tf.data.Dataset进行处理。


```python
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras,metrics
```

# 1、数据加载

在进行数据预处理前必然需要先加载数据，详细内容可参考加载数据，主要有从内存加载数据和从文件加载数据2种方式。


```python
x = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(x)
print(dataset)
for data in dataset:
    print(data)
```

    <TensorSliceDataset shapes: (), types: tf.int32>
    tf.Tensor(0, shape=(), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    tf.Tensor(5, shape=(), dtype=int32)
    tf.Tensor(6, shape=(), dtype=int32)
    tf.Tensor(7, shape=(), dtype=int32)
    tf.Tensor(8, shape=(), dtype=int32)
    tf.Tensor(9, shape=(), dtype=int32)


等效于：


```python
dataset = tf.data.Dataset.range(10)
print(dataset)
for data in dataset:
    print(data)
```

    <RangeDataset shapes: (), types: tf.int64>
    tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor(1, shape=(), dtype=int64)
    tf.Tensor(2, shape=(), dtype=int64)
    tf.Tensor(3, shape=(), dtype=int64)
    tf.Tensor(4, shape=(), dtype=int64)
    tf.Tensor(5, shape=(), dtype=int64)
    tf.Tensor(6, shape=(), dtype=int64)
    tf.Tensor(7, shape=(), dtype=int64)
    tf.Tensor(8, shape=(), dtype=int64)
    tf.Tensor(9, shape=(), dtype=int64)


## 2、基本的链式转换

我们对dataset做一些基本的转换：


```python
dataset = dataset.repeat(3).batch(7)
for data in dataset:
    print(data)
```

    tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
    tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
    tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
    tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
    tf.Tensor([8 9], shape=(2,), dtype=int64)


我们也可以只保留所有相同大小的批次，删除最后一个批次：


```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(3).batch(7, drop_remainder=True)
for data in dataset:
    print(data)
```

    tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
    tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
    tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
    tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)



```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.map(lambda x:x**2)
for data in dataset:
    print(data)
```

    tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor(1, shape=(), dtype=int64)
    tf.Tensor(4, shape=(), dtype=int64)
    tf.Tensor(9, shape=(), dtype=int64)
    tf.Tensor(16, shape=(), dtype=int64)
    tf.Tensor(25, shape=(), dtype=int64)
    tf.Tensor(36, shape=(), dtype=int64)
    tf.Tensor(49, shape=(), dtype=int64)
    tf.Tensor(64, shape=(), dtype=int64)
    tf.Tensor(81, shape=(), dtype=int64)



```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.filter(lambda x:x<5)
for data in dataset:
    print(data)
```

    tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor(1, shape=(), dtype=int64)
    tf.Tensor(2, shape=(), dtype=int64)
    tf.Tensor(3, shape=(), dtype=int64)
    tf.Tensor(4, shape=(), dtype=int64)



```python
for data in dataset.take(2):
    print(item)
```

    tf.Tensor(2, shape=(), dtype=int64)
    tf.Tensor(2, shape=(), dtype=int64)


# 3、乱序
我们介绍一下基本的乱序方式，这种方式要求数据能完全加载进内存。若是大文件的数据乱序，可以参考数据加载中的内容。


```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for data in dataset:
    print(data)
```

    tf.Tensor([0 2 3 6 7 9 4], shape=(7,), dtype=int64)
    tf.Tensor([5 0 1 1 8 6 5], shape=(7,), dtype=int64)
    tf.Tensor([4 8 7 1 2 3 0], shape=(7,), dtype=int64)
    tf.Tensor([5 4 2 7 8 9 9], shape=(7,), dtype=int64)
    tf.Tensor([3 6], shape=(2,), dtype=int64)


如你所知，当训练集中的实例相互独立且分布均匀时，梯度下降效果最佳。确保这一点的一种简单方法是使用shuffle（）方法对实例进行乱序。它会创建一个新的数据集，该数据集首先将源数据集的第一项元素填充到缓冲区中。然后无论任何时候要求提供一个元素，它都会从缓冲区中随机取出一个元素，并用源数据集中的新元素替换它，直到完全遍历源数据集为止。它将继续从缓冲区中随机抽取元素直到其为空。你必须指定缓冲区的大小，重要的是要使其足够大，否则乱序不会非常有效。不要超出你有的RAM的数量，即使你有足够的RAM，也不需要超出数据集的大小。如果每次运行程序都想要相同的随机顺序，你可以提供随机种子。


# 4、pipeline综合处理
本部分示范了使用pipeline完成一系列的预处理。

首先我们先准备一下csv文件，文件内容是california_housing数据。


```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_
print(X_mean, X_std)
```

    [ 3.89175860e+00  2.86245478e+01  5.45593655e+00  1.09963474e+00
      1.42428122e+03  2.95886657e+00  3.56464315e+01 -1.19584363e+02] [1.90927329e+00 1.26409177e+01 2.55038070e+00 4.65460128e-01
     1.09576000e+03 2.36138048e+00 2.13456672e+00 2.00093304e+00]


将数据写到csv文件中：


```python
def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
```

数据文件准备好了，我们看一下数据：


```python
pd.read_csv(train_filepaths[0]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedianHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5214</td>
      <td>15.0</td>
      <td>3.049945</td>
      <td>1.106548</td>
      <td>1447.0</td>
      <td>1.605993</td>
      <td>37.63</td>
      <td>-122.43</td>
      <td>1.442</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.3275</td>
      <td>5.0</td>
      <td>6.490060</td>
      <td>0.991054</td>
      <td>3464.0</td>
      <td>3.443340</td>
      <td>33.69</td>
      <td>-117.39</td>
      <td>1.687</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.1000</td>
      <td>29.0</td>
      <td>7.542373</td>
      <td>1.591525</td>
      <td>1328.0</td>
      <td>2.250847</td>
      <td>38.44</td>
      <td>-122.98</td>
      <td>1.621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.1736</td>
      <td>12.0</td>
      <td>6.289003</td>
      <td>0.997442</td>
      <td>1054.0</td>
      <td>2.695652</td>
      <td>33.55</td>
      <td>-117.70</td>
      <td>2.621</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0549</td>
      <td>13.0</td>
      <td>5.312457</td>
      <td>1.085092</td>
      <td>3297.0</td>
      <td>2.244384</td>
      <td>33.93</td>
      <td>-116.93</td>
      <td>0.956</td>
    </tr>
  </tbody>
</table>
</div>




```python
with open(train_filepaths[0]) as f:
    for i in range(5):
        print(f.readline(), end="")
```

    MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedianHouseValue
    3.5214,15.0,3.0499445061043287,1.106548279689234,1447.0,1.6059933407325193,37.63,-122.43,1.442
    5.3275,5.0,6.490059642147117,0.9910536779324056,3464.0,3.4433399602385686,33.69,-117.39,1.687
    3.1,29.0,7.5423728813559325,1.5915254237288134,1328.0,2.2508474576271187,38.44,-122.98,1.621
    7.1736,12.0,6.289002557544757,0.9974424552429667,1054.0,2.6956521739130435,33.55,-117.7,2.621



```python
train_filepaths
```




    ['datasets/housing/my_train_00.csv',
     'datasets/housing/my_train_01.csv',
     'datasets/housing/my_train_02.csv',
     'datasets/housing/my_train_03.csv',
     'datasets/housing/my_train_04.csv',
     'datasets/housing/my_train_05.csv',
     'datasets/housing/my_train_06.csv',
     'datasets/housing/my_train_07.csv',
     'datasets/housing/my_train_08.csv',
     'datasets/housing/my_train_09.csv',
     'datasets/housing/my_train_10.csv',
     'datasets/housing/my_train_11.csv',
     'datasets/housing/my_train_12.csv',
     'datasets/housing/my_train_13.csv',
     'datasets/housing/my_train_14.csv',
     'datasets/housing/my_train_15.csv',
     'datasets/housing/my_train_16.csv',
     'datasets/housing/my_train_17.csv',
     'datasets/housing/my_train_18.csv',
     'datasets/housing/my_train_19.csv']



下面我们开始数据预处理。

我们先将文件加载到一个dataset并乱序：


```python
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
for filepath in filepath_dataset:
    print(filepath)
```

    tf.Tensor(b'datasets/housing/my_train_05.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_16.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_01.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_17.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_00.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_14.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_10.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_02.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_12.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_19.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_07.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_09.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_13.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_15.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_11.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_18.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_04.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_06.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_03.csv', shape=(), dtype=string)
    tf.Tensor(b'datasets/housing/my_train_08.csv', shape=(), dtype=string)


然后我们使用interleave()函数一次读取n_reader个文件并交织他们的行：


```python
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)

for line in dataset.take(5):
    print(line.numpy())
```

    b'4.5909,16.0,5.475877192982456,1.0964912280701755,1357.0,2.9758771929824563,33.63,-117.71,2.418'
    b'2.4792,24.0,3.4547038327526134,1.1341463414634145,2251.0,3.921602787456446,34.18,-118.38,2.0'
    b'4.2708,45.0,5.121387283236994,0.953757225433526,492.0,2.8439306358381504,37.48,-122.19,2.67'
    b'2.1856,41.0,3.7189873417721517,1.0658227848101265,803.0,2.0329113924050635,32.76,-117.12,1.205'
    b'4.1812,52.0,5.701388888888889,0.9965277777777778,692.0,2.4027777777777777,33.73,-118.31,3.215'


interleave（）方法将创建一个数据集，该数据集将从filepath_dataset中拉出5个文件路径，对于每个路径，它将调用你为其提供的函数（在此示例中为lambda）来创建新的数据集（在此示例中为TextLineDataset）。为了清楚起见，在此阶段总共有7个数据集：文件路径数据集、交织数据集和由交织数据集在内部创建的5个TextLineDataset。当我们遍历交织数据集时，它将循环遍历这5个TextLineDatasets，每次读取一行，直到所有数据集都读出为止。然后它将从filepath_dataset获取剩下的5个文件路径，并以相同的方式对它们进行交织，以此类推，直到读完文件路径。

默认情况下，interleave（）不使用并行。它只是顺序地从每个文件中一次读取一行。如果你希望它并行读取文件，则可以将num_parallel_calls参数设置为所需的线程数（请注意map（）方法也具有此参数）。你甚至可以将其设置为tf.data.experimental.AUTOTUNE，使TensorFlow根据可用的CPU动态地选择合适的线程数（但是目前这还是实验功能）。

由于下面要用到tf.io.decode_csv()，我们先简单介绍一下：


```python
record_defaults=[0, np.nan, tf.constant(np.nan, dtype=tf.float64), "Hello", tf.constant([])]
parsed_fields = tf.io.decode_csv('1,2,3,4,5', record_defaults)
parsed_fields
```




    [<tf.Tensor: shape=(), dtype=int32, numpy=1>,
     <tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
     <tf.Tensor: shape=(), dtype=float64, numpy=3.0>,
     <tf.Tensor: shape=(), dtype=string, numpy=b'4'>,
     <tf.Tensor: shape=(), dtype=float32, numpy=5.0>]



上述示例由于第4个字段的默写值是'hello'，所以这个字段的所有值都会被认为是字符串。


```python
parsed_fields = tf.io.decode_csv(',,,,5', record_defaults)
parsed_fields
```




    [<tf.Tensor: shape=(), dtype=int32, numpy=0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=nan>,
     <tf.Tensor: shape=(), dtype=float64, numpy=nan>,
     <tf.Tensor: shape=(), dtype=string, numpy=b'Hello'>,
     <tf.Tensor: shape=(), dtype=float32, numpy=5.0>]




```python
try:
    parsed_fields = tf.io.decode_csv(',,,,', record_defaults)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

    Field 4 is required but missing in record 0! [Op:DecodeCSV]


最后一个字段没有默认值，所以必填。

好，下面我们开始定义预处理函数。


```python
n_inputs = 8 # X_train.shape[-1]

@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
```




    (<tf.Tensor: shape=(8,), dtype=float32, numpy=
     array([ 0.16579157,  1.216324  , -0.05204565, -0.39215982, -0.5277444 ,
            -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>,
     <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.782], dtype=float32)>)



下面我们把上述所有的预处理功能汇总起来，包括读取数据、batch、数据加工、prefetch()等：


```python
def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
```

数据准备好了，开始构建并训练模型：


```python
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=valid_set)
```

    Epoch 1/10
    362/362 [==============================] - 1s 1ms/step - loss: 1.4679 - val_loss: 21.5124
    Epoch 2/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.8735 - val_loss: 0.6648
    Epoch 3/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.6317 - val_loss: 0.6196
    Epoch 4/10
    362/362 [==============================] - 0s 963us/step - loss: 0.5933 - val_loss: 0.5669
    Epoch 5/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5629 - val_loss: 0.5402
    Epoch 6/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5693 - val_loss: 0.5209
    Epoch 7/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5231 - val_loss: 0.6130
    Epoch 8/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5074 - val_loss: 0.4818
    Epoch 9/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.4963 - val_loss: 0.4904
    Epoch 10/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5023 - val_loss: 0.4585





    <tensorflow.python.keras.callbacks.History at 0x7fbb482bc5e0>



完整代码如下：


```python
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras,metrics
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

X_mean = [ 3.89175860e+00,2.86245478e+01,5.45593655e+00,  1.09963474e+00,
  1.42428122e+03,  2.95886657e+00,  3.56464315e+01, -1.19584363e+02] 
X_std = [1.90927329e+00, 1.26409177e+01, 2.55038070e+00, 4.65460128e-01,
 1.09576000e+03, 2.36138048e+00, 2.13456672e+00, 2.00093304e+00]

n_inputs = 8 # X_train.shape[-1]

@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

train_set = csv_reader_dataset('datasets/housing/my_train*', repeat=None)
valid_set = csv_reader_dataset('datasets/housing/my_test*')
test_set = csv_reader_dataset('datasets/housing/my_valid*')

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=valid_set)
```

    Epoch 1/10
    362/362 [==============================] - 1s 1ms/step - loss: 1.4679 - val_loss: 0.8987
    Epoch 2/10
    362/362 [==============================] - 0s 959us/step - loss: 0.8735 - val_loss: 0.6598
    Epoch 3/10
    362/362 [==============================] - 0s 944us/step - loss: 0.6317 - val_loss: 0.6097
    Epoch 4/10
    362/362 [==============================] - 0s 955us/step - loss: 0.5933 - val_loss: 0.5795
    Epoch 5/10
    362/362 [==============================] - 0s 921us/step - loss: 0.5629 - val_loss: 0.5553
    Epoch 6/10
    362/362 [==============================] - 0s 1ms/step - loss: 0.5693 - val_loss: 0.5344
    Epoch 7/10
    362/362 [==============================] - 0s 964us/step - loss: 0.5231 - val_loss: 0.5146
    Epoch 8/10
    362/362 [==============================] - 0s 931us/step - loss: 0.5074 - val_loss: 0.5003
    Epoch 9/10
    362/362 [==============================] - 0s 962us/step - loss: 0.4963 - val_loss: 0.4884
    Epoch 10/10
    362/362 [==============================] - 0s 938us/step - loss: 0.5023 - val_loss: 0.4785





    <tensorflow.python.keras.callbacks.History at 0x7f6790456580>



# 5、prefect

通过最后调用prefetch（1），我们正在创建一个数据集，该数据集将尽最大可能总是提前准备一个批次。换句话说，当我们的训练算法正处理一个批次时，数据集已经并行工作以准备下一批次了（例如从磁盘中读取数据并对其进行预处理）。如图133所示，这可以显著提高性能。如果我们确保加载和预处理是多线程的（通过在调用interleave（）和map（）时设置num_parallel_calls），我们可以在CPU上利用多个内核，希望准备一个批次数据的时间比在GPU上执行一个训练步骤的时间要短一些：这样，GPU将达到几乎100％的利用率（从CPU到GPU的数据传输时间除外），并且训练会运行得更快。

如果数据集足够小，可以放到内存里，则可以使用数据集的cache（）方法将其内容缓存到RAM中，从而显著加快训练速度。通常应该在加载和预处理数据之后，但在乱序、重复、批处理和预取之前执行此操作。这样，每个实例仅被读取和预处理一次（而不是每个轮次一次），但数据仍会在每个轮次进行不同的乱序，并且仍会提前准备下一批次。

现在你知道如何构建有效的输入流水线来从多个文本文件加载和预处理数据了。我们已经讨论了最常见的数据集方法，但还有更多方法：concatenate（）、zip（）、window（）、reduce（）、shard（）、flat_map（）和padded_batch（）。还有另外两个类方法：from_generator（）和from_tensors（），它们分别从Python生成器或张量列表创建新的数据集。请查看API文档以了解更多详细信息。还要注意，tf.data.experimental中有一些实验性功能，其中许多功能可能会在将来的版本中成为核心API（查看CsvDataset类以及make_csv_dataset（）方法，该方法负责推断每一列的类型）。


# 6、定义预处理层
为神经网络准备数据需要将所有特征转换为数值特征，通常将其归一化等。特别是如果你的数据包含分类特征或文本特征，则需要将它们转换为数字。在准备数据文件时，可以使用任何你喜欢的工具（例如NumPy、pandas或ScikitLearn）提前完成此操作。或者，你可以在使用DataAPI加载数据时动态地预处理数据（例如使用数据集的map（）方法，如我们之前看到的），也可以在模型中直接包含预处理层。现在让我们来看最后一个选项。

例如，这是使用Lambda层实现标准化层的方法。对于每个特征，它减去均值并除以其标准差（加上一个微小的平滑项，以避免被零除）：
model = keras.models.Sequential([
    keras.layers.Lambda(lambda inputs: (input-means)/(stds+eps))
])
这不太难！但是你可能更喜欢使用一个很好的自包含自定义层（非常类似于ScikitLearn的StandardScaler），而不是像means和stds之类的全局变量：


```python
class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)
    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())
```

然后就可以使用这个Layer了：
standardization = Standardization(input_shape=[28, 28])
standardization.adapt(sample_images)
model = keras.models.Sequential([
    standardization,
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
# 7、onehot及类别数字化
我们在探讨的加州住房数据集中的ocean_proximity特征时，它是一个具有5个可能值的分类特征："<1HOCEAN""INLAND""NEAROCEAN""NEARBAY"和"ISLAND"。再将其提供给神经网络之前我们需要对该特征进行编码。由于类别很少，我们可以使用独热编码。为此我们首先需要将每个类别映射到其索引（0到4），这可以使用查找表来完成：


```python
vocab = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```

让我们看一下这段代码：
* 我们首先定义词汇表：这是所有可能类别的列表。
* 然后，我们创建带有相应索引（0到4）的张量。
* 接下来，我们为查找表创建一个初始化程序，将类别列表及其对应的索引传递给它。在此示例中，我们已经有此数据，因此我们使用KeyValueTensorInitializer。但是如果类别在文本文件中列出（每行一个类别），我们要使用TextFileInitializer。
* 在最后两行中，我们创建了查找表，为其提供了初始化程序并指定了词汇表外（outofvocabulary，oov）桶的数量。如果我们查找词汇表中不存在的类别，则查找表将计算该类别的哈希并将这个未知类别分配给oov桶之中的一个。它们的索引从已知类别开始，因此在此示例中，两个oov桶的索引为5和6。

为什么要使用oov桶？如果类别数量很大（例如邮政编码、城市、单词、产品或用户）并且数据集也很大，或者它们一直在变化，那么得到类别的完整列表可能不是很方便。一种解决方法是基于数据样本（而不是整个训练集）定义词汇表，并为不在数据样本中的其他类别添加一些桶。你希望在训练期间找到的类别越多，就应该使用越多的oov桶。如果没有足够的oov桶，就会发生冲突：不同的类别最终会出现在同一个桶中，因此神经网络将无法区分它们（至少不是基于这个特征）。

ok, onehot准备好了，现在开始转换： 


```python
categories = tf.constant(['NEAR BAY', 'INLAND', 'DESERT'])
cat_indices = table.lookup(categories)
print(cat_indices)
```

    tf.Tensor([3 1 5], shape=(3,), dtype=int64)



```python
cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab)+num_oov_buckets)
print(cat_one_hot)
```

    tf.Tensor(
    [[0. 0. 0. 1. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0.]], shape=(3, 7), dtype=float32)


可以看出来，上面是先将文本转成数字，然后再做onehot。如果数据本身已经是数字，那就可以直接做onehot了：


```python
num = tf.constant([1,2,3,4,5])
num_one_hot = tf.one_hot(num, 6)
print(num_one_hot)
```

    tf.Tensor(
    [[0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]], shape=(5, 6), dtype=float32)


如你所见，"NEARBAY"被映射到索引3，未知类别"DESERT"被映射到两个oov桶之一（在索引5），而"INLAND"被映射到索引1，两次。然后我们使用tf.one_hot（）对这些索引进行独热编码。注意我们必须告诉该函数索引的总数，该总数等于词汇表大小加上oov桶的数量。现在你知道如何使用TensorFlow把分类特征编码为独热向量！

就像前面一样，将所有这些逻辑捆绑到一个包含所有的类中并不是很困难。它的adapt（）方法使用数据样本并提取其包含的所有不同类别。它创建一个查找表将每个类别映射到其索引（包括使用oov桶的未知类别）。然后其call（）方法使用查找表将输入类别映射到其索引。还有一个更好的消息：当你阅读本书时，Keras可能会包含一个名为keras.layers.TextVectorization的层，该层能够准确地做到这一点：它的adapt（）方法从数据样本中提取词汇表，而其call（）方法会将每个类别转换为其词汇表中的索引。如果要将这些索引转换为独热向量，则可以在模型的开头添加此层，然后添加将应用tf.one_hot（）函数的Lambda层。

这可能不是最佳解决方法。每个独热向量的大小是词汇表长度加上oov桶数。当只有几个可能的类别时这很可行，但是如果词汇表很大，则使用嵌入对它们进行编码会更加有效。

**根据经验，如果类别数少于10，则通常采用独热编码方式。（但数字可能会有所不同！）如果类别数大于50（通常这种情况需要使用哈希桶），通常最好使用嵌入。在10到50个类别中，你可能需要尝试两种方法，然后看看哪种最适合你。**


# 8、使用embedding分类特征

正如上面所言，当类别很多时，使用embedding做分类会更合适。

嵌入是表示类别的可训练密集向量。默认情况下，嵌入是随机初始化的，例如，"NEARBAY"类别最初可以由诸如[0.131，0.890]的随机向量表示，而"NEAROCEAN"类别可以由[0.631，0.791]表示。在此示例中，我们使用2D嵌入，但是维度是可以调整的超参数。由于这些嵌入是可训练的，因此它们在训练过程中会逐步改善。由于它们代表的类别相当相似，“梯度下降”肯定最终会把它们推到接近的位置，而把它们推离"INLAND"类别的嵌入（见图134）。实际上，表征越好，神经网络就越容易做出准确的预测，因此训练使嵌入成为类别的有用表征。这称为表征学习（我们将在第17章中看到其他类型的表征学习）。

嵌入通常不仅是当前任务的有用表示，而且很多时候这些相同的嵌入可以成功地重用于其他任务。最常见的示例是词嵌入（即单个单词的嵌入）：在执行自然语言处理任务时，与训练自己的词嵌入相比，重用预先训练的词嵌入通常效果会更好。

使用向量来表示词的想法可以追溯到20世纪60年代，许多复杂的技术已被用来生成有用的向量，包括使用神经网络。但是事情真正在2013年取得了成功，当时TomášMikolov和其他Google研究人员发表了一篇论文，描述了一种使用神经网络学习词嵌入的有效技术，大大优于以前的尝试。这使他们能够在非常大的文本语料库上学习嵌入：他们训练了一个神经网络来预测任何给定单词附近的单词，并获得了惊人的词嵌入。例如，同义词具有非常接近的嵌入，法国、西班牙和意大利等与语义相关的词最终聚类在一起。

但是这不仅与邻近性有关：词嵌入还沿着嵌入空间中有意义的轴进行组织。这是一个著名的示例：如果计算KingMan+Woman（添加和减去这些单词的嵌入向量），则结果非常接近Queen单词的嵌入（见图135）。换句话说，词嵌入编码了性别的概念！同样，你可以计算MadridSpain+France，其结果接近Paris（巴黎），这似乎表明首都的概念也在嵌入中进行了编码。

不幸的是，词嵌入有时会捕获我们最严重的偏见。例如，尽管它们正确地学习到男人是国王，女人是女王，但它们似乎也学习到了男人是医生，而女人是护士：这是一种性别歧视！公平地说，正如MalvinaNissim等人在2019年的论文中指出的那样，这个特定示例可能被夸大了。不管怎样，确保深度学习算法的公平性是重要且活跃的研究课题。

让我们看一下如何手动实现嵌入以了解它们的工作原理（然后我们将使用一个简单的Keras层）。首先我们需要创建一个包含每个类别嵌入的嵌入矩阵，并随机初始化。每个类别和每个oov桶都有一行，每个嵌入维度都有一列。

与onehot一样，我们先把类别转成数字ID：


```python
vocab = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```


```python
然后为数字生成emdding:
```


```python

embedding_dim = 2
embed_init = tf.random.uniform([len(vocab)+num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)
```

    <tf.Variable 'Variable:0' shape=(7, 2) dtype=float32, numpy=
    array([[0.5204637 , 0.98164105],
           [0.5929651 , 0.87584126],
           [0.31663084, 0.7100874 ],
           [0.13743544, 0.51320267],
           [0.91944194, 0.36314178],
           [0.9810499 , 0.23673522],
           [0.73040426, 0.5197154 ]], dtype=float32)>


上述已经为每个可能的类别都初始化了一个向量，现在我们使用这些向量：


```python
categories = tf.constant(['NEAR OCEAN', 'DESERT', 'INLAND', 'INLAND'])
cat_indices = table.lookup(categories)
print(cat_indices)
```

    tf.Tensor([2 5 1 1], shape=(4,), dtype=int64)



```python
tf.nn.embedding_lookup(embedding_matrix, cat_indices)
```




    <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
    array([[0.577116  , 0.3422978 ],
           [0.8767899 , 0.1696508 ],
           [0.38885117, 0.7005861 ],
           [0.38885117, 0.7005861 ]], dtype=float32)>



同样的，如果本身类别就已经是数字ID，则不需要转换成，直接做embedding即可：


```python
vocab = tf.range(5)
embedding_dim = 2
embed_init = tf.random.uniform([len(vocab)+2, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)
```

    <tf.Variable 'Variable:0' shape=(7, 2) dtype=float32, numpy=
    array([[0.2896663 , 0.42502713],
           [0.96564734, 0.3244139 ],
           [0.6590884 , 0.6410396 ],
           [0.01374483, 0.9046582 ],
           [0.21921551, 0.93730116],
           [0.2230134 , 0.818697  ],
           [0.7004936 , 0.55174303]], dtype=float32)>



```python
cat = tf.range(3)
tf.nn.embedding_lookup(embedding_matrix, cat)
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[0.2896663 , 0.42502713],
           [0.96564734, 0.3244139 ],
           [0.6590884 , 0.6410396 ]], dtype=float32)>



tf.nn.embedding_lookup（）函数以给定的索引查找在嵌入矩阵中的行，这就是它所做的全部。例如，查找表说"INLAND"类别位于索引1，因此tf.nn.embedding_lookup（）函数返回嵌入矩阵中第1行的嵌入（两次）。

Keras提供了一个keras.layers.Embedding层来处理嵌入矩阵（默认情况下是可训练的）。创建层时，它将随机初始化嵌入矩阵，然后使用某些类别索引进行调用时，它将返回嵌入矩阵中这些索引处的行


```python
vocab = tf.range(5)
embedding = keras.layers.Embedding(input_dim=len(vocab)+2, output_dim=2)
embedding(tf.range(3))
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[-0.02937205, -0.0260554 ],
           [ 0.02903572,  0.00200558],
           [-0.00577971,  0.03343553]], dtype=float32)>



实时上最后一种才是我们最常用的方式。

onehot是将类别转换成tf.constant，它是一个常量；而embedding是将类别转成成tf.Variable，它是一个变量，会在模型训练中训练调整。所以一般情况下，onehot可以在模型训练前先做好数据预处理，而embedding是作为模型训练的一部分。


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# 9、FeatureColumns API示例

TensorFlow团队正在努力提供一组标准的Keras预处理层。在你阅读本书时，可能可以使用它；但是届时API可能会稍有变化，因此如果有任何意外，请参考本章的notebook。这个新的API可能会取代现有的FeatureColumnsAPI，该API较难使用且不直观。

以下是FeatureColumns API的示例。


```python
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras,metrics
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing_median_age = tf.feature_column.numeric_column("housing_median_age")

age_mean, age_std = X_mean[1], X_std[1]  # The median age is column in 1
housing_median_age = tf.feature_column.numeric_column(
    "housing_median_age", normalizer_fn=lambda x: (x - age_mean) / age_std)
```


```python
median_income = tf.feature_column.numeric_column("median_income")
bucketized_income = tf.feature_column.bucketized_column(
    median_income, boundaries=[1.5, 3., 4.5, 6.])
bucketized_income
```




    BucketizedColumn(source_column=NumericColumn(key='median_income', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), boundaries=(1.5, 3.0, 4.5, 6.0))




```python
ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list(
    "ocean_proximity", ocean_prox_vocab)

ocean_proximity
```




    VocabularyListCategoricalColumn(key='ocean_proximity', vocabulary_list=('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'), dtype=tf.string, default_value=-1, num_oov_buckets=0)




```python
# Just an example, it's not used later on
city_hash = tf.feature_column.categorical_column_with_hash_bucket(
    "city", hash_bucket_size=1000)
city_hash

```




    HashedCategoricalColumn(key='city', hash_bucket_size=1000, dtype=tf.string)




```python
bucketized_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=[-1., -0.5, 0., 0.5, 1.]) # age was scaled
age_and_ocean_proximity = tf.feature_column.crossed_column(
    [bucketized_age, ocean_proximity], hash_bucket_size=100)


```


```python
latitude = tf.feature_column.numeric_column("latitude")
longitude = tf.feature_column.numeric_column("longitude")
bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=list(np.linspace(32., 42., 20 - 1)))
bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=list(np.linspace(-125., -114., 20 - 1)))
location = tf.feature_column.crossed_column(
    [bucketized_latitude, bucketized_longitude], hash_bucket_size=1000)
```


```python
ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)
```


```python
ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,
                                                           dimension=2)
```




```python
median_house_value = tf.feature_column.numeric_column("median_house_value")
```


```python
columns = [housing_median_age, median_house_value]
feature_descriptions = tf.feature_column.make_parse_example_spec(columns)
feature_descriptions
```




    {'housing_median_age': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None),
     'median_house_value': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=None)}




```python
Example = tf.train.Example
BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features

with tf.io.TFRecordWriter("my_data_with_features.tfrecords") as f:
    for x, y in zip(X_train[:, 1:2], y_train):
        example = Example(features=Features(feature={
            "housing_median_age": Feature(float_list=FloatList(value=[x])),
            "median_house_value": Feature(float_list=FloatList(value=[y]))
        }))
        f.write(example.SerializeToString())
```


```python
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
```


```python
def parse_examples(serialized_examples):
    examples = tf.io.parse_example(serialized_examples, feature_descriptions)
    targets = examples.pop("median_house_value") # separate the targets
    return examples, targets

batch_size = 32
dataset = tf.data.TFRecordDataset(["my_data_with_features.tfrecords"])
dataset = dataset.repeat().shuffle(10000).batch(batch_size).map(parse_examples)
```


```python
columns_without_target = columns[:-1]
model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns=columns_without_target),
    keras.layers.Dense(1)
])
model.compile(loss="mse",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])
model.fit(dataset, steps_per_epoch=len(X_train) // batch_size, epochs=5)
```

    Epoch 1/5
    WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'> input: {'housing_median_age': <tf.Tensor 'IteratorGetNext:0' shape=(None, 1) dtype=float32>}
    Consider rewriting this model with the Functional API.
    WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor, but we receive a <class 'dict'> input: {'housing_median_age': <tf.Tensor 'IteratorGetNext:0' shape=(None, 1) dtype=float32>}
    Consider rewriting this model with the Functional API.
    362/362 [==============================] - 1s 889us/step - loss: 3.9461 - accuracy: 0.0016
    Epoch 2/5
    362/362 [==============================] - 0s 773us/step - loss: 1.8772 - accuracy: 0.0024
    Epoch 3/5
    362/362 [==============================] - 0s 626us/step - loss: 1.4831 - accuracy: 0.0032
    Epoch 4/5
    362/362 [==============================] - 0s 581us/step - loss: 1.3460 - accuracy: 0.0028
    Epoch 5/5
    362/362 [==============================] - 0s 578us/step - loss: 1.3329 - accuracy: 0.0032





    <tensorflow.python.keras.callbacks.History at 0x7f67b4779fa0>




```python
some_columns = [ocean_proximity_embed, bucketized_income]
dense_features = keras.layers.DenseFeatures(some_columns)
dense_features({
    "ocean_proximity": [["NEAR OCEAN"], ["INLAND"], ["INLAND"]],
    "median_income": [[3.], [7.2], [1.]]
})
```




    <tf.Tensor: shape=(3, 7), dtype=float32, numpy=
    array([[ 0.        ,  0.        ,  1.        ,  0.        ,  0.        ,
            -0.14504611,  0.7563394 ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
            -1.1119912 ,  0.56957847],
           [ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            -1.1119912 ,  0.56957847]], dtype=float32)>




```python

```
