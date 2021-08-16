本文介绍了如何加载各种数据源，以生成可以用于tensorflow使用的数据集，一般指Dataset。主要包括以下几类数据源：
* 预定义的公共数据源
* 内存中的数据
* csv文件
* TFRecord
* 任意格式的数据文件
* 稀疏数据格式文件


**更完整的数据加载方式请参考：https://www.tensorflow.org/tutorials/load_data/images?hl=zh-cn**



```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
print(tf.__version__)

```

    2.5.0


## 1、预定义的公共数据源

为了方便使用，tensorflow将一些常用的数据源预先处理好，用户可以直接使用。完整内容请参考：

https://www.tensorflow.org/datasets/overview

tensorflow的数据集有2种类型：
* 简单的数据集，使用keras.datasets.***.load_data()即可以得到数据
* 在tensorflow_datasets中的数据集。

### 1.1 简单数据集

常见的有mnist，fashion_mnist等返回的是numpy.ndarray的数据格式。


```python
(x_train_all,y_train_all),(x_test,y_test) = keras.datasets.fashion_mnist.load_data()
print(type(x_train_all))
x_train_all[5,1],y_train_all[5]
```

    <class 'numpy.ndarray'>





    (array([  0,   0,   0,   1,   0,   0,  20, 131, 199, 206, 196, 202, 242,
            255, 255, 250, 222, 197, 206, 188, 126,  17,   0,   0,   0,   0,
              0,   0], dtype=uint8),
     2)




```python
(x_train_all,y_train_all),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train_all[14,14],y_train_all[14]
```




    (array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  29,
            255, 254, 109,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0], dtype=uint8),
     1)



### 1.2 tensorflow_datasets
tensorflow_datasets提供的数据集。

题外话，由于tensorflow dataset被墙，请自备梯子。若在服务器等无法fq的环境，可以先在其它机器下载好，数据一般会下载到~/tensorflow_datasets目录下，然后把目录下的数据集上传到服务器相同的目录即可。tensorflow会优先检查本地目录是否有文件，再去下载。

通过tfds.load()可以方便的加载数据集，返回值为tf.data.Dataset类型；如果with_info=True，则返回（Dataset,ds_info)组成的tuple。
完整内容可参考：
https://www.tensorflow.org/datasets/api_docs/python/tfds/load

### 1.3 flower数据集



```python
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)

# 画一些花朵看一下
plt.figure(figsize=(12, 10))
index = 0
for image, label in train_set_raw.take(9):
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

plt.show()
```


    
![png](tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_files/tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_6_0.png)
    



```python

```


```python

```

## 2、加载内存中的数据

本部分内容主要将内存中的数据（numpy）转换为Dataset。

from_tensor_slices()将numpy数组中的每一个元素都转化为tensorflow Dataset中的一个元素：


```python
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)
for item in dataset:
    print(item)
```

    <TensorSliceDataset shapes: (), types: tf.int64>
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


我们可以对这个Dataset做各种的操作，比如：


```python
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
```

    tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
    tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
    tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
    tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
    tf.Tensor([8 9], shape=(2,), dtype=int64)


我们还可以将多个数组整合成一个Dataset，常见的比如feature和label组合成训练样本：


```python
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())
```

    <TensorSliceDataset shapes: ((2,), ()), types: (tf.int64, tf.string)>
    [1 2] b'cat'
    [3 4] b'dog'
    [5 6] b'fox'


或者这样做：


```python
dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x,
                                               "label": y})
for item in dataset4:
    print(item["feature"].numpy(), item["label"].numpy())
```

    [1 2] b'cat'
    [3 4] b'dog'
    [5 6] b'fox'


## 3、加载csv文件的数据

本部分介绍了tensorflow如何加载csv文件生成Dataset。**除了本部分介绍的方法外，如果数据量不大，也可以使用pandas.read_csv加载到内存后，再使用上面介绍的from_tensor_slice()。**

### 3.1 生成csv文件
由于我们没有现成的csv文件，所以我们使用预定义好的公共数据集生成csv文件：


```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 获取数据
housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 写入csv文件
output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def save_to_csv(output_dir, data, name_prefix,
                header=None, n_parts=10):
    path_format = os.path.join(output_dir, "{}_{:02d}.csv")
    filenames = []
    
    for file_idx, row_indices in enumerate(
        np.array_split(np.arange(len(data)), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filenames.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join(
                    [repr(col) for col in data[row_index]]))
                f.write('\n')
    return filenames

train_data = np.c_[x_train_scaled, y_train]
valid_data = np.c_[x_valid_scaled, y_valid]
test_data = np.c_[x_test_scaled, y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)

train_filenames = save_to_csv(output_dir, train_data, "train",
                              header_str, n_parts=20)
valid_filenames = save_to_csv(output_dir, valid_data, "valid",
                              header_str, n_parts=10)
test_filenames = save_to_csv(output_dir, test_data, "test",
                             header_str, n_parts=10)

# 看一下生成的文件：
import pprint
print("train filenames:")
pprint.pprint(train_filenames)
print("valid filenames:")
pprint.pprint(valid_filenames)
print("test filenames:")
pprint.pprint(test_filenames)
```

    (11610, 8) (11610,)
    (3870, 8) (3870,)
    (5160, 8) (5160,)
    train filenames:
    ['generate_csv/train_00.csv',
     'generate_csv/train_01.csv',
     'generate_csv/train_02.csv',
     'generate_csv/train_03.csv',
     'generate_csv/train_04.csv',
     'generate_csv/train_05.csv',
     'generate_csv/train_06.csv',
     'generate_csv/train_07.csv',
     'generate_csv/train_08.csv',
     'generate_csv/train_09.csv',
     'generate_csv/train_10.csv',
     'generate_csv/train_11.csv',
     'generate_csv/train_12.csv',
     'generate_csv/train_13.csv',
     'generate_csv/train_14.csv',
     'generate_csv/train_15.csv',
     'generate_csv/train_16.csv',
     'generate_csv/train_17.csv',
     'generate_csv/train_18.csv',
     'generate_csv/train_19.csv']
    valid filenames:
    ['generate_csv/valid_00.csv',
     'generate_csv/valid_01.csv',
     'generate_csv/valid_02.csv',
     'generate_csv/valid_03.csv',
     'generate_csv/valid_04.csv',
     'generate_csv/valid_05.csv',
     'generate_csv/valid_06.csv',
     'generate_csv/valid_07.csv',
     'generate_csv/valid_08.csv',
     'generate_csv/valid_09.csv']
    test filenames:
    ['generate_csv/test_00.csv',
     'generate_csv/test_01.csv',
     'generate_csv/test_02.csv',
     'generate_csv/test_03.csv',
     'generate_csv/test_04.csv',
     'generate_csv/test_05.csv',
     'generate_csv/test_06.csv',
     'generate_csv/test_07.csv',
     'generate_csv/test_08.csv',
     'generate_csv/test_09.csv']


### 3.2 加载csv的文件内的数据


```python
# 1. filename -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

def parse_csv_line(line, n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x, y

train_set = csv_reader_dataset(train_filenames, batch_size=3)
for x_batch, y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("y:")
    pprint.pprint(y_batch)
```

    x:
    <tf.Tensor: shape=(3, 8), dtype=float32, numpy=
    array([[-0.32652634,  0.4323619 , -0.09345459, -0.08402992,  0.8460036 ,
            -0.02663165, -0.56176794,  0.1422876 ],
           [ 0.48530516, -0.8492419 , -0.06530126, -0.02337966,  1.4974351 ,
            -0.07790658, -0.90236324,  0.78145146],
           [-1.0591781 ,  1.3935647 , -0.02633197, -0.1100676 , -0.6138199 ,
            -0.09695935,  0.3247131 , -0.03747724]], dtype=float32)>
    y:
    <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    array([[2.431],
           [2.956],
           [0.672]], dtype=float32)>
    x:
    <tf.Tensor: shape=(3, 8), dtype=float32, numpy=
    array([[ 8.0154431e-01,  2.7216142e-01, -1.1624393e-01, -2.0231152e-01,
            -5.4305160e-01, -2.1039616e-02, -5.8976209e-01, -8.2418457e-02],
           [ 4.9710345e-02, -8.4924191e-01, -6.2146995e-02,  1.7878747e-01,
            -8.0253541e-01,  5.0660671e-04,  6.4664572e-01, -1.1060793e+00],
           [ 2.2754266e+00, -1.2497431e+00,  1.0294788e+00, -1.7124432e-01,
            -4.5413753e-01,  1.0527152e-01, -9.0236324e-01,  9.0129471e-01]],
          dtype=float32)>
    y:
    <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    array([[3.226],
           [2.286],
           [3.798]], dtype=float32)>



```python
batch_size = 32
train_set = csv_reader_dataset(train_filenames,
                               batch_size = batch_size)
valid_set = csv_reader_dataset(valid_filenames,
                               batch_size = batch_size)
test_set = csv_reader_dataset(test_filenames,
                              batch_size = batch_size)
```

### 3.3 训练模型


```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(train_set,
                    validation_data = valid_set,
                    steps_per_epoch = 11160 // batch_size,
                    validation_steps = 3870 // batch_size,
                    epochs = 10,
                    callbacks = callbacks)
```

    Epoch 1/100
    348/348 [==============================] - 1s 3ms/step - loss: 1.5927 - val_loss: 2.1706
    Epoch 2/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.7043 - val_loss: 0.5049
    Epoch 3/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.4733 - val_loss: 0.4638
    Epoch 4/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.4384 - val_loss: 0.4345
    Epoch 5/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.4070 - val_loss: 0.4233
    Epoch 6/100
    348/348 [==============================] - 1s 4ms/step - loss: 0.4066 - val_loss: 0.4139
    Epoch 7/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.4051 - val_loss: 0.4155
    Epoch 8/100
    348/348 [==============================] - 1s 4ms/step - loss: 0.3824 - val_loss: 0.3957
    Epoch 9/100
    348/348 [==============================] - 1s 3ms/step - loss: 0.3956 - val_loss: 0.3884
    Epoch 10/100
    348/348 [==============================] - 1s 3ms/step - loss: 0.3814 - val_loss: 0.3856
    Epoch 11/100
    348/348 [==============================] - 1s 2ms/step - loss: 0.4826 - val_loss: 0.3887
    Epoch 12/100
    348/348 [==============================] - 1s 3ms/step - loss: 0.3653 - val_loss: 0.3853
    Epoch 13/100
    348/348 [==============================] - 1s 3ms/step - loss: 0.3765 - val_loss: 0.3810
    Epoch 14/100
    348/348 [==============================] - 1s 4ms/step - loss: 0.3632 - val_loss: 0.3775
    Epoch 15/100
    348/348 [==============================] - 1s 4ms/step - loss: 0.3654 - val_loss: 0.3758



```python
model.evaluate(test_set, steps = 5160 // batch_size)
```

    161/161 [==============================] - 1s 2ms/step - loss: 0.3811





    0.38114801049232483



# 4、加载TFRecorde文件

TFRecord格式是TensorFlow首选的格式，用于存储大量数据并有效读取数据。这是一种非常简单的二进制格式，只包含大小不同的二进制记录序列（每个记录由一个长度、一个用于检查长度是否损坏的CRC校验和、实际数据以及最后一个CRC校验和组成）。

如果TFRecord文件内只是简单的数据类型，那可以直接写和读取即可。但若数据是复杂的数据类型，则需要protobuf协助数据解释。下面我们分别介绍。

## 4.1 简单数据类型

对于简单的数据类型，可以使用tf.io.TFRecordWriter类写入数据，使用tf.data.TFRecordDataset读取一个或多个文件并返回dataset。


```python
with tf.io.TFRecordWriter('my_data.tfrecord') as f:
    f.write(b'my first record')
    f.write(b'my second record')
```

文件已经生成并写入数据，下面我们读取这个文件里面的数据：


```python
filepaths = ['my_data.tfrecord']
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
```

    tf.Tensor(b'my first record', shape=(), dtype=string)
    tf.Tensor(b'my second record', shape=(), dtype=string)


默认情况下，TFRecordDataset将一个接一个地读取文件，但是你可以通过设置num_parallel_reads使其并行读取多个文件并交织记录。另外，你可以使用list_files（）和interleave（）得到与前面读取多个CSV文件相同的结果。


我们还可以对tfrecod文件进行压缩：


```python
option = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter('my_data2.tfrecord', option) as f:
    f.write(b'my first record')
    f.write(b'my second record')
```

## 4.2 复杂数据格式

对于复杂的数据格式，你需要通过protobuf来说明各个字段的含义。即使每个记录可以使用你想要的任何二进制格式，TFRecord文件通常包含序列化的协议缓冲区（也称为protobufs）。这是一种可移植、可扩展且高效的二进制格式，在2001年由Google开发，并于2008年开源。

一般情况下，你需要定义proto文件并编译，然后才能使用。但TensorFlow包含特殊的protobuf定义，并为其提供了解析操作。

TFRecord文件中通常使用的主要protobuf是Exampleprotobuf，它表示数据集中的一个实例。它包含一个已命名特征的列表，其中每个特征可以是字节字符串列表、浮点数列表或整数列表。以下是protobuf的定义(可能已经发送改变，请查阅最新的API)：

![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/截屏2021-08-1217.07.25.png)
BytesList、FloatList和Int64List的定义非常简单直接。请注意，[packed=true]用于重复的数字字段以实现更有效的编码。一个Feature包含BytesList或FloatList或Int64List。一个Features（带有s）包含将特征名称映射到相应特征值的字典。最后，一个Example仅包含Features对象[3]。以下是创建tf.train.Example的方法，该示例表示与先前相同的person并将其写入TFRecord文件：



```python
BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))


```

该代码有点冗长和重复，但相当简单（你可以轻松地将其包装在一个小的辅助函数中）。既然有了Exampleprotobuf，我们可以通过调用其SerializeToString（）方法对其进行序列化，然后将结果数据写入TFRecord文件：


```python
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())
```

通常你会编写不止一个Example！你会创建一个转换脚本，该脚本从你当前的格式（例如CSV文件）中读取，为每个实例创建一个Exampleprotobuf，对其进行序列化，然后将其保存到多个TFRecord文件中，最好在处理过程中对其进行乱序。这需要一些工作量，因此请再次确保确实有必要（也许你的流水线可以使用CSV文件正常工作）。现在我们有了一个不错的TFRecord文件，其中包含序列化的Example，让我们尝试加载它。

要加载序列化的Exampleprotobufs，我们再次使用tf.data.TFRecordDataset，并使用tf.io.parse_single_example（）解析每个Example。这是一个TensorFlow操作，因此可以包含在TF函数中。它至少需要两个参数：一个包含序列化数据的字符串标量张量，以及每个特征的描述。这个描述是一个字典，将每个特征名称映射到表示特征形状、类型和默认值的tf.io.FixedLenFeature描述符，或者仅表示类型的tf.io.VarLenFeature描述符（如果特征列表的长度有所不同，例如"emails"特征）。

以下代码定义了一个描述字典，然后遍历TFRecordDataset并解析该数据集包含的序列化的Exampleprotobuf：



```python
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)

parsed_example
```




    {'emails': <tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7f95700a0550>,
     'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>,
     'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}



## 4.3 使用tfrecord表示图片

BytesList可以包含你想要的任何二进制数据，包括任何序列化的对象。例如，你可以使用tf.io.encode_jpeg（）对JPEG格式的图像进行编码，然后将此二进制数据放入BytesList。稍后，当你的代码读取TFRecord时，它将从解析Example开始，然后它需要调用tf.io.decode_jpeg（）来解析数据并获取原始图像（或者你可以使用tf.io.decode_image（）来解码任何BMP、GIF、JPEG或PNG图像）。你还可以通过使用tf.io.serialize_tensor（）来序列化张量，并存储在BytesList中，然后将生成的字节字符串放入BytesList特征中。稍后当你解析TFRecord时，可以使用tf.io.parse_tensor（）解析此数据。

与其使用tf.io.parse_single_example（）一个接一个地解析用例，不如使用tf.io.parse_example（）一个批次接一个批次地解析它们：




```python
from sklearn.datasets import load_sample_images

img = load_sample_images()["images"][0]
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()
```


    
![png](tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_files/tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_39_0.png)
    



```python
data = tf.io.encode_jpeg(img)
example_with_image = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))
serialized_example = example_with_image.SerializeToString()
# then save to TFRecord
```


```python
feature_description = { "image": tf.io.VarLenFeature(tf.string) }
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_jpeg(example_with_image["image"].values[0])
```


```python
decoded_img = tf.io.decode_image(example_with_image["image"].values[0])
```


```python
plt.imshow(decoded_img)
plt.title("Decoded Image")
plt.axis("off")
plt.show()
```


    
![png](tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_files/tensorflow%E7%B3%BB%E5%88%97%E4%B9%8B1%EF%BC%9A%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE_43_0.png)
    


## 4.4 将张量和稀疏张量写入TFRecord文件


```python
t = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
s = tf.io.serialize_tensor(t)
s
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'\x08\x01\x12\x08\x12\x02\x08\x03\x12\x02\x08\x02"\x18\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x00\x00\xa0@'>




```python
tf.io.parse_tensor(s, out_type=tf.float32)
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[0., 1.],
           [2., 3.],
           [4., 5.]], dtype=float32)>




```python
serialized_sparse = tf.io.serialize_sparse(parsed_example["emails"])
serialized_sparse
```




    <tf.Tensor: shape=(3,), dtype=string, numpy=
    array([b'\x08\t\x12\x08\x12\x02\x08\x02\x12\x02\x08\x01"\x10\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00',
           b'\x08\x07\x12\x04\x12\x02\x08\x02"\x10\x07\x07a@b.comc@d.com',
           b'\x08\t\x12\x04\x12\x02\x08\x01"\x08\x02\x00\x00\x00\x00\x00\x00\x00'],
          dtype=object)>




```python
BytesList(value=serialized_sparse.numpy())
```




    value: "\010\t\022\010\022\002\010\002\022\002\010\001\"\020\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000"
    value: "\010\007\022\004\022\002\010\002\"\020\007\007a@b.comc@d.com"
    value: "\010\t\022\004\022\002\010\001\"\010\002\000\000\000\000\000\000\000"




```python
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(10)
for serialized_examples in dataset:
    parsed_examples = tf.io.parse_example(serialized_examples,
                                          feature_description)

parsed_examples
```




    {'image': <tensorflow.python.framework.sparse_tensor.SparseTensor at 0x7f9570094190>}




```python

```
