```python
本文介绍了加载数据后，对数据需要做的一些预处理，包括乱序、数据拆分、缺失值、onehot、embedding、特征缩放、批量归一化等等。
```


```python
import numpy as np
import pandas as pd
import sklearn
import urllib
import os
import tarfile
```

## 1、数据乱序 

我们分别介绍numpy.ndarray和pandas.dataframe的乱序。

### 1.1 numpy.ndarray

拆分前，一般会先对数据进行随机排序。

numpy.random中有shuffle()和permutation()2个函数均可用于对数据进行乱序。主要区别在于：
* shuffle()直接对原数据进行重排，无返回值。
* permutation()复制原数据，然后再重排，返回重排后的数组。原数据没有任何变化。

生成数据：


```python
data = np.arange(100).reshape(10,-1)
print(data)
```

    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]
     [30 31 32 33 34 35 36 37 38 39]
     [40 41 42 43 44 45 46 47 48 49]
     [50 51 52 53 54 55 56 57 58 59]
     [60 61 62 63 64 65 66 67 68 69]
     [70 71 72 73 74 75 76 77 78 79]
     [80 81 82 83 84 85 86 87 88 89]
     [90 91 92 93 94 95 96 97 98 99]]


使用permutation()重排：


```python
x = np.random.permutation(data)
print(x)
print(data)
```

    [[50 51 52 53 54 55 56 57 58 59]
     [70 71 72 73 74 75 76 77 78 79]
     [40 41 42 43 44 45 46 47 48 49]
     [90 91 92 93 94 95 96 97 98 99]
     [20 21 22 23 24 25 26 27 28 29]
     [10 11 12 13 14 15 16 17 18 19]
     [30 31 32 33 34 35 36 37 38 39]
     [80 81 82 83 84 85 86 87 88 89]
     [60 61 62 63 64 65 66 67 68 69]
     [ 0  1  2  3  4  5  6  7  8  9]]
    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]
     [30 31 32 33 34 35 36 37 38 39]
     [40 41 42 43 44 45 46 47 48 49]
     [50 51 52 53 54 55 56 57 58 59]
     [60 61 62 63 64 65 66 67 68 69]
     [70 71 72 73 74 75 76 77 78 79]
     [80 81 82 83 84 85 86 87 88 89]
     [90 91 92 93 94 95 96 97 98 99]]


使用shuffle()重排：


```python
np.random.shuffle(data)
print(data)
```

    [[40 41 42 43 44 45 46 47 48 49]
     [20 21 22 23 24 25 26 27 28 29]
     [50 51 52 53 54 55 56 57 58 59]
     [80 81 82 83 84 85 86 87 88 89]
     [30 31 32 33 34 35 36 37 38 39]
     [ 0  1  2  3  4  5  6  7  8  9]
     [60 61 62 63 64 65 66 67 68 69]
     [10 11 12 13 14 15 16 17 18 19]
     [90 91 92 93 94 95 96 97 98 99]
     [70 71 72 73 74 75 76 77 78 79]]


### 1.2 pandas.dataframe

对datafame进行乱序，只需要使用sample()即可。

我们使用iris数据集生成datafame:


```python
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame()
df['heigh'] = iris['data'][:,0]
df['length'] = iris['data'][:,1]
df['label'] = iris['target']

print(df.head())
```

       heigh  length  label
    0    5.1     3.5      0
    1    4.9     3.0      0
    2    4.7     3.2      0
    3    4.6     3.1      0
    4    5.0     3.6      0


#### sample()方式

我们使用sample对df进行shuffle。我们可以看到df自身是没有变化的：


```python
df_shuffle = df.sample(frac=1)
print(df_shuffle.head())
print(df.head())
```

         heigh  length  label
    40     5.0     3.5      0
    80     5.5     2.4      1
    55     5.7     2.8      1
    96     5.7     2.9      1
    108    6.7     2.5      2
       heigh  length  label
    0    5.1     3.5      0
    1    4.9     3.0      0
    2    4.7     3.2      0
    3    4.6     3.1      0
    4    5.0     3.6      0


参数frac是要返回的比例。如果需要打混后数据集的index（索引）还是按照正常的排序：


```python
df_shuffle2 = df.sample(frac=1).reset_index(drop=True)
print(df_shuffle2.head())
```

       heigh  length  label
    0    7.7     2.6      2
    1    6.5     3.2      2
    2    5.6     2.8      2
    3    4.6     3.6      0
    4    7.4     2.8      2


#### sklearn的方式

sklearn.utils.shuffle()也可以对datafame乱序：


```python
df_shuffle3 = sklearn.utils.shuffle(df)
print(df_shuffle3.head())
```

         heigh  length  label
    88     5.6     3.0      1
    17     5.1     3.5      0
    7      5.0     3.4      0
    132    6.4     2.8      2
    67     5.8     2.7      1


#### numpy的方式
不推荐此方式


```python
df_shuffle4 = df.iloc[np.random.permutation(len(df))]
print(df_shuffle4.head())
```

         heigh  length  label
    130    7.4     2.8      2
    144    6.7     3.3      2
    110    6.5     3.2      2
    99     5.7     2.8      1
    143    6.8     3.2      2


## 2、数据拆分

### 2.1 基本拆分
我们使用自定义函数的方式，随机抽取20%的样本作为测试集：



```python

#如果是pandas数据
def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)                                                     
    training_idx, test_idx = shuffle_indices[test_size:], shuffle_indices[:test_size]
    return data.iloc[training_idx], data.iloc[test_idx]

trining_data, test_data = split_train_test(pd.DataFrame(x), 0.2)
print(test_data)
```

        0   1   2   3   4   5   6   7   8   9
    9   0   1   2   3   4   5   6   7   8   9
    2  40  41  42  43  44  45  46  47  48  49



```python
#如果是numpy数据,建议使用pd.DataFrame()先转换为pandas数据，也可以使用以下方式：
def split_train_test_np(data, test_ratio):
    shuffle_indeices = np.random.permutation(data.shape[0])
    test_size = int(data.shape[0] * test_ratio)                                                     
    training_idx, test_idx = shuffle_indeices[test_size:], shuffle_indeices[:test_size]
    return data[training_idx], data[test_idx]

trining_data, test_data = split_train_test_np(x, 0.2)
print(test_data)
```

    [[10 11 12 13 14 15 16 17 18 19]
     [70 71 72 73 74 75 76 77 78 79]]


### 2.2 固定样本

运行上面的代码会发现，每次运行时得到的样本都不同，我们可以增加一个随机种子，使得每次随机结果都相同。


```python
np.random.seed(42)

trining_data, test_data = split_train_test(pd.DataFrame(x), 0.2)
print(test_data)
```

        0   1   2   3   4   5   6   7   8   9
    8  60  61  62  63  64  65  66  67  68  69
    1  70  71  72  73  74  75  76  77  78  79


### 2.3 样本集更新导致的测试集变化
上述虽然解决了每次运行得到不同随机结果的问题，但如果由于样本增加或者减少时，一个样本有可能会被重新划分到另一个数据集。

解决这个问题的思路是：计算每个实例标识符的hash值，如果hash小于最大值的20%，则将实例放入测试集：


```python
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# 由于housing数据没有index，所以我们使用行索引作为ID：
x_pd = pd.DataFrame(x)
x_with_id = x_pd.reset_index()
split_train_test_by_id(x_with_id, 0.2, 'index')
```




    (   index   0   1   2   3   4   5   6   7   8   9
     0      0  50  51  52  53  54  55  56  57  58  59
     1      1  70  71  72  73  74  75  76  77  78  79
     3      3  90  91  92  93  94  95  96  97  98  99
     4      4  20  21  22  23  24  25  26  27  28  29
     6      6  30  31  32  33  34  35  36  37  38  39
     7      7  80  81  82  83  84  85  86  87  88  89
     8      8  60  61  62  63  64  65  66  67  68  69
     9      9   0   1   2   3   4   5   6   7   8   9,
        index   0   1   2   3   4   5   6   7   8   9
     2      2  40  41  42  43  44  45  46  47  48  49
     5      5  10  11  12  13  14  15  16  17  18  19)



### 2.4 使用sklearn的方式

其实sklearn也提供了一个函数用于同样的功能：


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(x_pd, test_size=0.2, random_state=42)
test_set.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>60</td>
      <td>61</td>
      <td>62</td>
      <td>63</td>
      <td>64</td>
      <td>65</td>
      <td>66</td>
      <td>67</td>
      <td>68</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70</td>
      <td>71</td>
      <td>72</td>
      <td>73</td>
      <td>74</td>
      <td>75</td>
      <td>76</td>
      <td>77</td>
      <td>78</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```

sklearn提供了丰富、方便的数据预处理功能，本文介绍常用的一些功能。

本文使用housing数据作为示例：


```python


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
HOUSING_PATH = 'datasets/housing'
#HOUSING_PATH = os.path.join("datasets","housing")

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path=os.path.join(housing_path, "housing.tgz")
    print(tgz_path)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)
    
fetch_housing_data()
housing = load_data()
housing.info()
housing.head()

```

    datasets/housing/housing.tgz
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB





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



## 3、缺失值处理
housing数据集中的total_bedrooms有部分缺失，对于缺失值，通常我们有以下几种方式处理：
* 放弃有缺失值的样本
* 放弃整个特征
* 将缺失值设置为某个默认值：0、平均值、中位数等。


### 1.1 pandas方式
通过DataFrame的dropna(), drop()和fillna()函数，可以方便的实现以上3个功能：


```python
housing.dropna(subset=['total_bedrooms'])
housing.info()

housing.drop('total_bedrooms', axis = 1)
housing.info()

median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)
housing.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20640 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB


### 1.2 sklearn方式
使用pandas方式需要对每个属性进行处理，我们使用sklearn来批量处理整个数据集的所有属性。

sklearn提供了一个非常容易上手的类来处理缺失值：Simpleimputer。同时，由于中位数只能在数值类属性上计算，所以我们需要创建一个没有文本属性ocean_proximity的数据副本：


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
housing_num = housing.drop('ocean_proximity', axis=1)

# fit、transform，然后转换回DataFrame
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_num_pd = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

我们可以看一下各个属性的中位数：


```python
print(imputer.statistics_)
print(housing_num.median().values)
```

    [-1.1849e+02  3.4260e+01  2.9000e+01  2.1270e+03  4.3500e+02  1.1660e+03
      4.0900e+02  3.5348e+00  1.7970e+05]
    [-1.1849e+02  3.4260e+01  2.9000e+01  2.1270e+03  4.3500e+02  1.1660e+03
      4.0900e+02  3.5348e+00  1.7970e+05]


## 4、处理文本和分类属性

### 2.1 类别转换成数字
我们看一下文本属性。在此数据集中，只有一个：ocean_proximity属性：


```python
housing_cat = housing[['ocean_proximity']]
sklearn.utils.shuffle(housing_cat).head()
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10763</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>322</th>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>11124</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>8115</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2630</th>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



它不是任意文本，而是有限个可能的取值，每个值代表一个类别。因此，此属性是分类属性。大多数机器学习算法更喜欢使用数字，因此让我们将这些类别从文本转到数字:


```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
np.random.permutation(housing_cat_encoded)[:5]
```




    array([[4.],
           [1.],
           [0.],
           [4.],
           [4.]])



我们看一下每个数字的含义：



```python
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



## 2.2 类别one-hot 
上述将类型转化为数字的方式产生的一个问题是，机器学习算法会认为两个相近的值比两个离得较远的值更为相似一些。在某些情况下这是对的（对一些有序类别，像“坏”“平均”“好”“优秀”），但是，对ocean_proximity而言情况并非如此（例如，类别0和类别4之间就比类别0和类别1之间的相似度更高）。为了解决这个问题，常见的解决方案是给每个类别创建一个二进制的属性：当类别是“<1HOCEAN”时，一个属性为1（其他为0），当类别是“INLAND”时，另一个属性为1（其他为0），以此类推。这就是独热编码，因为只有一个属性为1（热），其他均为0（冷）。新的属性有时候称为哑（dummy）属性。ScikitLearn提供了一个OneHotEncoder编码器，可以将整数类别值转换为独热向量。我们用它来将类别编码为独热向量




```python
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder()
housing_1hot = oh_encoder.fit_transform(housing_cat)
housing_1hot
```




    <20640x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 20640 stored elements in Compressed Sparse Row format>



注意到这里的输出是一个SciPy稀疏矩阵，而不是一个NumPy数组。当你有成千上万个类别属性时，这个函数会非常有用。因为在独热编码完成之后，我们会得到一个几千列的矩阵，并且全是0，每行仅有一个1。占用大量内存来存储0是一件非常浪费的事情，因此稀疏矩阵选择仅存储非零元素的位置。而你依旧可以像使用一个普通的二维数组那样来使用他，当然如果你实在想把它转换成一个（密集的）NumPy数组，只需要调用toarray（）方法即可：


```python
housing_1hot.toarray()
```




    array([[0., 0., 0., 1., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 1., 0.],
           ...,
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])



## 5、数值分箱与one-hot

本部分仅介绍数值类特征的one-hot，关于文本的one-hot请参考上一部分。

数值one-hot可以使用pandas.cut()和get_dummies()或者sklearn.OnehotEncoder。
此外，skearn的preprocessing.KBinsDiscretizer类和Binarizer类也可以用于数值分箱。

### 5.1 pandas方式
基本思路是先使用cut()对数值进行分箱，分箱后使用get_dummies()得到onehot值。API:

https://pandas.pydata.org/docs/reference/api/pandas.cut.html?highlight=cut#pandas.cut

https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

我们先对数据进行分箱：

**我们这里使用的是指定分隔值的方式，还可以简单的指定平均分成N个等分等，详见cut()的API。**


```python
lst = np.arange(0,100, 3)
print(lst)
```

    [ 0  3  6  9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69
     72 75 78 81 84 87 90 93 96 99]



```python
lst_bins = pd.cut(lst, [-1,10,50,100])
print(lst_bins)
```

    [(-1, 10], (-1, 10], (-1, 10], (-1, 10], (10, 50], ..., (50, 100], (50, 100], (50, 100], (50, 100], (50, 100]]
    Length: 34
    Categories (3, interval[int64]): [(-1, 10] < (10, 50] < (50, 100]]


我们看一下每个区间的数量：


```python
print(pd.value_counts(lst_bins))
```

    (50, 100]    17
    (10, 50]     13
    (-1, 10]      4
    dtype: int64


但这样分箱后不是很适合阅读，所以我们可以加上标签：


```python
lst_bins = pd.cut(lst, [-1,10,50,100], labels=['1','2','3'])
print(lst_bins)
```

    ['1', '1', '1', '1', '2', ..., '3', '3', '3', '3', '3']
    Length: 34
    Categories (3, object): ['1' < '2' < '3']


我们还可以简单的将数据分箱成N份：


```python
lst_bins2 = pd.cut(lst, 5, labels=['1','2','3','4','5'])
print(lst_bins2)
print(pd.value_counts(lst_bins2))
```

    ['1', '1', '1', '1', '1', ..., '5', '5', '5', '5', '5']
    Length: 34
    Categories (5, object): ['1' < '2' < '3' < '4' < '5']
    5    7
    4    7
    2    7
    1    7
    3    6
    dtype: int64


得到分箱值后，我们就可以对分箱进行one-hot了。get_dummies处理的是DataFrame，所以我们先把数据包装成DataFame。


```python
df = pd.DataFrame()
df['score'] = lst_bins
print(df)

df_onehot = pd.get_dummies(df['score'])
print(df_onehot)
```

       score
    0      1
    1      1
    2      1
    3      1
    4      2
    5      2
    6      2
    7      2
    8      2
    9      2
    10     2
    11     2
    12     2
    13     2
    14     2
    15     2
    16     2
    17     3
    18     3
    19     3
    20     3
    21     3
    22     3
    23     3
    24     3
    25     3
    26     3
    27     3
    28     3
    29     3
    30     3
    31     3
    32     3
    33     3
        1  2  3
    0   1  0  0
    1   1  0  0
    2   1  0  0
    3   1  0  0
    4   0  1  0
    5   0  1  0
    6   0  1  0
    7   0  1  0
    8   0  1  0
    9   0  1  0
    10  0  1  0
    11  0  1  0
    12  0  1  0
    13  0  1  0
    14  0  1  0
    15  0  1  0
    16  0  1  0
    17  0  0  1
    18  0  0  1
    19  0  0  1
    20  0  0  1
    21  0  0  1
    22  0  0  1
    23  0  0  1
    24  0  0  1
    25  0  0  1
    26  0  0  1
    27  0  0  1
    28  0  0  1
    29  0  0  1
    30  0  0  1
    31  0  0  1
    32  0  0  1
    33  0  0  1


完整代码：


```python
lst = np.arange(0,100, 3)
lst_bins = pd.cut(lst, [-1,10,50,100])
lst_bins = pd.cut(lst, [-1,10,50,100], labels=['1','2','3'])

df = pd.DataFrame()
df['score'] = lst_bins
df_onehot = pd.get_dummies(df['score'])
print(df_onehot)
```

        1  2  3
    0   1  0  0
    1   1  0  0
    2   1  0  0
    3   1  0  0
    4   0  1  0
    5   0  1  0
    6   0  1  0
    7   0  1  0
    8   0  1  0
    9   0  1  0
    10  0  1  0
    11  0  1  0
    12  0  1  0
    13  0  1  0
    14  0  1  0
    15  0  1  0
    16  0  1  0
    17  0  0  1
    18  0  0  1
    19  0  0  1
    20  0  0  1
    21  0  0  1
    22  0  0  1
    23  0  0  1
    24  0  0  1
    25  0  0  1
    26  0  0  1
    27  0  0  1
    28  0  0  1
    29  0  0  1
    30  0  0  1
    31  0  0  1
    32  0  0  1
    33  0  0  1


如果df中有多个字段：(如需先分箱，则参考上面）


```python
        
df = pd.DataFrame({
          'A':['a','b','a'],
          'B':['b','a','c']
        })

# Get one hot encoding of columns B
one_hot = pd.get_dummies(df['B'])
# Drop column B as it is now encoded
df = df.drop('B',axis = 1)
# Join the encoded df
df = df.join(one_hot)
print(df)  

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
      <th>A</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 将多个标签onehot
使用get_dummies可以直接将所有的feature做onehot：


```python

pd.Series(['a|b', 'a', 'a|c']).str.get_dummies()

df = pd.DataFrame({
            'f':['a,b', 'a', 'a,c']
            })
df['f'].str.get_dummies(",")
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



如果数据量比较大，可以使用MultiLabelBinarizer

https://stackoverflow.com/questions/63544536/convert-pd-get-dummies-result-to-df-str-get-dummies


```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

output = pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df['f'].str.split(',')),
                                          columns=mlb.classes_)
print(output)
```

       a  b  c
    0  1  1  0
    1  1  0  0
    2  1  0  1


看一个完整的例子，我们将以下数据做onehot，如果有这个标签则为0，没有则为1：
label,features
1,80801|898509
0,80801|898509|59834
1,80801|898509|48983


```python
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

sample_dir = '/home/ljhn1829/jupyter/ljh/data/onehot_sample.csv'
df_sample_all = pd.read_csv(sample_dir)

mlb = MultiLabelBinarizer(sparse_output=True)
output = pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df_sample_all['features'].str.split('|')),
                                           columns=mlb.classes_)
df_sample_onehot_all = pd.DataFrame()
df_sample_onehot_all['label'] = df_sample_all['label']
print(df_sample_onehot_all)

df_sample_onehot_all= pd.concat([df_sample_onehot_all,output], axis=1)
print(df_sample_onehot_all)
```

       label
    0      1
    1      0
    2      1
       label  48983  59834  80801  898509
    0      1      0      0      1       1
    1      0      0      1      1       1
    2      1      1      0      1       1



```python

```


```python

```

### sklearn 方式

对于分类数值的onehot，其处理方式和上述的文本类别的处理方式并无不同。

如果是连续数值onehot，则需要使用上述的cut()或者skearn的preprocessing.KBinsDiscretizer类和Binarizer类先进行分箱。一般使用cut()即可。


```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
# 类别的数量：
print(enc.categories_)
#onehot编码
print(enc.transform([[0, 1, 1]]).toarray())
```

    [array([0, 1]), array([0, 1, 2]), array([0, 1, 2, 3])]
    [[1. 0. 0. 1. 0. 0. 1. 0. 0.]]



```python

```


```python

```


```python

```


```python

```

## 5、特征缩放
最重要也最需要应用到数据上的转换就是特征缩放。如果输入的数值属性具有非常大的比例差异，往往会导致机器学习算法的性能表现不佳，当然也有极少数特例。案例中的房屋数据就是这样：房间总数的范围从6～39320，而收入中位数的范围是0～15。注意，目标值通常不需要缩放。

同比例缩放所有属性的两种常用方法是最小最大缩放和标准化。

最小最大缩放（又叫作归一化）很简单：将值重新缩放使其最终范围归于0～1之间。实现方法是将值减去最小值并除以最大值和最小值的差。对此，ScikitLearn提供了一个名为MinMaxScaler的转换器。如果出于某种原因，你希望范围不是0～1，那么可以通过调整超参数feature_range进行更改。

标准化则完全不一样：首先减去平均值（所以标准化值的均值总是零），然后除以方差，从而使得结果的分布具备单位方差。不同于最小最大缩放的是，标准化不将值绑定到特定范围，对某些算法而言，这可能是个问题（例如，神经网络期望的输入值范围通常是0～1）。但是标准化的方法受异常值的影响更小。例如，假设某个地区的平均收入为100（错误数据），最小最大缩放会将所有其他值从0～15降到0～0.15，而标准化则不会受到很大影响。ScikitLearn提供了一个标准化的转换器StandadScaler。

详细示例请见下一部分的转化流水线。

## 9、转化流水线

### 4.1 多个预处理步骤

正如你所见，许多数据转换的步骤需要以正确的顺序来执行。而ScikitLearn正好提供了Pipeline类来支持这样的转换。


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
```

Pipeline构造函数会通过一系列名称/估算器的配对来定义步骤序列。除了最后一个是估算器之外，前面都必须是转换器（也就是说，必须有fit_transform（）方法）。至于命名可以随意，你喜欢就好（只要它们是独一无二的，不含双下划线），它们稍后在超参数调整中会有用。

当调用流水线的fit（）方法时，会在所有转换器上按照顺序依次调用fit_transform（），将一个调用的输出作为参数传递给下一个调用方法，直到传递到最终的估算器，则只会调用fit（）方法。

流水线的方法与最终的估算器的方法相同。在本例中，最后一个估算器是StandardScaler，这是一个转换器，因此流水线有一个transform（）方法，可以按顺序将所有的转换应用到数据中（这也是我们用过的fit_transform（）方法）。


### 4.2 同时处理数字和分类属性
到目前为止，我们分别处理了类别列和数值列。拥有一个能够处理所有列的转换器会更方便，将适当的转换应用于每个列。在0.20版中，ScikitLearn为此引入了ColumnTransformer，好消息是它与pandasDataFrames一起使用时效果很好。让我们用它来将所有转换应用到房屋数据




```python
from sklearn.compose import ColumnTransformer
num_attribs=list(housing_num)

cat_attribs=["ocean_proximity"]
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])

housing_prepared=full_pipeline.fit_transform(housing)


```

首先导入ColumnTransformer类，接下来获得数值列名称列表和类别列名称列表，然后构造一个ColumnTransformer。构造函数需要一个元组列表，其中每个元组都包含一个名字、一个转换器，以及一个该转换器能够应用的列名字（或索引）的列表。在此示例中，我们指定数值列使用之前定义的num_pipeline进行转换，类别列使用OneHotEncoder进行转换。最后，我们将ColumnTransformer应用到房屋数据：它将每个转换器应用于适当的列，并沿第二个轴合并输出（转换器必须返回相同数量的行）。

请注意，OneHotEncoder返回一个稀疏矩阵，而num_pipeline返回一个密集矩阵。当稀疏矩阵和密集矩阵混合在一起时，ColumnTransformer会估算最终矩阵的密度（即单元格的非零比率），如果密度低于给定的阈值，则返回一个稀疏矩阵（通过默认值为sparse_threshold=0.3）。在此示例中，它返回一个密集矩阵。我们有一个预处理流水线，该流水线可以获取全部房屋数据并对每一列进行适当的转换。




```python

```
