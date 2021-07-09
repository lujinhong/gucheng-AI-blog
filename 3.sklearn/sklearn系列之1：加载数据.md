本文介绍了如何加载各种数据源，以生成可以用于sklearn使用的数据集。主要包括以下几类数据源：
* 预定义的公共数据源
* 内存中的数据
* csv文件
* 任意格式的数据文件
* 稀疏数据格式文件

sklearn使用的数据集一般为numpy ndarray，或者pandas dataframe。


```python
import numpy as np
import pandas as pd
import sklearn
import os
import urllib
import tarfile
```

## 1、预定义的公共数据源

更多数据集请见：https://scikitlearn.com.cn/0.21.3/47/

### minst数据集
以下示例用于判断图片是否数字5


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X,y = pd.DataFrame.to_numpy(mnist['data']), pd.DataFrame.to_numpy(mnist['target'])

X_train, X_test = X[:6000], X[6000:]
y_train, y_test = y[:6000].astype(np.uint8), y[6000:].astype(np.uint8)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='hinge')
model.fit(X_train, y_train_5)
print(model.predict([X[0]]))
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~/opt/anaconda3/envs/tf/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3079             try:
    -> 3080                 return self._engine.get_loc(casted_key)
       3081             except KeyError as err:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 0

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-2-af298a763d15> in <module>
         11 model = SGDClassifier(loss='hinge')
         12 model.fit(X_train, y_train_5)
    ---> 13 print(model.predict([X[0]]))
    

    ~/opt/anaconda3/envs/tf/lib/python3.8/site-packages/pandas/core/frame.py in __getitem__(self, key)
       3022             if self.columns.nlevels > 1:
       3023                 return self._getitem_multilevel(key)
    -> 3024             indexer = self.columns.get_loc(key)
       3025             if is_integer(indexer):
       3026                 indexer = [indexer]


    ~/opt/anaconda3/envs/tf/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3080                 return self._engine.get_loc(casted_key)
       3081             except KeyError as err:
    -> 3082                 raise KeyError(key) from err
       3083 
       3084         if tolerance is not None:


    KeyError: 0


## iris数据集
这是一个非常著名的数据集，共有150朵鸢尾花，分别来自三个不同品种（山鸢尾、变色鸢尾和维吉尼亚鸢尾），数据里包含花的萼片以及花瓣的长度和宽度。


```python
from sklearn import datasets
iris = datasets.load_iris()
```

我们看一下数据集。注意，sklearn的dataset都包含这些keys：


```python
print(iris.keys())
print(iris['data'][:10], iris['target'][:], iris['frame'], iris['target_names'][:10],
      iris['DESCR'], iris['feature_names'][:10])
```


```python

```


```python

```


```python

```

## 2、内存中的数据

本示例，我们在内存中生成numpy ndarray，然后使用线性回归拟合数据。


```python
X = 2 * np.random.rand(100,1)
y = 3 * X + 4 + np.random.rand(100,1)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model.intercept_, model.coef_)
```

我们也可以使用pandas dataframe作为模型的输入。


```python
X = pd.DataFrame(2 * np.random.rand(100,1))
y = pd.DataFrame(3 * X + 4 + np.random.rand(100,1))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model.intercept_, model.coef_)
```

下面使用csv文件中的数据时，大部分情况也是转化为pandas.DataFrame。

## 3、csv文件中的数据
我们用housing数据做示例，使用线性回归拟合一个地区的房价中位数。
由于我们没有数据文件，先下载下来：


```python
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_file = os.path.join(housing_path,'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_file)
    housing_tgz = tarfile.open(tgz_file)
    housing_tgz.extractall(path = housing_path) #解压文件
    housing_tgz.close()
    
fetch_housing_data()
```

csv文件准备好了以后，我们使用pandas.read_csv()加载文件中的内容：


```python
housing = pd.read_csv(os.path.join(HOUSING_PATH,'housing.csv'))
# 简单看几行数据
housing.head()
```

由于housing中有缺失值，所以我们需要先填充数据。看一下缺失值的情况：


```python
housing.info()
```

我们看到total_bedromms中有缺失值，我们使用均值来做填充。如果有很多字段都有缺失值，可以使用sklearn的Simpleimputer批量处理，详见sklearn系列：数据预处理。


```python
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)
```

下面，我们分离label和feature。同时，先暂时忽略ocean_proximity这个非数值特征：


```python
housing_label = housing['median_house_value']
housing_feature = housing.drop(['median_house_value','ocean_proximity'], axis=1)
```


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_feature,housing_label)
print(model.intercept_, model.coef_)
```

### 完整代码


```python
housing = pd.read_csv(os.path.join(HOUSING_PATH,'housing.csv'))

median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)

housing_label = housing['median_house_value']
housing_feature = housing.drop(['median_house_value','ocean_proximity'], axis=1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_feature,housing_label)
print(model.intercept_, model.coef_)
```


```python

```
