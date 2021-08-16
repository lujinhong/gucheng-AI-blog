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

    [ True]


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

    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]
     [5.4 3.9 1.7 0.4]
     [4.6 3.4 1.4 0.3]
     [5.  3.4 1.5 0.2]
     [4.4 2.9 1.4 0.2]
     [4.9 3.1 1.5 0.1]] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2] None ['setosa' 'versicolor' 'virginica'] .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ... ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


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

    [4.45291269] [[2.99295562]]


我们也可以使用pandas dataframe作为模型的输入。


```python
X = pd.DataFrame(2 * np.random.rand(100,1))
y = pd.DataFrame(3 * X + 4 + np.random.rand(100,1))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model.intercept_, model.coef_)
```

    [4.45003988] [[3.02825472]]


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



由于housing中有缺失值，所以我们需要先填充数据。看一下缺失值的情况：


```python
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
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB


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

    -3570118.06149459 [-4.26104026e+04 -4.24754782e+04  1.14445085e+03 -6.62091740e+00
      8.11609666e+01 -3.98732002e+01  7.93047225e+01  3.97522237e+04]


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

    -3570118.06149459 [-4.26104026e+04 -4.24754782e+04  1.14445085e+03 -6.62091740e+00
      8.11609666e+01 -3.98732002e+01  7.93047225e+01  3.97522237e+04]



```python

```
