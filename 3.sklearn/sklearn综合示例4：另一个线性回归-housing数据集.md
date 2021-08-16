本文参考自《机器学习实战》chpt 2。展示了使用sklearn完成房价预测项目的全流程，主要目地是熟悉sklearn的基本用法以及常用功能。

本文侧重模型训练、性能评估、模型调整，关于数据拆分及预处理，请见其它笔记。


```python
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import tarfile
import urllib

print(sys.version_info,sklearn.__version__,mpl.__version__,np.__version__)
```

    sys.version_info(major=3, minor=8, micro=5, releaselevel='final', serial=0) 0.23.2 3.3.2 1.19.2



```python
# 图像的格式
%matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# 保存图片
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
```

## 1、数据准备

### 1.1 下载数据


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

### 1.2 载入并查看数据
使用pandas读取并简单分析数据。


```python
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

```

我们看看数据。其中
* head()用于查看前几行数据
* info()用于查看行数、非空数据量、数据类型等
* describe()用于分析每个字段的均值、均方差、最大最小值、各个分布的数量等。


```python
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



```python
housing.describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



我们再用图形的方式看看数据的分布：


```python
%matplotlib inline
housing.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()
```


    
![png](sklearn%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B4%EF%BC%9A%E5%8F%A6%E4%B8%80%E4%B8%AA%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92-housing%E6%95%B0%E6%8D%AE%E9%9B%86_files/sklearn%E7%BB%BC%E5%90%88%E7%A4%BA%E4%BE%8B4%EF%BC%9A%E5%8F%A6%E4%B8%80%E4%B8%AA%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92-housing%E6%95%B0%E6%8D%AE%E9%9B%86_12_0.png)
    


### 1.3 拆分数据集
将housing数据拆出20%作为验证数据


```python
# to make this notebook's output identical at every run
np.random.seed(42)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)
print(len(housing),len(train_set),len(test_set))
```

    20640 16512 4128


### 1.4 数据预处理




```python
housing_labels = train_set['median_house_value']
housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_num = housing.drop('ocean_proximity', axis=1)

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])


housing_prepared = full_pipeline.fit_transform(housing)


```

## 2、模型训练

sklearn提供了很多预定义好的模型，我们先用一个最简单的线性回归模型：


```python
# 模型训练。
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```




    LinearRegression()



我们看一下几个数据的预测结果：


```python
some_data = housing.iloc[5:10]
some_labels = housing_labels.iloc[5:10]
some_data_prepared = full_pipeline.transform(some_data)
print('Prediction:', lin_reg.predict(some_data_prepared))
print('Labels:', list(some_labels))
```

    Prediction: [313934.96195006 153633.74406862 406149.28021971  93616.31093607
     234277.09070168]
    Labels: [264800.0, 157300.0, 500001.0, 139800.0, 315600.0]


## 3、效果评估

### 3.1 MSE
上面我们简单的抽查了几条数据，现在我们用其它方式科学的评估模型的效果。我们先看一下均方根误差：


```python
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse = np.sqrt(lin_mse)
lin_mse
```




    68433.93736666226



我们看看决策树的效果：


```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_mse = np.sqrt(tree_mse)
tree_mse
```




    0.0



对比两种方案，线性回归对自身训练数据的预测误差都很差，这是典型的对训练数据欠拟合的案例。这通常是由于特征无法提供足够的信息，或者模型不够复杂导致的。
而决策树的误差为0，这通常会是过拟合。所以我们尝试用其它方式去评估模型。

### 3.2 交叉验证

另一个不错的选择是使用ScikitLearn的K折交叉验证功能。它将训练集随机分割成K个不同的子集，每个子集称为一个折叠，然后对决树模型进行K次训练和评估——每次挑选1个折叠进行评估，使用另外的K-1个折叠进行训练。产生的结果是一个包含K次评估分数的数组：



```python
from sklearn.model_selection import cross_val_score
tree_reg = DecisionTreeRegressor()
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_score = np.sqrt(-scores)

```


```python
print('score:', tree_rmse_score)
print('mean:', tree_rmse_score.mean())
print('std:', tree_rmse_score.std())
```

    score: [64856.51386495 70030.39848802 67233.69703686 71476.23193985
     65898.64473006 66835.32830164 63023.65507787 69134.41153338
     69542.01249378 68789.36987412]
    mean: 67682.02633405241
    std: 2459.4373154388645


## 4、模型调整
我们得到模型的评估数据后，就需要想办法优化模型，其中一个常见的优化就是调整超参数。你可以手工调整，但这个过程枯燥且低效，常用的方式可以有网格搜索和随机搜索。

### 4.1 网格搜索


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

#最优超参数组合
grid_search.best_params_

```




    {'max_features': 6, 'n_estimators': 30}



如果GridSearchCV被初始化为refit=True（默认值），那么一旦通过交叉验证找到了最佳估算器，它将在整个训练集上重新训练。这通常是个好方法，因为提供更多的数据很可能提升其性能。

我们看看详细的结果：


```python
cvres=grid_search.cv_results_

for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)


```

    64016.59946998782 {'max_features': 2, 'n_estimators': 3}
    54631.31575072553 {'max_features': 2, 'n_estimators': 10}
    52195.51217155853 {'max_features': 2, 'n_estimators': 30}
    61008.4047006966 {'max_features': 4, 'n_estimators': 3}
    52941.76153532989 {'max_features': 4, 'n_estimators': 10}
    50399.140234726656 {'max_features': 4, 'n_estimators': 30}
    58238.18136446821 {'max_features': 6, 'n_estimators': 3}
    51522.146843127375 {'max_features': 6, 'n_estimators': 10}
    49391.2283040508 {'max_features': 6, 'n_estimators': 30}
    58308.72013029112 {'max_features': 8, 'n_estimators': 3}
    51515.66096982488 {'max_features': 8, 'n_estimators': 10}
    49661.23600682289 {'max_features': 8, 'n_estimators': 30}
    62038.46647697507 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    52940.60072344172 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59492.93740877108 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    53245.14083240022 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    58712.27087046005 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51864.47273598254 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


### 4.2 随机搜索

暂略



```python

```
