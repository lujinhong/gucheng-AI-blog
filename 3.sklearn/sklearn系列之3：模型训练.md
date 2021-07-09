```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os, urllib, tarfile

```

本文主要介绍了模型训练中的各方面内容，包括模型构建、loss、优化器、metrics、正则化、学习率、激活函数、epochs、参数初始化、超参数搜索等。



## 1、模型训练
sklearn使用numpy ndarray或者pandas dataframe作为训练数据，调用fit()函数即可完成训练。

### 1.1 二分类

我们先看一个二分类问题，将mnist分类成数字5和非5两类：


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X,y = mnist['data'], mnist['target']

X_train, X_test = X[:6000], X[6000:]
y_train, y_test = y[:6000].astype(np.uint8), y[6000:].astype(np.uint8)
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss='hinge')
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([X[0]]))

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_model, X_train, y_train_5, cv=3, scoring='accuracy')
```

    [False]





    array([0.96  , 0.9575, 0.964 ])



### 1.2 回归

我们再看一个回归算法的示例，使用的是housing数据集，预测地区房产的中位数。


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
    
# fetch_housing_data()

housing = pd.read_csv(os.path.join(HOUSING_PATH,'housing.csv'))

median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace=True)

housing_label = housing['median_house_value']
housing_feature = housing.drop(['median_house_value','ocean_proximity'], axis=1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_feature,housing_label)
print(model.intercept_, model.coef_)
```

    -3570118.0614940603 [-4.26104026e+04 -4.24754782e+04  1.14445085e+03 -6.62091740e+00
      8.11609666e+01 -3.98732002e+01  7.93047225e+01  3.97522237e+04]



```python

```

## 4、正则化

正则化是处理模拟过拟合最常用的方式之一。本部分我们介绍常见的正则化方法。

###  4.1 L1和L2

Lasso回归的一个重要特点是它倾向于完全消除掉最不重要特征的权重（也就是将它们设置为零）因为所有高阶多项式的特征权重都等于零。换句话说，Lasso回归会自动执行特征选择并输出一个稀疏模型（即只有很少的特征有非零权重）。

你可以通过查下图来了解为什么会这样：轴代表两个模型参数，背景轮廓代表不同的损失函数。在左上图中，轮廓线代表1损失（|θ1|+|θ2|），当你靠近任何轴时，该损失呈线性下降。例如，如果将模型参数初始化为θ1=2和θ2=0.5，运行梯度下降会使两个参数均等地递减（如黄色虚线所示）。因此θ2将首先达到0（因为开始时接近0）。之后，梯度下降将沿山谷滚动直到其达到θ1=0（有一点反弹，因为1的梯度永远不会接近0：对于每个参数，它们都是1或1）。在右上方的图中，轮廓线代表Lasso的成本函数（即MSE成本函数加L1损失）。白色的小圆圈显示了梯度下降优化某些模型参数的路径，这些参数在θ1=0.25和θ2=1附近初始化：再次注意该路径如何快速到达θ2=0，然后向下滚动并最终在全局最优值附近反弹（由红色正方形表示）。如果增加α，则全局最优值将沿黄色虚线向左移动；如果减少α，则全局最优值将向右移动（在此示例中，非正则化的MSE的最优参数为θ1=2和θ2=0.5）。
![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/%E6%88%AA%E5%B1%8F2021-05-25%20%E4%B8%8A%E5%8D%8810.02.31.png)
底部的两个图显示了相同的内容，但惩罚为L2。在左下图中，你可以看到L2损失随距原点的距离而减小，因此梯度下降沿该点直走。在右下图中，轮廓线代表岭回归的成本函数（即MSE成本函数加L2损失）。Lasso有两个主要区别。首先，随着参数接近全局最优值，梯度会变小，因此，梯度下降自然会减慢，这有助于收敛（因为周围没有反弹）。其次，当你增加α时，最佳参数（用红色正方形表示）越来越接近原点，但是它们从未被完全被消除。



#### L2正则化、岭回归

本部分介绍了sklearn中岭回归的实现方式。

sklearn提供了一个Ridge的线性领回归的实现，但更常用的方式是在其它模型中加入penalty='l2'的参数。我们先看一下Rideg类的使用：


```python
X = np.random.rand(1000, 1)
y = 2 * X + 1

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(X, y)
print(ridge_reg.coef_, ridge_reg.intercept_)

```

    [[1.97643979]] [1.01167015]


我们再看一下使用其它模型+penalty参数的方式：


```python
X = np.random.rand(1000, 1)
y = 2 * X + 1

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(X, y.ravel())
print(ridge_reg.coef_, ridge_reg.intercept_)

```

    [[1.97643979]] [1.01167015]


#### L1正则化、Lasso回归

与L2类似，L1也有Lasso和penalty2种实现方式：


```python
X = np.random.rand(100, 1) * 10
y = 12 * X + 256

from sklearn.linear_model import Lasso
lasso_reg = Lasso()
lasso_reg.fit(X, y)
print(lasso_reg.coef_, lasso_reg.intercept_)
```

    [11.88574327] [256.58076116]



```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty='l1')
sgd_reg.fit(X, y.ravel())
print(sgd_reg.coef_, sgd_reg.intercept_)
```

    [12.05735998] [255.60450449]


#### 弹性网络

弹性网络ElasticNet同时使用L1和L2。


```python
X = np.random.rand(100, 1) * 10
y = 12 * X + 256

from sklearn.linear_model import ElasticNet
en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
en_reg.fit(X,y)
print(en_reg.coef_, en_reg.intercept_)
```

    [11.91128824] [256.42500209]



```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty='elasticnet')
sgd_reg.fit(X, y.ravel())
print(sgd_reg.coef_, sgd_reg.intercept_)
```

    [12.06448304] [255.59790417]


### 4.2 提前停止

为了避免过拟合，也为了记录训练过程中的最优模型，我们经常需要用到提前停止。


```python
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)
```

## 5、metrics

chpt3的二分类问题以及metrics； 综合示例4housing数据集回归算法的metric。

多分类的metrix问题，请见sklearn综合示例5：多分类问题。
### 5.1 MSE

我们先看一下回归问题常用的均方根误差MSE。


```python
from sklearn.metrics import mean_squared_error
housing_pred = lin_reg.predict(housing_feature)
lin_mse = mean_squared_error(housing_label, housing_pred)
print(np.sqrt(lin_mse))
```

    69658.1903557702


### 5.2 交叉验证
使用sklearn提供的cross_val_score()，我们可以很方便的交叉验证模型效果。比如，我们看一下上面5和非5的线性分类器的准确率：


```python
from sklearn.model_selection import cross_val_score, cross_val_predict
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

```




    array([0.9615, 0.9595, 0.9535])



上述代码中，我们随机划分训练数据和测试数据，训练模型后计算准确率，并重复了3次。

### 5.3 准确率、精度、召回率、F1、AUC

下面我们主要看一下准确率、精度、召回率、F1、ROC/AUC等常用于二分类问题的metrics。

#### 准确率


```python
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
y_pred_5 = sgd_clf.predict(X_test)

accuracy_score(y_test_5, y_pred_5)
```




    0.96165625



#### 混淆矩阵


```python
confusion_matrix(y_test_5, y_pred_5)
```




    array([[57323,   878],
           [ 1576,  4223]])



#### 精度、召回率、F1


```python
precision_score(y_test_5, y_pred_5)

```




    0.8278768868849246




```python
recall_score(y_test_5, y_pred_5)

```




    0.7282290050008622




```python
f1_score(y_test_5, y_pred_5)
```




    0.774862385321101



#### ROC & AUC


```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test_5, y_pred_5)
```




    0.856571676775787



### 5.4 阈值衡量、ROC曲线

sklearn不允许对分类模型直接设置阈值，但是可以访问它用于预测的决策分数。不是调用分类器的predict()函数，而是调用decision_function()函数，这种方法返回每个实例的分数，然后就可以根据这些分数，使用任意阈值进行预测了。

我们先看个示例：


```python
y_pred = sgd_clf.predict([X_test[11]])
print(y_pred)

y_score = sgd_clf.decision_function([X_test[11]])
print(y_score)
```

    [ True]
    [58446.52780903]


我们随机抽取了一个样本，其score=41983，而默认的阈值为0，所以预测结果为True。如果我们现在想提高精度（降低其召回率），那可以提高其阈值：


```python
threshold = 50000
y_predict_t = (y_score > threshold)
print(y_predict_t)

accuracy = accuracy_score(y_test, y_predict_t)
precision = precision_score(y_test, y_predict_t)
recall = recall_score(y_test, y_predict_t)
f1 = f1_score(y_test, y_predict_t)
auc = roc_auc_score(y_test, y_predict_t)
print(accuracy, precision, recall, f1, auc)
```

    [ True]


#### 阈值选择

那怎么选取合适的阈值呢？

我们先使用cross_val_predict()获取决策分数而非预测结果；然后使用precision_recall_curve()计算所有可能阈值的精度和召回率；最后使用matplotlib绘制精度和召回率相对于阈值的函数组：


```python
y_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_score)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
```


    
![png](sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_files/sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_39_0.png)
    


根据上图，可以选择合适的阈值。

假设你决定将精度设置为90%：


```python
threshold_90_precision = thresholds[np.argmax(precisions>=0.90)]
print(threshold_90_precision)
```

    261289.38745837728


取的合适的阈值后，我们可以这样指定最终的预测结果：


```python
y_pred_90 = (y_score >= threshold_90_precision)
print(y_pred_90)
```

    [False False False ... False False False]


#### ROC曲线

画ROC曲线和上述的精度、召回率曲线类似，但要先算出FPR和TPR：


```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_score)
def plt_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    
plt_roc_curve(fpr, tpr)
plt.show()
```


    
![png](sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_files/sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_45_0.png)
    


画出ROC曲线后，可用上述的方法计算得到AUC：


```python
roc_auc_score(y_test_5, y_pred_5)
```




    0.856571676775787



## 6、多项式回归与学习曲线

我们先简单看看多项式回归，然后通过学习曲线看一下过拟合的情况。

### 6.1 多项式回归

我们使用二次方程生成一些数据，然后拟合。


```python
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(100,1)
```

显然，线性模型很难拟合上述数据，于是我们使用sklearn的PolynomialFeatures类来转换训练数据，将训练集中每个特征的平方（二次多项式）添加为新特征（在这种情况下，只有一个特征）：


```python
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_feature.fit_transform(X)
print(X[0], X_poly[0])
```

    [0.20297397] [0.20297397 0.04119843]


使用ScikitLearn的PolynomialFeatures类来转换训练数据，将训练集中每个特征的平方（二次多项式）添加为新特征（在这种情况下，只有一个特征）：


```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)
```

    [2.47928928] [[0.97666236 0.51063963]]


可以看出，非常接近原来的0.5， 1.0， 2。其中截距由于加了一个0-1间的随机数，所以是2.5。

### 6.2 学习曲线

我看可以使用交叉验证来估计模型的泛化性能。如果模型在训练数据上表现良好，但根据交叉验证的指标泛化较差，则你的模型过拟合。如果两者的表现均不理想，则说明欠拟合。这是一种区别模型是否过于简单或过于复杂的方法。

还有一种方法是观察学习曲线：这个曲线绘制的是模型在训练集和验证集上关于训练集大小（或训练迭代）的性能函数。要生成这个曲线，只需要**在不同大小的训练子集上多次训练模型**即可。


```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
        val_errors.append(mean_squared_error(y_val, y_val_pred))
        plt.plot(np.sqrt(train_errors), 'b--', label='train')
        plt.plot(np.sqrt(val_errors), 'r-+', label='val')
        plt.grid()
```

我们看一下使用线性模型的学习曲线：


```python
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

print(lin_reg.intercept_, lin_reg.coef_)
```

    [4.22219747] [[0.96475535]]



    
![png](sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_files/sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_57_1.png)
    


这种欠拟合的模型值得解释一下。首先，让我们看一下在训练数据上的性能：当训练集中只有一个或两个实例时，模型可以很好地拟合它们，这就是曲线从零开始的原因。但是，随着将新实例添加到训练集中，模型就不可能完美地拟合训练数据，这既因为数据有噪声，又因为它根本不是线性的。因此，训练数据上的误差会一直上升，直到达到平稳状态，此时在训练集中添加新实例并不会使平均误差变好或变差。现在让我们看一下模型在验证数据上的性能。当在很少的训练实例上训练模型时，它无法正确泛化，这就是验证误差最初很大的原因。然后，随着模型经历更多的训练示例，它开始学习，因此验证错误逐渐降低。但是，直线不能很好地对数据进行建模，因此误差最终达到一个平稳的状态，非常接近另外一条曲线。这些学习曲线是典型的欠拟合模型。两条曲线都达到了平稳状态。它们很接近而且很高。

下面我们看一下二次多项式的拟合曲线：


```python
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_feature.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

plot_learning_curves(lin_reg, X, y)
```

    [2.47928928] [[0.97666236 0.51063963]]



    
![png](sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_files/sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_59_1.png)
    


我们再看一下10次多项式的拟合情况：


```python
poly_feature = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly_feature.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

plot_learning_curves(lin_reg, X, y)
```

    [2.42873441] [[ 9.37659988e-01  7.48071894e-01  3.63005431e-02 -1.80873839e-01
      -2.84013663e-02  4.92345732e-02  7.38648608e-03 -5.63850017e-03
      -5.45818240e-04  2.36152880e-04]]



    
![png](sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_files/sklearn%E7%B3%BB%E5%88%97%E4%B9%8B3%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83_61_1.png)
    


统计学和机器学习的重要理论成果是以下事实：模型的泛化误差可以表示为三个非常不同的误差之和：

**偏差**

这部分泛化误差的原因在于错误的假设，比如假设数据是线性的，而实际上是二次的。高偏差模型最有可能欠拟合训练数据。

**方差**

这部分是由于模型对训练数据的细微变化过于敏感。具有许多自由度的模型（例如高阶多项式模型）可能具有较高的方差，因此可能过拟合训练数据。

**不可避免的误差**

这部分误差是因为数据本身的噪声所致。减少这部分误差的唯一方法就是清理数据（例如修复数据源（如损坏的传感器），或者检测并移除异常值）。

增加模型的复杂度通常会显著提升模型的方差并减少偏差。反过来，降低模型的复杂度则会提升模型的偏差并降低方差。这就是为什么称其为权衡。



```python

```
