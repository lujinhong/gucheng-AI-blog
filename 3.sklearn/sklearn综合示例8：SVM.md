## 1、线性SVM

**SVM特别适用于中小型复杂数据集的分类。**

**SVM对特征的缩放非常敏感**

SVM的基本思想可以用一些图来说明。图51所示的数据集来自第4章末尾引用的鸢尾花数据集的一部分。两个类可以轻松地被一条直线（它们是线性可分离的）分开。左图显示了三种可能的线性分类器的决策边界。其中虚线所代表的模型表现非常糟糕，甚至都无法正确实现分类。其余两个模型在这个训练集上表现堪称完美，但是它们的决策边界与实例过于接近，导致在面对新实例时，表现可能不会太好。相比之下，右图中的实线代表SVM分类器的决策边界，这条线不仅分离了两个类，并且尽可能远离了最近的训练实例。你可以将SVM分类器视为在类之间拟合可能的最宽的街道（平行的虚线所示）。因此这也叫作大间隔分类。

请注意，在“街道以外”的地方增加更多训练实例不会对决策边界产生影响，也就是说它完全由位于街道边缘的实例所决定（或者称之为“支持”）。这些实例被称为支持向量（在图51中已圈出）。

![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/截屏2021-06-09上午10.25.33.png)


```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris['data'][:,(2,3)]#长度与宽度
y = (iris['target']= =2).astype(np.float64)

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge'))
])
svm_clf.fit(X, y)

svm_clf.predict([[5.5,1.7]])

```




    array([1.])



## 2、非线性SVM分类：多项式
虽然在许多情况下，线性SVM分类器是有效的，并且通常出人意料的好，但是，有很多数据集远不是线性可分离的。处理非线性数据集的方法之一是添加更多特征，比如多项式特征。某些情况下，这可能导致数据集变得线性可分离。假设一个简单的数据集，只有一个特征x1，数据集线性不可分。但是如果添加第二个特征x2=（x1)^2，生成的2D数据集则完全线性可分离。


```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
```




    Pipeline(steps=[('poly_features', PolynomialFeatures(degree=3)),
                    ('scaler', StandardScaler()),
                    ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))])



添加多项式特征实现起来非常简单，并且对所有的机器学习算法（不只是SVM）都非常有效。但问题是，如果多项式太低阶，则处理不了非常复杂的数据集。而高阶则会创造出大量的特征，导致模型变得太慢。幸运的是，使用SVM时，有一个魔术般的数学技巧可以应用，这就是核技巧（稍后解释）。它产生的结果就跟添加了许多多项式特征（甚至是非常高阶的多项式特征）一样，但实际上并不需要真的添加。因为实际没有添加任何特征，所以也就不存在数量爆炸的组合特征了。这个技巧由SVC类来实现，我们看看在卫星数据集上的测试：



```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=5, coef0=1, kernel='poly'))])



## 非线性SVM分类：高斯核

与多项式特征方法一样，相似特征法也可以用任意机器学习算法，但是要计算出所有附加特征，其计算代价可能非常昂贵，尤其是对大型训练集来说。然而，核技巧再一次施展了它的SVM魔术：它能够产生的结果就跟添加了许多相似特征一样（但实际上也并不需要添加）。我们来使用SVC类试试高斯RBF核：


```python
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=0.001, gamma=5))])




```python

```
