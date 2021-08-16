假设你创建了一个包含1000个分类器的集成，每个分类器都只有51%的概率是正确的（几乎不比随机猜测强多少）。如果你以大多数投票的类别作为预测结果，可以期待的准确率高达75%。但是，这基于的前提是所有的分类器都是完全独立的，彼此的错误毫不相关。显然这是不可能的，因为它们都是在相同的数据上训练的，很可能会犯相同的错误，所以也会有很多次大多数投给了错误的类别，导致集成的准确率有所降低。

当预测器尽可能互相独立时，集成方法的效果最优。获得多种分类器的方法之一就是使用不同的算法进行训练。这会增加它们犯不同类型错误的机会，从而提升集成的准确率。


```python
import numpy as np
import pandas as pd
import sklearn
```

## 1、投票分类器
我们使用多个模型对同一个事件进行预测，然后根据各个分类器的预测结果投票决定最终的分类结果。


```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_clf = LogisticRegression()
rndf_clf = RandomForestClassifier()
svm_clf = SVC()

vot_clf = VotingClassifier(
    estimators=[('lr',log_clf),('rf',rndf_clf),('svc',svm_clf)],
    voting='hard'
)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rndf_clf, svm_clf, vot_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
```

    LogisticRegression 0.864
    RandomForestClassifier 0.888
    SVC 0.896
    VotingClassifier 0.904


可以看出来，投票分类器略优于其它单个分类器。

如果所有分类器都能够估算出类别的概率（即有predict_proba（）方法），那么你可以将概率在所有单个分类器上平均，然后让ScikitLearn给出平均概率最高的类别作为预测。这被称为软投票法。通常来说，它比硬投票法的表现更优，因为它给予那些高度自信的投票更高的权重。而所有你需要做的就是用voting="soft"代替voting="hard"，并确保所有分类器都可以估算出概率。默认情况下，SVC类是不行的，所以你需要将其超参数probability设置为True（这会导致SVC使用交叉验证来估算类别概率，减慢训练速度，并会添加predict_proba（）方法）。

## 2、bagging与pasting

前面提到，获得不同种类分类器的方法之一是使用不同的训练算法。还有另一种方法是每个预测器使用的算法相同，但是在不同的训练集随机子集上进行训练。采样时如果将样本放回，这种方法叫作bagging（bootstrap aggregating的缩写，也叫自举汇聚法）。采样时样本不放回，这种方法则叫作pasting。

换句话说，bagging和pasting都允许训练实例在多个预测器中被多次采样，但是只有bagging允许训练实例被同一个预测器多次采样。

一旦预测器训练完成，集成就可以通过简单地聚合所有预测器的预测来对新实例做出预测。聚合函数通常是统计法（即最多数的预测与硬投票分类器一样）用于分类，或是平均法用于回归。每个预测器单独的偏差都高于在原始训练集上训练的偏差，但是通过聚合，同时降低了偏差和方差。总体来说，最终结果是，与直接在原始训练集上训练的单个预测器相比，集成的偏差相近，但是方差更低。

* 你可以通过不同的CPU内核甚至不同的服务器并行地训练预测器。类似地，预测也可以并行。参数n_jobs用来指示ScikitLearn用多少CPU内核进行训练和预测（1表示让ScikitLearn使用所有可用内核）
* 这是一个bagging的示例，如果你想使用pasting，只需要设置bootstrap=False即可



```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.936


如果基本分类器可以估计类别概率（如果它具有predict_proba（）方法），则BaggingClassifier自动执行软投票而不是硬投票，在决策树分类器中就是这种情况。


### 包外评估

对于任意给定的预测器，使用bagging，有些实例可能会被采样多次，而有些实例则可能根本不被采样。BaggingClassifier默认采样m个训练实例，然后放回样本（bootstrap=True），m是训练集的大小。这意味着对每个预测器来说，平均只对63%的训练实例进行采样。剩余37%未被采样的训练实例称为包外（oob）实例。

由于预测器在训练过程中从未看到oob实例，因此可以在这些实例上进行评估，而无须单独的验证集。你可以通过平均每个预测器的oob评估来评估整体。在ScikitLearn中，创建BaggingClassifier时，设置oob_score=True就可以请求在训练结束后自动进行包外评估。


```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    oob_score=True, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.904


## 3、随机补丁和随机子空间

BaggingClassifier类也支持**对特征进行采样**。采样由两个超参数控制：max_features和bootstrap_features。它们的工作方式与max_samples和bootstrap相同，但用于特征采样而不是实例采样。因此，每个预测器将用输入特征的随机子集进行训练。

这对于处理高维输入（例如图像）特别有用。对训练实例和特征都进行抽样，这称为随机补丁方法。而保留所有训练实例（即bootstrap=False并且max_samples=1.0）但是对特征进行抽样（即bootstrap_features=True并且/或max_features<1.0），这被称为随机子空间法。

对特征抽样给预测器带来更大的多样性，所以以略高一点的偏差换取了更低的方差。



## 4、随机森林

随机森林大致与上面的BaggingClassifier相同。

除少数例外，RandomForestClassifier具有DecisionTreeClassifier的所有超参数（以控制树的生长方式），以及BaggingClassifier的所有超参数来控制集成本身。

随机森林在树的生长上引入了更多的随机性：分裂节点时不再是搜索最好的特征（见第6章），而是在一个随机生成的特征子集里搜索最好的特征。这导致决策树具有更大的多样性，（再一次）用更高的偏差换取更低的方差，总之，还是产生了一个整体性能更优的模型。


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rndf_clf = RandomForestClassifier(n_estimators=500,oob_score=True, bootstrap=True, n_jobs=-1, max_leaf_nodes=16)

rndf_clf.fit(X_train, y_train)
y_pred = rndf_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.912


### 特征重要性
随机森林的另一个好特性是它们使测量每个特征的相对重要性变得容易。ScikitLearn通过查看使用该特征的树节点平均（在森林中的所有树上）减少不纯度的程度来衡量该特征的重要性。更准确地说，它是一个加权平均值，其中每个节点的权重等于与其关联的训练样本的数量。

ScikitLearn会在训练后为每个特征自动计算该分数，然后对结果进行缩放以使所有重要性的总和等于1。你可以使用feature_importances_变量来访问结果。例如，以下代码在鸢尾花数据集上训练了RandomForestClassifier，并输出每个特征的重要性。看起来最重要的特征是花瓣长度（47%）和宽度（42%），而花萼的长度和宽度则相对不那么重要。


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()

rndf_clf = RandomForestClassifier()
rndf_clf.fit(iris['data'], iris['target'])
for name,score in zip(iris["feature_names"],rndf_clf.feature_importances_):
    print(name, score)
```

    sepal length (cm) 0.0916662781753619
    sepal width (cm) 0.017065083772560878
    petal length (cm) 0.4729672623846955
    petal width (cm) 0.41830137566738174


## 5、提升法之AdaBoost
本部分只介绍了基本的原理，更关注的是其实现。详细的理论请参考其它文章。

提升法（boosting，最初被称为假设提升）是指可以将几个弱学习器结合成一个强学习器的任意集成方法。大多数提升法的总体思路是循环训练预测器，每一次都对其前序做出一些改正。可用的提升法有很多，但目前最流行的方法是AdaBoost（AdaptiveBoosting的简称）和梯度提升。

新预测器对其前序进行纠正的方法之一就是更多地关注前序欠拟合的训练实例，从而使新的预测器不断地越来越专注于难缠的问题，这就是AdaBoost使用的技术。

例如，当训练AdaBoost分类器时，该算法首先训练一个基础分类器（例如决策树），并使用它对训练集进行预测。然后，该算法会增加分类错误的训练实例的相对权重。然后，它使用更新后的权重训练第二个分类器，并再次对训练集进行预测，更新实例权重，以此类推

这种依序学习技术有一个重要的缺陷就是无法并行（哪怕只是一部分），因为每个预测器只能在前一个预测器训练完成并评估之后才能开始训练。因此，在扩展方面，它的表现不如bagging和pasting方法。



```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.896


## 6、提升法之梯度提升：GBDT xgboost
与AdaBoost一样，梯度提升也是逐步在集成中添加预测器，每一个都对其前序做出改正。不同之处在于，它不是像AdaBoost那样在每个迭代中调整实例权重，而是让新的预测器针对前一个预测器的残差进行拟合。

详见GBDT&xgboost相关文章。

## 7、堆叠法

本章我们要讨论的最后一个集成方法叫作堆叠法（stacking），又称层叠泛化法。它基于一个简单的想法：与其使用一些简单的函数（比如硬投票）来聚合集成中所有预测器的预测，我们为什么不训练一个模型来执行这个聚合呢？图示显示了在新实例上执行回归任务的这样一个集成。底部的三个预测器分别预测了不同的值，然后最终的预测器（称为混合器或元学习器）将这些预测作为输入，进行最终预测。

训练混合器的常用方法是使用留存集。我们看看它是如何工作的。首先，将训练集分为两个子集，第一个子集用来训练第一层的预测器。

然后，用第一层的预测器在第二个（留存）子集上进行预测）。因为预测器在训练时从未见过这些实例，所以可以确保预测是“干净的”。那么现在对于留存集中的每个实例都有了三个预测值。我们可以使用这些预测值作为输入特征，创建一个新的训练集（新的训练集有三个维度），并保留目标值。在这个新的训练集上训练混合器，让它学习根据第一层的预测来预测目标值。

事实上，通过这种方法可以训练多种不同的混合器（例如，一个使用线性回归，另一个使用随机森林回归，等等）。于是我们可以得到一个混合器层。诀窍在于将训练集分为三个子集：第一个用来训练第一层，第二个用来创造训练第二层的新训练集（使用第一层的预测），而第三个用来创造训练第三层的新训练集（使用第二层的预测）。一旦训练完成，我们可以按照顺序遍历每层来对新实例进行预测，




```python

```
