本文主要示范使用sklearn实现决策树的方法。


```python
import numpy as np
import pandas as pd
import sklearn
```

示例比较简单，我们使用iris数据集的长度和宽度特征，判断鸢尾花的类别。


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X, y)
```




    DecisionTreeClassifier(max_depth=3)



要将决策树可视化，我们可以使用export_graphviz()方法输出一个图形定义文件：


```python
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file='/Users/lujinhong/Downloads/iris_tree.dot',
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
```

我们使用以下命令将上述文件生成图片：
dot -Tpng iris_tree.dot -o iris_tree.png

![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/iris_tree.png)

DecisionTreeClassfier还有很多参数，比如：


![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/1.png)
![](https://lujinhong-markdown.oss-cn-beijing.aliyuncs.com/md/截屏2021-06-03下午4.53.11.png)

决策树DecisionTreeRegressor可以用于回归问题。


```python

```
