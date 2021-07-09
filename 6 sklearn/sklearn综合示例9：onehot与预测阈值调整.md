本文介绍了：
1. 如何将多个标签做onehot，比如说总共有1000个标签，用户带了其中100个标签，那就是一个1000维的feautre，其中100维=1，其余900维=0。
2. 调整分类算法的分类阈值，比如将LR中的默认阈值从0调整到0.9，降低recall提升精度。
3. 各种算法的使用方式。

# 1、数据预处理

## 样本格式
最终得到的样本格式如下，第一列是label，第二列是一“|”分割的一些特征，可以理解为用户观看了哪部电影，喜欢哪本书，关注了哪个微博id等。
label,features
1,20018
0,20006|20025
1,1509|8713|2000341|9010
我们读取数据后，将数据做onehot。比如总共有10000个标签，如果设备带了其中1000个，则标签有10000列，其中1000列为1，其余为0.


```python
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
```

## 载入数据


```python
sample_dir = '/home/ljhn1829/jupyter_data/sklearn_onehot_threshold.csv'
df_sample_all = pd.read_csv(sample_dir)
print(df_sample_all.head())
```

       label                                           features
    0      1                                            2001841
    1      0                                    2000641|2002541
    2      1  1509|871305|2000341|901005|147409|132905|13560...
    3      1  1034005|20909|9505|1083505|69209|19109|10905|9...
    4      1  148009|4109|3809|169105|685006|62409|99805|200...


## onehot

除了本文的sklearn方式外，也可以使用pandas.get_dummies()。但数据量比较大，而且稀疏时，建议使用sklearn。

见《sklearn系列之2：数据预处理》及https://stackoverflow.com/questions/63544536/convert-pd-get-dummies-result-to-df-str-get-dummies


```python
mlb = MultiLabelBinarizer(sparse_output=True)
onehot_output = pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df_sample_all['features'].str.split('|')),
                                           columns=mlb.classes_)
df_sample_onehot_all = pd.DataFrame()
df_sample_onehot_all['label'] = df_sample_all['label']
df_sample_onehot_all= pd.concat([df_sample_onehot_all,onehot_output], axis=1)
print(df_sample_onehot_all.head())
```

       label  1000005  100008  100009  10001  1000108  10002  1000208  10005  \
    0      1        0       0       0      0        0      0        0      0   
    1      0        0       0       0      0        0      0        0      0   
    2      1        0       0       0      0        0      0        0      0   
    3      1        0       0       0      0        0      0        0      0   
    4      1        0       0       0      0        0      0        0      0   
    
       10007  ...  998805  999008  99905  99908  99909  999505  999508  999708  \
    0      0  ...       0       0      0      0      0       0       0       0   
    1      0  ...       0       0      0      0      0       0       0       0   
    2      0  ...       0       0      0      0      0       0       0       0   
    3      0  ...       0       0      0      0      0       0       0       0   
    4      0  ...       0       0      0      0      0       0       0       0   
    
       999805  999808  
    0       0       0  
    1       0       0  
    2       0       0  
    3       0       0  
    4       0       0  
    
    [5 rows x 23720 columns]



```python
print(df_sample_onehot_all['label'].value_counts())
```

    0    5493
    1    4506
    Name: label, dtype: int64



```python
#print(df_sample_onehot_all['1000005'].value_counts())
```

## 数据集拆分
将数据集拆分为训练、测试集。


```python
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(df_sample_onehot_all, test_size=0.2, random_state=42)
# test_set, train_set = df_sample_onehot_all[0:30], df_sample_onehot_all[30:100]

def split_train_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio) 
    print(test_size)
    training_idx, test_idx = shuffle_indices[test_size:], shuffle_indices[:test_size]
    return data.iloc[training_idx], data.iloc[test_idx]

train_set, test_set = split_train_test(df_sample_onehot_all, 0.2)
```

    1999


看一下训练集和测试集：


```python
print(train_set['label'].value_counts())
print(test_set['label'].value_counts())
```

    0    4384
    1    3616
    Name: label, dtype: int64
    0    1109
    1     890
    Name: label, dtype: int64


将XY分开：


```python
y_train, X_train = train_set['label'], train_set.iloc[:, 1:]
y_test, X_test = test_set['label'], test_set.iloc[:, 1:]
```


```python
print(train_set.head())
```

          label  1000005  100008  100009  10001  1000108  10002  1000208  10005  \
    45        1        0       0       0      0        0      0        0      0   
    9521      0        0       0       0      0        0      0        0      0   
    7718      0        0       0       0      0        0      0        0      0   
    7054      0        0       0       0      0        0      0        0      0   
    4605      1        0       0       0      0        0      0        0      0   
    
          10007  ...  998805  999008  99905  99908  99909  999505  999508  999708  \
    45        0  ...       0       0      0      0      0       0       0       0   
    9521      0  ...       0       0      0      0      0       0       0       0   
    7718      0  ...       0       0      0      0      0       0       0       0   
    7054      0  ...       0       0      0      0      0       0       0       0   
    4605      0  ...       0       0      0      0      0       0       0       0   
    
          999805  999808  
    45         0       0  
    9521       0       0  
    7718       0       0  
    7054       0       0  
    4605       0       0  
    
    [5 rows x 23720 columns]


看一下哪些属性和label最相关


```python
corr_matrix = train_set.corr()
corr_matrix['label'].sort_values(ascending=False)
```

# 模型训练
我们使用各种模型训练上述数据

## LR


```python
from sklearn.linear_model import SGDClassifier, LogisticRegression
#clf = LogisticRegression(loss='log')
clf = LogisticRegression(penalty='l2', C=0.1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
#print(pred)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,roc_auc_score

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, pred)
print(accuracy, precision, recall, f1, auc)

from sklearn.model_selection import cross_val_score, cross_val_predict
# cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')

```

    0.5722861430715358 0.5323741007194245 0.4884488448844885 0.5094664371772806 0.5653253398734368


## 阈值调整

我们调整一下预测分类的阈值，LogisticRegression的预测范围是[-1,1]，阈值默认取0.0。

下面我们先得到预测的具体数值，然后通过调整阈值的方式，调整其分类。


```python
y_score = clf.decision_function(X_test)
print(y_score)
```

    [ 0.21320269  0.25052858 -0.47845967 ...  0.83939436  0.36785
      0.3116261 ]



```python
threshold = 0.99
y_predict_t = (y_score > threshold)
print(y_predict_t)
accuracy = accuracy_score(y_test, y_predict_t)
precision = precision_score(y_test, y_predict_t)
recall = recall_score(y_test, y_predict_t)
f1 = f1_score(y_test, y_predict_t)
auc = roc_auc_score(y_test, y_predict_t)
print(accuracy, precision, recall, f1, auc)

```

    [False False False ... False False False]
    0.5617808904452226 0.6222222222222222 0.0924092409240924 0.16091954022988506 0.5228101250492021

可以看到，提高阈值，可以降低recall，提升precision。
也就是说，提高分类阈值，recall从0.49降到0.09，同时精度从0.53上升到0.62。
阈值=0时，accuracy, precision, recall, f1, auc = 0.5722861430715358 0.5323741007194245 0.4884488448844885 0.5094664371772806 0.5653253398734368
阈值=0.99时，accuracy, precision, recall, f1, auc = 0.5617808904452226 0.6222222222222222 0.0924092409240924 0.16091954022988506 0.5228101250492021
## SVM


```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,roc_auc_score

accuracy = accuracy_score(y_test, pred)
#precision_score(y_test, pred)
#precision = precision_score(y_test, pred)
#recall = recall_score(y_test, pred)
#f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, pred)
#print(accuracy, precision, recall, f1, uac)
print(accuracy, auc)

cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')
```

    0.5322661330665333 0.529358807440377





    array([0.42452043, 0.5646372 , 0.39366138])



## 决策树


```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve,roc_auc_score

accuracy = accuracy_score(y_test, pred)
#precision_score(y_test, pred)
#precision = precision_score(y_test, pred)
#recall = recall_score(y_test, pred)
#f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, pred)
#print(accuracy, precision, recall, f1, uac)
print(accuracy, auc)
```

    0.5537768884442221 0.5478048263541951


## 深度学习Tensorflow


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

columns_count = X_train.shape[1]
print(columns_count)

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[columns_count,1])) #将输入的二维数组展开成一维向量
# model.add(keras.layers.Dense(300,activation='relu'))
# model.add(keras.layers.Dense(100,activation='relu'))
# model.add(keras.layers.Dense(1,activation='sigmoid'))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=[columns_count,1]),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])
model.fit(X_train, y_train)
```

    23719
    250/250 [==============================] - 1s 4ms/step - loss: 0.6861 - accuracy: 0.5537





    <tensorflow.python.keras.callbacks.History at 0x7f83b42c5a00>



用Keras做文本二分类，总是遇到以下错误，我的类别是0或1，但是错误跟我说不能是1.

参见：Received a label value of 1 which is outside the valid range of [0, 1) - Python, Keras loss function的问题。

原来用的是sparse_categorical_crossentropy，改为binary_crossentropy问题解决。

https://blog.csdn.net/The_Time_Runner/article/details/93889004



```python

```
