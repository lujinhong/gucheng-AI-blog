### 按行求和并添加到df中


```python
X_train['label_count'] = X_train.apply(lambda x: x.sum(), axis=1)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-fc5dadd28777> in <module>
    ----> 1 X_train['label_count'] = X_train.apply(lambda x: x.sum(), axis=1)
    

    NameError: name 'X_train' is not defined



```python

```
