# Import libraries

First, we need to import all required libraries and specific items from library modules.


```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
```

# KNN 


KNN is a machine learning algorithm that is used for classification and regression tasks. It works by finding the K closest data points in the training set to a new data point in the test set and using those points to make a prediction for the label of the new data point. KNN is simple to understand, but can be computationally expensive and sensitive to the choice of K.

After importing the required libraries, the next step is to import the data and check our data type.


```python
data = pd.read_excel("data.xlsx")
data.dtypes
```




    Name        object
    Location    object
    Price       object
    Bedrooms     int64
    Surface      int64
    Type         int64
    dtype: object



As we can see, there were some errors where importing the data that needs to be fixed.
Price column type needs to be changed to int64 and not object.


```python
data['Price']= data['Price'].str.replace(' ', '')
data['Price'] = data['Price'].astype('int64')
print(data['Price'])
```

    0        70000
    1       350000
    2       450000
    3        85000
    4       170000
            ...   
    112    1000000
    113    2280000
    114    2200000
    115    1500000
    116     720000
    Name: Price, Length: 117, dtype: int64
    

Since the type is changed we can now select our features and target


```python
Y = data["Type"]
X = data[["Price", "Surface", "Bedrooms"]]
```

Then we split our data into train and test sets. 
We will use the parameter "random_state=0" to maintain the same results on each run.


```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
```

Next, we will need to scale our data using MinMaxScaler.


```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Once that's done, we will build our KNN model with k=4 in this case.


```python
knn = KNeighborsClassifier(n_neighbors=4)
```

Now let's fit the data sets into model we got.


```python
knn.fit(X_train_scaled, y_train)
```




    KNeighborsClassifier(n_neighbors=4)



Last step is to check the accuracy on both train and test sets.


```python
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_scaled, y_test)))

```

    Accuracy of K-NN classifier on training set: 0.86
    Accuracy of K-NN classifier on test set: 0.80
    

# SVM 


What is SVM? 
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.
SVMs are one of the most robust prediction methods
In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

The next logical steps after importing the required data is to import data, split it into train and test sets, and scale it. But since we've already don that, we will directly move to building our SVM model.


```python
SVM = SVC(C=20)
```

Now let's fit the data sets into model we got.


```python
SVM.fit(X_train_scaled, y_train)
```




    SVC(C=20)



Last step is to check the accuracy on both train and test sets.


```python
print('RBF-kernel SVC training set accuracy: {:.2f}'
     .format(SVM.score(X_train_scaled, y_train)))
print('RBF-kernel SVC test set accuracy: {:.2f}'
     .format(SVM.score(X_test_scaled, y_test)))
```

    RBF-kernel SVC training set accuracy: 0.95
    RBF-kernel SVC test set accuracy: 0.83
    

# EVALUATION 


The two models will be evaluated using confusion matrixes. A confusion matrix is a table that compares the predicted and actual labels of a classification model. It shows how many predictions were correct or incorrect and provides insights into the model's performance.

First, let's build the confusion matrix for our KNN model.



```python
KNN_predicted = knn.predict(X_test)
print(confusion_matrix(y_test, KNN_predicted))
print(classification_report(y_test, KNN_predicted))
```

    [[ 0  0  7]
     [ 0  0 16]
     [ 0  0  7]]
                  precision    recall  f1-score   support
    
               2       0.00      0.00      0.00         7
               3       0.00      0.00      0.00        16
               4       0.23      1.00      0.38         7
    
        accuracy                           0.23        30
       macro avg       0.08      0.33      0.13        30
    weighted avg       0.05      0.23      0.09        30
    
    

    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\base.py:443: UserWarning: X has feature names, but KNeighborsClassifier was fitted without feature names
      warnings.warn(
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

Next, we build the confusion matrix for our SVM model.


```python
svm_predicted = SVM.predict(X_test)
print(confusion_matrix(y_test, svm_predicted))
print(classification_report(y_test, svm_predicted))
```

    [[ 0  0  7]
     [ 0  0 16]
     [ 0  0  7]]
                  precision    recall  f1-score   support
    
               2       0.00      0.00      0.00         7
               3       0.00      0.00      0.00        16
               4       0.23      1.00      0.38         7
    
        accuracy                           0.23        30
       macro avg       0.08      0.33      0.13        30
    weighted avg       0.05      0.23      0.09        30
    
    

    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\base.py:443: UserWarning: X has feature names, but SVC was fitted without feature names
      warnings.warn(
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\Aziz\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

We can now choose our model not only based on the accuracy like we did originally but also based on other metrics that can be as important like recall or f-1 score or precision
