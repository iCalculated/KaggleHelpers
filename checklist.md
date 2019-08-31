---
geometry: margin=2cm
---

# Competition Checklist

## Loading and Checking Data

- `pd.read_csv(".../input/*.csv")`{.python} 
- `test['Id']`{.python}
- Check for major outliers (helpers/outliers.py)
- Concat datasets for feature analysis: `pd.concat(objs=[train, test], axis=0).reset_index(drop=True)`{.python} 
- `train_len = len(train)`{.python} for better separation later
- **Many issues can be caught with pandas_profiling**
- Homogenize nulls: `df.fillna(np.nan)`{.python}
- `dataset.isnull().sum()`{.python}

## Graphing and Feature Analysis

- I'll fill this out next time I do it, currently can't access data 

## Missing Values and Cleanup

- Direct map: `.map({"type1": 0, "type2": 1})`{.python}
- Dropping: `.drop(labels=["column"], axis=1, inplace=True")`{.python}
  - Notably inplace means there will be no return
- One-Hot: `pd.get_dummies(dataset, colmns = ["column"], prefix="optional")`{.python}

## Modeling

- Separate `train = dataset[:train_len]`{.python} and 'test = dataset[train_len:]'
- Remember to drop target in train
- Classifier ensembles are nicer if you make a list:

```python
random_state = 0 
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
```

- Why list: 

```python
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
```

- Check [this](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling) notebook for cool things to implement
- Tune hyperparams, grid search is your friend but not your best friend
- Plot learning curves (helpers/curves)
- Ensemble (Ghouzam): 

```python
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)
```

- Pass best estimators in voting 
