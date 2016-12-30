import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv(r'C:\Users\Kwon\Dropbox\TF\Titanic\train.csv')
test = pd.read_csv(r'C:\Users\Kwon\Dropbox\TF\Titanic\test.csv')
full = pd.concat([train,test]).set_index("PassengerId")

print(full.describe())

def split_title(x):
    return (x.split(",")[1].split(".")[0].strip())

full["Title"] = full["Name"].apply(split_title)

title_by_sex = pd.DataFrame(index = full["Title"].drop_duplicates().values)
title_by_sex["Male"] = full[full["Sex"] == "male"]["Title"].value_counts()
title_by_sex["Female"] = full[full["Sex"] == "female"]["Title"].value_counts()
title_by_sex = title_by_sex.fillna(value = 0)
title_by_sex

rare_title = ["Don", "Dona", "Rev", "Dr", "Major", "Lady", "Sir",
              "Col", "Capt", "the Countess", "Jonkheer"]
             
for index, row in full.iterrows():
    if row['Title'] == "Mlle":
        full.set_value(index, 'Title', 'Miss')
    elif row['Title'] == "Ms":
        full.set_value(index, 'Title', 'Miss')
    elif row['Title'] == "Mme":
        full.set_value(index, 'Title', 'Mrs')
    elif row['Title'] in rare_title:
        full.set_value(index, 'Title', 'Rare Title')
        
title_by_sex2 = pd.DataFrame(index = ["Master", "Miss", "Mr", "Mrs", "Rare Title"])
title_by_sex2["Male"] = full[full["Sex"] == "male"]["Title"].value_counts()
title_by_sex2["Female"] = full[full["Sex"] == "female"]["Title"].value_counts()
title_by_sex2

def split_surname(x):
    return (x.split(",")[0])

full["Surname"] = full["Name"].apply(split_surname)

full["Family"] = full["SibSp"] + full["Parch"] + 1

survived = full["Family"][full['Survived'] == 1].value_counts()
died = full["Family"][full['Survived'] == 0].value_counts()
y = pd.concat([survived, died], axis = 1)
y.columns = ['survived', 'died']
family_plot = y.plot.bar()
#family_plot.set_xlabel("Family Size")
#family_plot.set_ylabel("Count")


def assign_size(x):
    if x == 1:
        return 'singleton'
    elif (x < 5) & (x > 1):
        return 'small'
    elif (x > 4):
        return 'large'
        
full["Family_D"] = full["Family"].apply(assign_size)
mosaic(full, ['Family_D', 'Survived'])

full["Cabin"] = full["Cabin"].fillna(0)

def assign_deck(x):
    if x == 0:
        return 'Z'
    else:
        return x[0]
        
full["Deck"] = full["Cabin"].apply(assign_deck)



tmp = full[["Pclass", "Fare"]][full["Embarked"].isnull()]
full[full["Pclass"]==1].groupby(full["Embarked"]).median()
full["Embarked"] = full["Embarked"].fillna("C")
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

full["Age"].dropna().astype(int).plot.hist(bins=70)

for index, row in full.iterrows():
    if np.isnan(row["Age"]):
        rand = np.random.randint(full["Age"].mean() - full["Age"].std(), 
                                 full["Age"].mean() + full["Age"].std())        
        full.set_value(index, "Age", rand)
        
full["Age"].astype(int).plot.hist(bins=70)

def is_child(x):
    if x < 18:
        return 1
    else:
        return 0
   
full["Child"] = full["Age"].apply(is_child)
     
def is_mother(x):
    if (x["Sex"] == "female") & (x["Parch"] > 0) & (x["Age"] > 18) & (x["Title"] != "Miss"):
        return 1
    else:
        return 0
            
full["Mother"] = full.apply(is_mother, axis=1)

full["Fare"] = full["Fare"].fillna(full[(full["Pclass"] == 3) & (full["Embarked"] == "S")]["Fare"].median())

full.set_value(full["Sex"] == "male", "Sex", 0)
full.set_value(full["Sex"] == "female", "Sex", 1)

full.set_value(full["Embarked"] == "S", "Embarked", 0)
full.set_value(full["Embarked"] == "C", "Embarked", 1)
full.set_value(full["Embarked"] == "Q", "Embarked", 2)

full.set_value(full["Title"] == "Mr", "Title", 0)
full.set_value(full["Title"] == "Mrs", "Title", 1)
full.set_value(full["Title"] == "Miss", "Title", 2)
full.set_value(full["Title"] == "Master", "Title", 3)
full.set_value(full["Title"] == "Rare Title", "Title", 4)

train = full[:891]
test = full[891:1310]
predictors = ["Age", "Embarked", "Fare", "Parch", "Pclass", "Sex", "SibSp", "Title", "Family", "Child", "Mother"]
x_train = train[predictors]
y_train = train["Survived"]
x_test= test[predictors]

alg = LogisticRegression(random_state = 1)
scores = cross_validation.cross_val_score(alg, x_train, y_train, cv=3)
print(scores.mean())

alg_2 = RandomForestClassifier(random_state = 1, n_estimators = 150, min_samples_split = 4, min_samples_leaf = 2)
scores_2 = cross_validation.cross_val_score(alg_2, train[predictors], train["Survived"], cv=3)
print(scores_2.mean())
alg_2.fit(x_train, y_train)
predictions = alg_2.predict(x_test)
submission = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions})
submission.to_csv('titanic_submission.csv', index=False)