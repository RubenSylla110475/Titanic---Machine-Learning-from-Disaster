import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings("ignore")

# **1. Load Data**
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# Preserve PassengerId for submission
passenger_ids = test_df['PassengerId']

combine = [train_df, test_df]

# **2. Feature Engineering**
for dataset in combine:
    # Extract Title from Name
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Jonkheer', 'Dona'], 'Lady')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Don', 'Major', 'Sir'], 'Sir')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Lady": 5, "Sir": 6, "Dr": 7, "Col": 8, "Rev": 9}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Encode categorical variables
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Handle missing values and create new features
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['Fare'] = np.log1p(dataset['Fare'])  # Log transform to handle skewness
    dataset['Age'] = dataset['Age'].fillna(dataset.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=[1, 2, 3, 4]).astype(int)
    dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 12, 18, 35, 60, 80], labels=[1, 2, 3, 4, 5]).astype(int)

# Drop unnecessary columns
train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df[X_train.columns]

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# **3. Feature Selection**
# Use RFECV to select the most important features
rf = RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

selected_features = X_train.columns[rfecv.support_]
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# **4. Hyperparameter Tuning**
# Tune Random Forest
rf_params = {'n_estimators': [200, 300], 'max_depth': [6, 8, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Tune Gradient Boosting
gb_params = {'n_estimators': [200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='accuracy')
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

# **5. Stacking Classifier**
base_learners = [
    ('rf', best_rf),
    ('gb', best_gb),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))
]

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=500),
    cv=5
)

# **6. Cross-Validation**
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Cross-validated accuracy: {np.mean(cv_scores):.4f}")

# Fit the stacking classifier
stacking_clf.fit(X_train, y_train)

# **7. Prediction and Submission**
y_pred = stacking_clf.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': y_pred
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved.")
