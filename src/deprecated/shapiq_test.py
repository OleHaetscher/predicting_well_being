import shapiq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X, y = shapiq.load_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.25, random_state=42
)
n_features = X_train.shape[1]

model = RandomForestRegressor(
    n_estimators=100, max_depth=n_features, max_features=2 / 3, max_samples=2 / 3, random_state=42
)
model.fit(X_train, y_train)
print("Train R2: {:.4f}".format(model.score(X_train, y_train)))
print("Test  R2: {:.4f}".format(model.score(X_test, y_test)))

explainer = shapiq.TreeExplainer(model=model, index="k-SII", min_order=1, max_order=3)
x = X_test[24]
interaction_values = explainer.explain(x)
print(interaction_values)
