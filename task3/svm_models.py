from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm_linear(X_train, X_test, y_train, y_test):
    model = SVC(kernel="linear", random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


def train_svm_rbf(X_train, X_test, y_train, y_test):
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc