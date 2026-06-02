import sys
import os
import json

# Add skjson to path so we can import it
sys.path.append(r"c:\Users\hongy\Desktop\skjson\src")
import skjson

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

OUTPUT_DIR = r"c:\Users\hongy\Desktop\skjson-js\demo\src\models"

def save_model(name, model, feature_names, class_names):
    model_json = skjson.serialize_model(model, feature_names=feature_names)
    # the schema expects `classes` in meta, but skjson might not serialize string classes natively in the meta depending on version
    # Let's enforce class names in meta
    if 'meta' in model_json:
        model_json['meta']['classes'] = class_names
        
    with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
        json.dump(model_json, f, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = list(iris.target_names)
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    save_model("RandomForest", rf, feature_names, class_names)
    
    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
    save_model("LogisticRegression", lr, feature_names, class_names)
    
    # 3. LinearSVC
    from sklearn.svm import LinearSVC
    svc = LinearSVC(random_state=42, dual=False).fit(X, y)
    save_model("SVC", svc, feature_names, class_names)
    
    # 4. KNN
    knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    save_model("KNN", knn, feature_names, class_names)
    
    # 5. Gaussian NB
    gnb = GaussianNB().fit(X, y)
    save_model("GaussianNB", gnb, feature_names, class_names)

if __name__ == "__main__":
    main()
