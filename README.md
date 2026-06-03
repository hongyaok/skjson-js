# skjson-js

`skjson-js` is a zero-dependency, universal Javascript inference library for Python `skjson` exported models. It parses the JSON representation of trained scikit-learn models and accurately executes inference directly in Node.js or any browser environment.

[Live Demo](https://skjson-js.vercel.app/)

## Features

- **Zero dependencies**: No web assembly, no native bindings, pure Javascript/TypeScript.
- **Universal compatibility**: Works effortlessly in browsers and Node.js.
- **100% Mathematical Parity**: Strict test suites guarantee identical predictions compared to Python scikit-learn outcomes.

## Installation

```bash
npm install skjson-js
```

## Supported Models

Currently, `skjson-js` supports inference for the following model types:
- **Linear Models**: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`
- **Trees & Forests**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`
- **Naive Bayes**: `GaussianNB`
- **Support Vector Machines**: `SVC`, `SVR` (with linear kernels)
- **Nearest Neighbors**: `KNeighborsClassifier`, `KNeighborsRegressor`
- **Preprocessing**: `StandardScaler`, `MinMaxScaler`

## Usage Example

First, export your model in Python using [`skjson`](https://github.com/hongyaok/skjson):

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skjson

# Load the Iris dataset and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Export the trained model to JSON
skjson.save(clf, 'model.json')
```

Then, load and predict in Javascript:

```javascript
import { loadModel } from 'skjson-js';
import myModelJson from './model.json'; // Or fetch() in the browser

// Instantiate the predictor
const predictor = loadModel(myModelJson);

// Perform inference
const predictions = predictor.predict([
  [2.3, -1.0, 0.4, 5.0],
  [0.5,  0.2, 0.1, -1.1]
]);
console.log('Predictions:', predictions);

// For classification models, you can also get probabilities
if (predictor.predict_proba) {
  const probas = predictor.predict_proba([
    [2.3, -1.0, 0.4, 5.0]
  ]);
  console.log('Probabilities:', probas);
}
```

## Preprocessing Example

```javascript
import { loadModel } from 'skjson-js';
import scalerJson from './scaler.json';

const scaler = loadModel(scalerJson);

// Transformers use the `.transform()` method instead of predict()
const transformed = scaler.transform([
  [2.3, -1.0, 0.4, 5.0],
  [0.5,  0.2, 0.1, -1.1]
]);
console.log('Transformed:', transformed);
```

## HTML-Only Usage (Browser via CDN)

You can use `skjson-js` directly in plain HTML without any bundlers by utilizing modern ES modules and a CDN like unpkg.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>skjson-js HTML Demo</title>
</head>
<body>
    <h1>Prediction Result: <span id="result">Loading...</span></h1>

    <script type="module">
        // Import directly from a CDN
        import { loadModel } from 'https://unpkg.com/skjson-js/dist/index.js';

        async function run() {
            // Fetch your JSON model from a local path or URL
            const response = await fetch('./model.json');
            const modelJson = await response.json();

            // Load and predict
            const predictor = loadModel(modelJson);
            const preds = predictor.predict([[2.3, -1.0, 0.4, 5.0]]);
            
            document.getElementById('result').innerText = preds[0];
        }

        run();
    </script>
</body>
</html>
```
## Security

This package operates with 0 runtime dependencies to ensure full immunity against supply-chain attacks. CI enforces `npm audit` on all development tools prior to publishing.

## AI use

This project uses AI-assisted development with Gemini and Claude Opus. 


## License

ISC
