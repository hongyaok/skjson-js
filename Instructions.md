# skjson-js: Development & Publishing Instructions

This document provides a highly detailed, step-by-step roadmap to building out `skjson-js`. The goal is to build a vulnerability-free NPM package (published automatically via GitHub Actions) capable of performing inference for **ANY** model exported by the Python `skjson` library.

## 1. Security & Core Principles

To ensure this package remains immune to supply-chain attacks and runtime vulnerabilities:
- **Zero Runtime Dependencies:** `dependencies` in `package.json` must remain empty. All internal logic must use Vanilla Javascript/TypeScript.
- **Strict Audits:** CI must run `npm audit` and automatically fail the build if vulnerabilities are found in the `devDependencies`.
- **Dependabot:** Enable GitHub Dependabot to automatically keep tooling (TypeScript, bundlers, testing frameworks) secure and up-to-date.
- **Language:** The language used in the inference MUST BE PURELY javascript, so that the package works without ANY web assembly etc.

## 2. Project Setup

Initialize your application with modern, secure tooling.

```bash
mkdir skjson-js && cd skjson-js
npm init -y
# Install only strictly necessary DEV dependencies
npm install -D typescript tsup vitest eslint prettier
```

Update `package.json` with scripts:
```json
{
  "name": "skjson-js",
  "version": "1.0.0",
  "main": "./dist/index.js",
  "module": "./dist/index.mjs",
  "types": "./dist/index.d.ts",
  "scripts": {
    "build": "tsup src/index.ts --format cjs,esm --dts",
    "test": "vitest run",
    "lint": "eslint src/ --ext .ts",
    "audit:ci": "npm audit --audit-level=high"
  }
}
```

## 3. Test-Driven Development Strategy

Before writing any core inference logic, **start by creating exhaustive test cases**. This test-driven approach guarantees that your JS package resolves exact mathematical equivalence with the Python `skjson` outputs for all supported models.

### 3.1 Python Fixture Generation
1. In the Python `skjson` library, write a script to instantiate, train, and export one of **every supported model** (Random Forest, Linear, SVM, etc.) into JSON objects.
2. Feed a standardized dummy input set into the Python models and record the exact predictions (and probabilities, if applicable) to an `expected_outputs.json` file.
3. Place these output files and `model.json` exports into a `tests/fixtures/` directory inside your JS project.

### 3.2 Vitest Implementation
Create unit tests validating each target model format against its expected Python output using Vitest.

```typescript
// tests/inference.test.ts
import { describe, it, expect } from 'vitest';
import { loadModel } from '../src/index';
import rfModel from './fixtures/rf_model.json';
import rfData from './fixtures/rf_outputs.json';

describe('Model Inference Parity', () => {
  it('RandomForestClassifier matches Python exact outputs', () => {
    const predictor = loadModel(rfModel);
    
    rfData.inputs.forEach((input, index) => {
        const result = predictor.predict(input);
        // Use closeTo for floating point variance where necessary
        expect(result).toBe(rfData.predictions[index]); 
    });
  });

  // Replicate for Linear, SVM, NaiveBayes, Neighbors...
});
```

These suites ensure complete inference validity and are fully integrated into the security & publishing workflow below.

## 4. Universal Inference Implementation

`skjson-js` must seamlessly route standard outputs from your python package's serializers (Linear, SVM, Naive Bayes, Neighbors, Trees, Preprocessing) into JS functions.

### Architecture
Design a Factory pattern that parses the incoming parsed JSON object and instantiates the correct behavior based on `meta.model_type`.

```typescript
// src/index.ts
export function loadModel(modelJson: any): Predictor {
    if (modelJson?.meta?.library !== 'skjson') {
        throw new Error('Invalid skjson format');
    }

    switch (modelJson.meta.model_type) {
        case 'RandomForestClassifier':
            return new RandomForestClassifier(modelJson.params, modelJson.meta);
        case 'LinearRegression':
            return new LinearRegression(modelJson.params, modelJson.meta);
        // Implement others accordingly...
        default:
            throw new Error(`Model type ${modelJson.meta.model_type} is currently not supported.`);
    }
}
```

### 4.1 Trees & Forests (`tree.py`)
- **Model Types:** `DecisionTreeClassifier`, `RandomForestClassifier`, `DecisionTreeRegressor`, `RandomForestRegressor`.
- **Inference logic:**
  - Implement a recursive traversal logic comparing the input array at `input[feature]` against `threshold`.
  - For a **Forest** (like your `demo.json`), map over the `params.trees` array, aggregating results from individual Decision Trees and executing majority voting (classification) or averaging (regression).

### 4.2 Linear Models (`linear.py`)
- **Model Types:** `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`.
- **Inference logic:**
  - Multiply input features against `params.coef` (weights) array and add `params.intercept`.
  - For `LogisticRegression`, feed the output through the Sigmoid activation function: `1 / (1 + Math.exp(-x))`. Return `argmax` on the class probabilities for classifications.

### 4.3 Naive Bayes (`naive_bayes.py`)
- **Model Types:** `GaussianNB`, `MultinomialNB`.
- **Inference logic:** 
  - Evaluate probabilities using the class priors (`class_prior_`) and Gaussian densities parameterized via `theta_` (mean) and `var_` (variance).

### 4.4 Support Vector Machines (`svm.py`)
- **Model Types:** `SVC`, `SVR`.
- **Inference logic:**
  - Store the support vectors (`support_vectors_` ) and `dual_coef_` supplied by `skjson`.
  - In Javascript, apply the respective kernel function (e.g., RBF: `exp(-gamma * ||x - y||^2)`) across support vectors, sum with dual coefficients, and adjust by `intercept_`.

### 4.5 Nearest Neighbors (`neighbors.py`)
- **Model Types:** `KNeighborsClassifier`, `KNeighborsRegressor`.
- **Inference logic:**
  - Extract the exact `fit_data` saved down by `skjson`.
  - In Javascript, calculate Euclidean distances between the sample inputs and all entries in `fit_data`. Sort distances in ascending order, taking the top-K.

### 4.6 Transformers / Preprocessing (`preprocessing.py`)
- **Model Types:** `StandardScaler`, `MinMaxScaler`.
- **Transformation logic:** Provide a `.transform(inputs)` method.
  - `StandardScaler`: `(x - mean) / scale`
  - `MinMaxScaler`: `(x - min) / (max - min)`

## 5. GitHub Actions: Automated Security & Publishing

We will securely handle releasing to the NPM package registry strictly via Github Actions to verify audits and tests independently. We will add Continuous Integration (CI) test triggers on Pull Requests to enforce all test parity checks prior to publishing.

Write the workflow in `.github/workflows/publish.yml`:

```yaml
name: Publish skjson-js to NPM

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  release:
    types: [published]

jobs:
  test_and_audit:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Clean Install Defaults
        run: npm ci

      - name: Vulnerability Audit
        run: npm run audit:ci
        
      - name: Unit & Integration Tests (TDD Enforcement)
        run: npm run test

      - name: Build Package
        run: npm run build

  publish:
    needs: test_and_audit
    if: github.event_name == 'release' && github.event.action == 'published'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'

      - name: Clean Install Defaults
        run: npm ci

      - name: Build Package
        run: npm run build

      - name: Publish Package with Provenance
        run: npm publish --provenance --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

### Steps to trigger this Action:
1. Ensure your Javascript package operates purely without `dependencies`.
2. In NPM, navigate to **Access Tokens** -> Generate a new "Automation" token.
3. In your `skjson-js` GitHub Repository, go to **Settings > Secrets and variables > Actions > New repository secret**. Name it `NPM_TOKEN` and paste your NPM token.
4. When you open a **Pull Request** or commit to `main`, the test matrices (covering every model variant against its expected outputs) will execute, preventing broken inference states.
5. When you draft a **New GitHub Release**, the `publish` job will execute on top of the CI workflow, creating standard Javascript packages and uploading `skjson-js` securely to NPM.

## 6. Next Steps
1. Execute the Python script generation detailed in `Section 3.1` to prep standard Python inference outcomes for tests.
2. Draft up Vitest specs loading `tests/fixtures/rf_model.json` to start TDD on Random Forests!
\n\n...