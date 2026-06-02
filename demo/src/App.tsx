import { useState, useEffect } from 'react';
import { loadModel } from 'skjson-js';
import './index.css';

import rfModelJson from './models/RandomForest.json';
import lrModelJson from './models/LogisticRegression.json';
import svcModelJson from './models/SVC.json';
import knnModelJson from './models/KNN.json';
import gnbModelJson from './models/GaussianNB.json';

const MODELS: Record<string, any> = {
  'Random Forest': rfModelJson,
  'Logistic Regression': lrModelJson,
  'Linear SVC': svcModelJson,
  'K-Nearest Neighbors': knnModelJson,
  'Gaussian Naive Bayes': gnbModelJson
};

const FEATURE_NAMES = [
  "Sepal Length (cm)",
  "Sepal Width (cm)",
  "Petal Length (cm)",
  "Petal Width (cm)"
];

function App() {
  const [selectedModelKey, setSelectedModelKey] = useState<string>('Random Forest');
  const [features, setFeatures] = useState<number[]>([5.1, 3.5, 1.4, 0.2]); // Setosa-like defaults
  const [prediction, setPrediction] = useState<string | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [predictor, setPredictor] = useState<any>(null);

  useEffect(() => {
    try {
      const modelJson = MODELS[selectedModelKey];
      const model = loadModel(modelJson);
      setPredictor(model);
      
      // Clear old predictions
      setPrediction(null);
      setProbabilities(null);
    } catch (e) {
      console.error("Error loading model:", e);
    }
  }, [selectedModelKey]);

  const handlePredict = () => {
    if (!predictor) return;
    
    const preds = predictor.predict([features]);
    setPrediction(preds[0]);
    
    if (predictor.predict_proba) {
      const probas = predictor.predict_proba([features]);
      setProbabilities(probas[0]);
    } else {
      setProbabilities(null);
    }
  };

  const handleFeatureChange = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value) || 0;
    setFeatures(newFeatures);
  };

  const codeSnippet = `import { loadModel } from 'skjson-js';
import modelJson from './models/${selectedModelKey.replace(/ /g, '')}.json';

const predictor = loadModel(modelJson);

// Running inference
const input = [${features.join(', ')}];
const prediction = predictor.predict([input]);

console.log("Predicted Class:", prediction[0]);`;

  return (
    <div className="app-container">
      <div className="glass-card">
        <header className="header">
          <h1>Iris Dataset Inference</h1>
          <p>Powered by <strong>skjson-js</strong> with 0 runtime dependencies.</p>
          <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <a href="https://github.com/hongyaok/skjson-js" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', textDecoration: 'none', fontWeight: 600 }}>GitHub Repo</a>
            <a href="https://www.npmjs.com/package/skjson-js" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', textDecoration: 'none', fontWeight: 600 }}>NPM Package</a>
          </div>
        </header>

        <main>
          <div className="input-group" style={{ marginBottom: '2rem' }}>
            <label>Select Model</label>
            <select 
              value={selectedModelKey} 
              onChange={e => setSelectedModelKey(e.target.value)}
              className="model-select"
            >
              {Object.keys(MODELS).map(key => (
                <option key={key} value={key}>{key} ({MODELS[key].meta.model_type})</option>
              ))}
            </select>
          </div>

          <div className="input-grid">
            {features.map((feat, idx) => (
              <div key={idx} className="input-group">
                <label>{FEATURE_NAMES[idx]}</label>
                <input 
                  type="number" 
                  step="0.1" 
                  value={feat} 
                  onChange={(e) => handleFeatureChange(idx, e.target.value)} 
                />
              </div>
            ))}
          </div>

          <button className="predict-btn" onClick={handlePredict}>
            Run Inference
          </button>

          {prediction !== null && (
            <div className="result-container">
              <h2>Predicted Iris Species</h2>
              <div className="prediction-value">{prediction}</div>
              
              {probabilities && (
                <div className="probabilities">
                  {probabilities.map((prob, idx) => (
                    <div key={idx} className="prob-item">
                      <span className="prob-label">{MODELS[selectedModelKey].meta.classes?.[idx] ?? `Class ${idx}`}</span>
                      <span className="prob-val">{(prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="code-snippet-container">
            <h3>Usage Code Snippet</h3>
            <pre><code>{codeSnippet}</code></pre>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
