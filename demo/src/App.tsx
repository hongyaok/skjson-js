import { useState, useEffect } from 'react';
import { loadModel } from 'skjson-js';
import CodePanel from './CodePanel';

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

const FEATURE_RANGES = [
  { min: 4.3, max: 7.9, step: 0.1 },
  { min: 2.0, max: 4.4, step: 0.1 },
  { min: 1.0, max: 6.9, step: 0.1 },
  { min: 0.1, max: 2.5, step: 0.1 }
];

export default function App() {
  const [selectedModelKey, setSelectedModelKey] = useState<string>('Random Forest');
  const [features, setFeatures] = useState<number[]>([5.1, 3.5, 1.4, 0.2]); // Setosa-like defaults
  const [prediction, setPrediction] = useState<string | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [predictor, setPredictor] = useState<any>(null);
  const [isMobileCodeOpen, setIsMobileCodeOpen] = useState(false);

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

  const handleFeatureChange = (index: number, value: number) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  return (
    <div className="app-wrapper">
      {/* Background Video */}
      <div className="bg-video-container">
        <video 
          className="bg-video" 
          src="/bg/bg.mp4" 
          autoPlay 
          muted 
          loop 
          playsInline
        />
      </div>

      {/* Compact Header */}
      <header className="app-header">
        <div className="app-header-left">
          <h1>skjson-js</h1>
          <p>Zero-dependency universal Javascript inference for scikit-learn models</p>
        </div>
        <div className="app-header-links">
          <a href="https://github.com/hongyaok/skjson-js" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub
          </a>
          <a href="https://www.npmjs.com/package/skjson-js" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24">
              <path d="M0 7.396h24v9.208H12v1.562H8.052v-1.562H0V7.396zm19.688 1.563h-1.875v6.083h1.875V8.959zm-3.75 0h-3.75v6.083h1.875v-4.52h1.875v4.52h1.875V8.959zm-7.5 0H4.688v6.083h1.875v-4.52h1.875v4.52H8.44V8.959z"/>
            </svg>
            npm package
          </a>
        </div>
      </header>

      {/* Main Pitch Layout */}
      <main className="main-layout">
        {/* Left Side: Parameters and Inference */}
        <section className="panel">
          <div className="glass-card prediction-panel">
            <div className="scroll-content">
              {/* Model Select */}
              <div className="input-group">
                <label>Select Model</label>
                <div className="model-select-wrapper">
                  <select 
                    value={selectedModelKey} 
                    onChange={e => setSelectedModelKey(e.target.value)}
                    className="model-select"
                  >
                    {Object.keys(MODELS).map(key => (
                      <option key={key} value={key}>
                        {key} ({MODELS[key].meta.model_type})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Range sliders for features */}
              <div className="input-group">
                <label>Parameters (Iris Features)</label>
                <div className="sliders-grid">
                  {features.map((feat, idx) => (
                    <div key={idx} className="feature-slider-group">
                      <div className="feature-slider-info">
                        <label>{FEATURE_NAMES[idx]}</label>
                        <input 
                          type="number" 
                          step={FEATURE_RANGES[idx].step}
                          min={FEATURE_RANGES[idx].min}
                          max={FEATURE_RANGES[idx].max}
                          value={feat}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            handleFeatureChange(idx, isNaN(val) ? 0 : val);
                          }}
                          className="feature-value-input"
                        />
                      </div>
                      <div className="slider-control-row">
                        <span className="slider-min-max">{FEATURE_RANGES[idx].min}</span>
                        <input 
                          type="range"
                          min={FEATURE_RANGES[idx].min}
                          max={FEATURE_RANGES[idx].max}
                          step={FEATURE_RANGES[idx].step}
                          value={feat}
                          onChange={(e) => handleFeatureChange(idx, parseFloat(e.target.value))}
                          className="slider-range-input"
                        />
                        <span className="slider-min-max">{FEATURE_RANGES[idx].max}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Button */}
              <button className="predict-btn" onClick={handlePredict}>
                Run Inference
              </button>

              {/* Output Result Card */}
              {prediction !== null && (
                <div className="result-card">
                  <div className="result-header">Predicted Iris Species</div>
                  <div className="result-species">{prediction}</div>
                  
                  {probabilities && (
                    <div className="probabilities-list">
                      {probabilities.map((prob, idx) => {
                        const className = MODELS[selectedModelKey].meta.classes?.[idx] ?? `Class ${idx}`;
                        const percentage = (prob * 100).toFixed(1);
                        return (
                          <div key={idx} className="prob-row">
                            <span className="prob-label" title={className}>{className}</span>
                            <div className="prob-bar-wrapper">
                              <div 
                                className="prob-bar-fill" 
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                            <span className="prob-percentage">{percentage}%</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Side: Code Displays (Collapsible on mobile) */}
        <section className="panel">
          {/* Desktop static display */}
          <div className="code-panel-container-desktop">
            <CodePanel selectedModelKey={selectedModelKey} features={features} />
          </div>

          {/* Mobile Accordion */}
          <button 
            className={`mobile-code-toggle ${isMobileCodeOpen ? 'open' : ''}`}
            onClick={() => setIsMobileCodeOpen(!isMobileCodeOpen)}
          >
            <span>{isMobileCodeOpen ? 'Hide Integration Code' : 'Show Integration Code'}</span>
            <span className="mobile-code-toggle-arrow">▼</span>
          </button>
          
          <div className={`code-panel-container ${isMobileCodeOpen ? 'open' : ''}`}>
            <CodePanel selectedModelKey={selectedModelKey} features={features} />
          </div>
        </section>
      </main>
    </div>
  );
}
