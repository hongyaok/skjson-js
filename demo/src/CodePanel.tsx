import { useState } from 'react';

interface CodePanelProps {
  selectedModelKey: string;
  features: number[];
}

export default function CodePanel({ selectedModelKey, features }: CodePanelProps) {
  const [activeTab, setActiveTab] = useState<'js' | 'html' | 'python'>('js');
  const [copied, setCopied] = useState(false);

  const modelFile = selectedModelKey.replace(/ /g, '');

  const getJSPlainText = () => {
    return `import { loadModel } from 'skjson-js';
import modelJson from './models/${modelFile}.json';

const predictor = loadModel(modelJson);

// Running inference
const input = [${features.join(', ')}];
const prediction = predictor.predict([input]);

console.log("Predicted Class:", prediction[0]);`;
  };

  const getHTMLPlainText = () => {
    return `<!-- Import directly from CDN in plain HTML -->
<script type="module">
  import { loadModel } from 'https://unpkg.com/skjson-js/dist/index.js';

  async function run() {
    const response = await fetch('./models/${modelFile}.json');
    const modelJson = await response.json();
    const predictor = loadModel(modelJson);

    // Run inference on current inputs
    const input = [${features.join(', ')}];
    const prediction = predictor.predict([input]);
    document.getElementById('result').innerText = prediction[0];
  }
  run();
</script>`;
  };

  const getPythonPlainText = () => {
    let pyImport = '';
    let pyInstantiate = '';
    
    if (selectedModelKey === 'Random Forest') {
      pyImport = "from sklearn.ensemble import RandomForestClassifier";
      pyInstantiate = "clf = RandomForestClassifier(random_state=42)";
    } else if (selectedModelKey === 'Logistic Regression') {
      pyImport = "from sklearn.linear_model import LogisticRegression";
      pyInstantiate = "clf = LogisticRegression(max_iter=200, random_state=42)";
    } else if (selectedModelKey === 'Linear SVC') {
      pyImport = "from sklearn.svm import LinearSVC";
      pyInstantiate = "clf = LinearSVC(dual=False, random_state=42)";
    } else if (selectedModelKey === 'K-Nearest Neighbors') {
      pyImport = "from sklearn.neighbors import KNeighborsClassifier";
      pyInstantiate = "clf = KNeighborsClassifier(n_neighbors=5)";
    } else {
      pyImport = "from sklearn.naive_bayes import GaussianNB";
      pyInstantiate = "clf = GaussianNB()";
    }

    return `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
${pyImport}
import skjson

# Load the Iris dataset and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
${pyInstantiate}
clf.fit(X_train, y_train)

# Export the trained model to JSON
skjson.save(clf, '${modelFile}.json')`;
  };

  const handleCopy = async () => {
    let textToCopy = '';
    if (activeTab === 'js') textToCopy = getJSPlainText();
    else if (activeTab === 'html') textToCopy = getHTMLPlainText();
    else if (activeTab === 'python') textToCopy = getPythonPlainText();

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const renderJSCode = () => {
    return (
      <>
        <span className="code-keyword">import</span> {'{ loadModel }'} <span className="code-keyword">from</span> <span className="code-string">'skjson-js'</span>;<br />
        <span className="code-keyword">import</span> modelJson <span className="code-keyword">from</span> <span className="code-string">'./models/{modelFile}.json'</span>;<br />
        <br />
        <span className="code-keyword">const</span> predictor = <span className="code-function">loadModel</span>(modelJson);<br />
        <br />
        <span className="code-comment">// Running inference</span><br />
        <span className="code-keyword">const</span> input = [<span className="code-number">{features.join(', ')}</span>];<br />
        <span className="code-keyword">const</span> prediction = predictor.<span className="code-function">predict</span>([input]);<br />
        <br />
        console.<span className="code-function">log</span>(<span className="code-string">"Predicted Class:"</span>, prediction[<span className="code-number">0</span>]);
      </>
    );
  };

  const renderHTMLCode = () => {
    return (
      <>
        <span className="code-comment">&lt;!-- Import directly from CDN in plain HTML --&gt;</span><br />
        &lt;<span className="code-keyword">script</span> <span className="code-variable">type</span>=<span className="code-string">"module"</span>&gt;<br />
        &nbsp;&nbsp;<span className="code-keyword">import</span> {'{ loadModel }'} <span className="code-keyword">from</span> <span className="code-string">'https://unpkg.com/skjson-js/dist/index.js'</span>;<br />
        <br />
        &nbsp;&nbsp;<span className="code-keyword">async</span> <span className="code-keyword">function</span> <span className="code-function">run</span>() {'{'}<br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-keyword">const</span> response = <span className="code-keyword">await</span> <span className="code-function">fetch</span>(<span className="code-string">'./models/{modelFile}.json'</span>);<br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-keyword">const</span> modelJson = <span className="code-keyword">await</span> response.<span className="code-function">json</span>();<br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-keyword">const</span> predictor = <span className="code-function">loadModel</span>(modelJson);<br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-comment">// Run inference on current inputs</span><br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-keyword">const</span> input = [<span className="code-number">{features.join(', ')}</span>];<br />
        &nbsp;&nbsp;&nbsp;&nbsp;<span className="code-keyword">const</span> prediction = predictor.<span className="code-function">predict</span>([input]);<br />
        &nbsp;&nbsp;&nbsp;&nbsp;document.<span className="code-function">getElementById</span>(<span className="code-string">'result'</span>).innerText = prediction[<span className="code-number">0</span>];<br />
        &nbsp;&nbsp;{'}'}<br />
        &nbsp;&nbsp;<span className="code-function">run</span>();<br />
        &lt;/<span className="code-keyword">script</span>&gt;
      </>
    );
  };

  const renderPythonCode = () => {
    let pyImport = '';
    let pyInstantiate = '';
    
    if (selectedModelKey === 'Random Forest') {
      pyImport = "from sklearn.ensemble import RandomForestClassifier";
      pyInstantiate = "clf = RandomForestClassifier(random_state=42)";
    } else if (selectedModelKey === 'Logistic Regression') {
      pyImport = "from sklearn.linear_model import LogisticRegression";
      pyInstantiate = "clf = LogisticRegression(max_iter=200, random_state=42)";
    } else if (selectedModelKey === 'Linear SVC') {
      pyImport = "from sklearn.svm import LinearSVC";
      pyInstantiate = "clf = LinearSVC(dual=False, random_state=42)";
    } else if (selectedModelKey === 'K-Nearest Neighbors') {
      pyImport = "from sklearn.neighbors import KNeighborsClassifier";
      pyInstantiate = "clf = KNeighborsClassifier(n_neighbors=5)";
    } else {
      pyImport = "from sklearn.naive_bayes import GaussianNB";
      pyInstantiate = "clf = GaussianNB()";
    }

    return (
      <>
        <span className="code-keyword">from</span> sklearn.datasets <span className="code-keyword">import</span> load_iris<br />
        <span className="code-keyword">from</span> sklearn.model_selection <span className="code-keyword">import</span> train_test_split<br />
        <span className="code-keyword">{pyImport}</span><br />
        <span className="code-keyword">import</span> skjson<br />
        <br />
        <span className="code-comment"># Load the Iris dataset and split</span><br />
        X, y = <span className="code-function">load_iris</span>(return_X_y=<span className="code-keyword">True</span>)<br />
        X_train, X_test, y_train, y_test = <span className="code-function">train_test_split</span>(X, y, test_size=<span className="code-number">0.2</span>, random_state=<span className="code-number">42</span>)<br />
        <br />
        <span className="code-comment"># Train a classifier</span><br />
        {pyInstantiate}<br />
        clf.<span className="code-function">fit</span>(X_train, y_train)<br />
        <br />
        <span className="code-comment"># Export the trained model to JSON</span><br />
        skjson.<span className="code-function">save</span>(clf, <span className="code-string">'{modelFile}.json'</span>)
      </>
    );
  };

  return (
    <div className="glass-card">
      <div className="code-panel-header">
        <div className="tabs-wrapper">
          <button 
            className={`tab-btn ${activeTab === 'js' ? 'active' : ''}`}
            onClick={() => setActiveTab('js')}
          >
            JavaScript
          </button>
          <button 
            className={`tab-btn ${activeTab === 'html' ? 'active' : ''}`}
            onClick={() => setActiveTab('html')}
          >
            HTML (CDN)
          </button>
          <button 
            className={`tab-btn ${activeTab === 'python' ? 'active' : ''}`}
            onClick={() => setActiveTab('python')}
          >
            Python Export
          </button>
        </div>
        
        <button className="copy-btn" onClick={handleCopy} title="Copy code to clipboard">
          {copied ? (
            <>
              <svg viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
              </svg>
              Copied!
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24">
                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
              </svg>
              Copy
            </>
          )}
        </button>
      </div>

      <div className="code-container">
        <pre>
          <code>
            {activeTab === 'js' && renderJSCode()}
            {activeTab === 'html' && renderHTMLCode()}
            {activeTab === 'python' && renderPythonCode()}
          </code>
        </pre>
      </div>
    </div>
  );
}
