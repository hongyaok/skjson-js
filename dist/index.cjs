'use strict';

// src/models/linear.ts
var LinearModel = class {
  coef;
  intercept;
  meta;
  constructor(params, meta) {
    this.coef = params.coef;
    this.intercept = params.intercept;
    this.meta = meta;
  }
  dotProduct(input, weights) {
    return input.reduce((sum, val, i) => sum + val * weights[i], 0);
  }
  predict(inputs) {
    const isMultiClass = Array.isArray(this.intercept);
    return inputs.map((input) => {
      if (isMultiClass) {
        const numOutputs = this.intercept.length;
        const outputs = [];
        for (let i = 0; i < numOutputs; i++) {
          const w = this.coef[i];
          outputs.push(this.dotProduct(input, w) + this.intercept[i]);
        }
        return this.postProcess(outputs);
      } else {
        const output = this.dotProduct(input, this.coef) + this.intercept;
        return this.postProcessSingle(output);
      }
    });
  }
  postProcess(outputs) {
    return outputs;
  }
  postProcessSingle(output) {
    return output;
  }
};
var LinearRegression = class extends LinearModel {
};
var Ridge = class extends LinearModel {
};
var Lasso = class extends LinearModel {
};
var LogisticRegression = class extends LinearModel {
  predict_proba(inputs) {
    const isMultiClass = this.meta.classes && this.meta.classes.length > 2;
    return inputs.map((input) => {
      if (isMultiClass) {
        const numOutputs = this.intercept.length;
        const outputs = [];
        for (let i = 0; i < numOutputs; i++) {
          const w = this.coef[i];
          outputs.push(this.dotProduct(input, w) + this.intercept[i]);
        }
        const maxVal = Math.max(...outputs);
        let sumExp = 0;
        const exps = outputs.map((o) => {
          const e = Math.exp(o - maxVal);
          sumExp += e;
          return e;
        });
        return exps.map((e) => e / sumExp);
      } else {
        let output = 0;
        if (Array.isArray(this.coef[0])) {
          output = this.dotProduct(input, this.coef[0]) + (Array.isArray(this.intercept) ? this.intercept[0] : this.intercept);
        } else {
          output = this.dotProduct(input, this.coef) + this.intercept;
        }
        const prob = 1 / (1 + Math.exp(-output));
        return [1 - prob, prob];
      }
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.meta.classes ? this.meta.classes[maxIdx] : maxIdx;
    });
  }
};

// src/models/tree.ts
var TreeModel = class {
  meta;
  constructor(meta) {
    this.meta = meta;
  }
  traverseTree(nodes, input) {
    let node = nodes[0];
    while (node.left !== -1) {
      const val = input[node.feature];
      if (Number.isNaN(val)) {
        if (node.missing_go_to_left) {
          node = nodes[node.left];
        } else {
          node = nodes[node.right];
        }
      } else if (val <= node.threshold) {
        node = nodes[node.left];
      } else {
        node = nodes[node.right];
      }
    }
    return node.value;
  }
  predict(inputs) {
    throw new Error("Not implemented");
  }
};
var DecisionTreeClassifier = class extends TreeModel {
  nodes;
  classes;
  constructor(params, meta) {
    super(meta);
    this.nodes = params.nodes;
    this.classes = params.classes || meta.classes;
  }
  predict_proba(inputs) {
    return inputs.map((input) => {
      const value = this.traverseTree(this.nodes, input);
      const total = value.reduce((a, b) => a + b, 0);
      return total > 0 ? value.map((v) => v / total) : value;
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.classes ? this.classes[maxIdx] : maxIdx;
    });
  }
};
var DecisionTreeRegressor = class extends TreeModel {
  nodes;
  constructor(params, meta) {
    super(meta);
    this.nodes = params.nodes;
  }
  predict(inputs) {
    return inputs.map((input) => {
      const val = this.traverseTree(this.nodes, input);
      return Array.isArray(val) ? val[0] : val;
    });
  }
};
var RandomForestClassifier = class extends TreeModel {
  trees;
  classes;
  constructor(params, meta) {
    super(meta);
    this.trees = params.trees;
    this.classes = params.classes || meta.classes;
  }
  predict_proba(inputs) {
    return inputs.map((input) => {
      const numClasses = this.classes ? this.classes.length : this.trees[0][0].value.length;
      const avgProbas = new Array(numClasses).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        const value = this.traverseTree(this.trees[i], input);
        const total = value.reduce((a, b) => a + b, 0);
        const probas = total > 0 ? value.map((v) => v / total) : value;
        for (let j = 0; j < numClasses; j++) {
          avgProbas[j] += probas[j];
        }
      }
      for (let j = 0; j < numClasses; j++) {
        avgProbas[j] /= this.trees.length;
      }
      return avgProbas;
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.classes ? this.classes[maxIdx] : maxIdx;
    });
  }
};
var RandomForestRegressor = class extends TreeModel {
  trees;
  constructor(params, meta) {
    super(meta);
    this.trees = params.trees;
  }
  predict(inputs) {
    return inputs.map((input) => {
      let sum = 0;
      for (let i = 0; i < this.trees.length; i++) {
        const val = this.traverseTree(this.trees[i], input);
        sum += Array.isArray(val) ? val[0] : val;
      }
      return sum / this.trees.length;
    });
  }
};
var GradientBoostingClassifier = class extends TreeModel {
  trees;
  learningRate;
  initRaw;
  classes;
  nTreeOutputs;
  constructor(params, meta) {
    super(meta);
    this.trees = params.trees;
    this.learningRate = params.learning_rate;
    this.initRaw = params.init_raw_predictions;
    this.classes = params.classes || meta.classes;
    this.nTreeOutputs = params.n_tree_outputs;
  }
  predict_proba(inputs) {
    return inputs.map((input) => {
      const raw = [...this.initRaw];
      for (const stageTrees of this.trees) {
        for (let k = 0; k < stageTrees.length; k++) {
          const val = this.traverseTree(stageTrees[k], input);
          const leafValue = Array.isArray(val) ? val[0] : val;
          raw[k] += this.learningRate * leafValue;
        }
      }
      if (this.nTreeOutputs === 1) {
        const p1 = 1 / (1 + Math.exp(-raw[0]));
        return [1 - p1, p1];
      } else {
        const maxRaw = Math.max(...raw);
        const expRaw = raw.map((r) => Math.exp(r - maxRaw));
        const sumExp = expRaw.reduce((a, b) => a + b, 0);
        return expRaw.map((e) => e / sumExp);
      }
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.classes ? this.classes[maxIdx] : maxIdx;
    });
  }
};
var GradientBoostingRegressor = class extends TreeModel {
  trees;
  learningRate;
  initRaw;
  constructor(params, meta) {
    super(meta);
    this.trees = params.trees;
    this.learningRate = params.learning_rate;
    this.initRaw = params.init_raw_predictions;
  }
  predict(inputs) {
    return inputs.map((input) => {
      const raw = [...this.initRaw];
      for (const stageTrees of this.trees) {
        for (let k = 0; k < stageTrees.length; k++) {
          const val = this.traverseTree(stageTrees[k], input);
          const leafValue = Array.isArray(val) ? val[0] : val;
          raw[k] += this.learningRate * leafValue;
        }
      }
      return raw[0];
    });
  }
};

// src/models/naive_bayes.ts
var GaussianNB = class {
  theta;
  var;
  class_prior;
  classes;
  meta;
  constructor(params, meta) {
    this.theta = params.theta;
    this.var = params.var;
    this.class_prior = params.class_prior;
    this.classes = params.classes || meta.classes;
    this.meta = meta;
  }
  calculateLogProb(x, mean, var_) {
    let logProb = 0;
    for (let i = 0; i < x.length; i++) {
      logProb -= 0.5 * Math.log(2 * Math.PI * var_[i]);
      logProb -= Math.pow(x[i] - mean[i], 2) / (2 * var_[i]);
    }
    return logProb;
  }
  predict_proba(inputs) {
    return inputs.map((input) => {
      const logProbas = this.classes.map((_, i) => {
        return Math.log(this.class_prior[i]) + this.calculateLogProb(input, this.theta[i], this.var[i]);
      });
      const maxLogProb = Math.max(...logProbas);
      let sumExp = 0;
      for (let i = 0; i < logProbas.length; i++) {
        sumExp += Math.exp(logProbas[i] - maxLogProb);
      }
      const probas = logProbas.map((lp) => Math.exp(lp - maxLogProb) / sumExp);
      return probas;
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
};

// src/models/svm.ts
var SVC = class extends LinearModel {
  predict(inputs) {
    const isMultiClass = Array.isArray(this.intercept) && this.intercept.length > 1;
    return inputs.map((input) => {
      if (isMultiClass) {
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < this.intercept.length; i++) {
          const w = this.coef[i];
          const val = this.dotProduct(input, w) + this.intercept[i];
          if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
          }
        }
        return this.meta.classes ? this.meta.classes[maxIdx] : maxIdx;
      } else {
        const coef = Array.isArray(this.coef[0]) ? this.coef[0] : this.coef;
        const intercept = Array.isArray(this.intercept) ? this.intercept[0] : this.intercept;
        const output = this.dotProduct(input, coef) + intercept;
        const idx = output > 0 ? 1 : 0;
        return this.meta.classes ? this.meta.classes[idx] : idx;
      }
    });
  }
};
var SVR = class extends LinearModel {
  predict(inputs) {
    return inputs.map((input) => {
      const coef = Array.isArray(this.coef[0]) ? this.coef[0] : this.coef;
      const intercept = Array.isArray(this.intercept) ? this.intercept[0] : this.intercept;
      return this.dotProduct(input, coef) + intercept;
    });
  }
};

// src/models/neighbors.ts
var KNeighborsModel = class {
  training_data;
  n_neighbors;
  metric;
  meta;
  constructor(params, meta) {
    this.training_data = params.training_data;
    this.n_neighbors = params.n_neighbors;
    this.metric = params.metric;
    this.meta = meta;
  }
  getDistances(input) {
    const distances = this.training_data.map((row, index) => {
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        const diff = input[i] - row[i];
        sum += diff * diff;
      }
      return { distance: Math.sqrt(sum), index };
    });
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, this.n_neighbors);
  }
};
var KNeighborsClassifier = class extends KNeighborsModel {
  training_labels;
  classes;
  constructor(params, meta) {
    super(params, meta);
    this.training_labels = params.training_labels;
    this.classes = params.classes || meta.classes;
  }
  predict_proba(inputs) {
    return inputs.map((input) => {
      const nearest = this.getDistances(input);
      const probas = new Array(this.classes.length).fill(0);
      for (let i = 0; i < nearest.length; i++) {
        const label = this.training_labels[nearest[i].index];
        const classIdx = this.classes.indexOf(label);
        if (classIdx !== -1) {
          probas[classIdx] += 1;
        }
      }
      for (let i = 0; i < probas.length; i++) {
        probas[i] /= this.n_neighbors;
      }
      return probas;
    });
  }
  predict(inputs) {
    const probas = this.predict_proba(inputs);
    return probas.map((p) => {
      let maxIdx = 0;
      let maxVal = p[0];
      for (let i = 1; i < p.length; i++) {
        if (p[i] > maxVal) {
          maxVal = p[i];
          maxIdx = i;
        }
      }
      return this.classes[maxIdx];
    });
  }
};
var KNeighborsRegressor = class extends KNeighborsModel {
  training_targets;
  constructor(params, meta) {
    super(params, meta);
    this.training_targets = params.training_targets || params.training_labels;
  }
  predict(inputs) {
    return inputs.map((input) => {
      const nearest = this.getDistances(input);
      let sum = 0;
      for (let i = 0; i < nearest.length; i++) {
        sum += this.training_targets[nearest[i].index];
      }
      return sum / this.n_neighbors;
    });
  }
};

// src/models/preprocessing.ts
var StandardScaler = class {
  mean;
  scale;
  constructor(params, meta) {
    this.mean = params.mean || new Array(meta.n_features).fill(0);
    this.scale = params.scale || new Array(meta.n_features).fill(1);
  }
  transform(inputs) {
    return inputs.map((row) => {
      return row.map((val, j) => (val - this.mean[j]) / this.scale[j]);
    });
  }
};
var MinMaxScaler = class {
  min;
  scale;
  constructor(params, meta) {
    this.min = params.min || new Array(meta.n_features).fill(0);
    this.scale = params.scale || new Array(meta.n_features).fill(1);
  }
  transform(inputs) {
    return inputs.map((row) => {
      return row.map((val, j) => val * this.scale[j] + this.min[j]);
    });
  }
};

// src/index.ts
function loadModel(modelJson) {
  if (modelJson?.meta?.library !== "skjson") {
    throw new Error("Invalid skjson format");
  }
  switch (modelJson.meta.model_type) {
    case "LinearRegression":
      return new LinearRegression(modelJson.params, modelJson.meta);
    case "Ridge":
      return new Ridge(modelJson.params, modelJson.meta);
    case "Lasso":
      return new Lasso(modelJson.params, modelJson.meta);
    case "LogisticRegression":
      return new LogisticRegression(modelJson.params, modelJson.meta);
    case "DecisionTreeClassifier":
      return new DecisionTreeClassifier(modelJson.params, modelJson.meta);
    case "DecisionTreeRegressor":
      return new DecisionTreeRegressor(modelJson.params, modelJson.meta);
    case "RandomForestClassifier":
      return new RandomForestClassifier(modelJson.params, modelJson.meta);
    case "RandomForestRegressor":
      return new RandomForestRegressor(modelJson.params, modelJson.meta);
    case "GradientBoostingClassifier":
      return new GradientBoostingClassifier(modelJson.params, modelJson.meta);
    case "GradientBoostingRegressor":
      return new GradientBoostingRegressor(modelJson.params, modelJson.meta);
    case "GaussianNB":
      return new GaussianNB(modelJson.params, modelJson.meta);
    case "SVC":
      return new SVC(modelJson.params, modelJson.meta);
    case "SVR":
      return new SVR(modelJson.params, modelJson.meta);
    case "KNeighborsClassifier":
      return new KNeighborsClassifier(modelJson.params, modelJson.meta);
    case "KNeighborsRegressor":
      return new KNeighborsRegressor(modelJson.params, modelJson.meta);
    case "StandardScaler":
      return new StandardScaler(modelJson.params, modelJson.meta);
    case "MinMaxScaler":
      return new MinMaxScaler(modelJson.params, modelJson.meta);
    default:
      throw new Error(`Model type ${modelJson.meta.model_type} is currently not supported.`);
  }
}

exports.loadModel = loadModel;
//# sourceMappingURL=index.cjs.map
//# sourceMappingURL=index.cjs.map