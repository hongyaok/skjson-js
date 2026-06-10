import { ModelMeta, Predictor } from '../types';

interface TreeNode {
  id: number;
  feature: number;
  threshold: number;
  left: number;
  right: number;
  value: number[] | number;
  missing_go_to_left?: boolean;
}

export class TreeModel implements Predictor {
  protected meta: ModelMeta;

  constructor(meta: ModelMeta) {
    this.meta = meta;
  }

  protected traverseTree(nodes: TreeNode[], input: number[]): number[] | number {
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
  
  public predict(inputs: number[][]): any[] {
    throw new Error('Not implemented');
  }
}

export class DecisionTreeClassifier extends TreeModel {
  private nodes: TreeNode[];
  private classes: any[];

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.nodes = params.nodes;
    this.classes = params.classes || meta.classes;
  }

  public predict_proba(inputs: number[][]): number[][] {
    return inputs.map(input => {
      const value = this.traverseTree(this.nodes, input) as number[];
      const total = value.reduce((a, b) => a + b, 0);
      return total > 0 ? value.map(v => v / total) : value;
    });
  }

  public predict(inputs: number[][]): any[] {
    const probas = this.predict_proba(inputs);
    return probas.map(p => {
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
}

export class DecisionTreeRegressor extends TreeModel {
  private nodes: TreeNode[];

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.nodes = params.nodes;
  }

  public predict(inputs: number[][]): number[] {
    return inputs.map(input => {
      const val = this.traverseTree(this.nodes, input);
      return Array.isArray(val) ? val[0] : (val as number);
    });
  }
}

export class RandomForestClassifier extends TreeModel {
  private trees: TreeNode[][];
  private classes: any[];

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.trees = params.trees;
    this.classes = params.classes || meta.classes;
  }

  public predict_proba(inputs: number[][]): number[][] {
    return inputs.map(input => {
      const numClasses = this.classes ? this.classes.length : (this.trees[0][0].value as number[]).length;
      const avgProbas = new Array(numClasses).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        const value = this.traverseTree(this.trees[i], input) as number[];
        const total = value.reduce((a, b) => a + b, 0);
        const probas = total > 0 ? value.map(v => v / total) : value;
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

  public predict(inputs: number[][]): any[] {
    const probas = this.predict_proba(inputs);
    return probas.map(p => {
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
}

export class RandomForestRegressor extends TreeModel {
  private trees: TreeNode[][];

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.trees = params.trees;
  }

  public predict(inputs: number[][]): number[] {
    return inputs.map(input => {
      let sum = 0;
      for (let i = 0; i < this.trees.length; i++) {
        const val = this.traverseTree(this.trees[i], input);
        sum += Array.isArray(val) ? val[0] : (val as number);
      }
      return sum / this.trees.length;
    });
  }
}

export class GradientBoostingClassifier extends TreeModel {
  private trees: TreeNode[][][];
  private learningRate: number;
  private initRaw: number[];
  private classes: any[];
  private nTreeOutputs: number;

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.trees = params.trees;
    this.learningRate = params.learning_rate;
    this.initRaw = params.init_raw_predictions;
    this.classes = params.classes || meta.classes;
    this.nTreeOutputs = params.n_tree_outputs;
  }

  public predict_proba(inputs: number[][]): number[][] {
    return inputs.map(input => {
      const raw = [...this.initRaw];

      for (const stageTrees of this.trees) {
        for (let k = 0; k < stageTrees.length; k++) {
          const val = this.traverseTree(stageTrees[k], input);
          const leafValue = Array.isArray(val) ? val[0] : (val as number);
          raw[k] += this.learningRate * leafValue;
        }
      }

      if (this.nTreeOutputs === 1) {
        // Binary classification: sigmoid
        const p1 = 1.0 / (1.0 + Math.exp(-raw[0]));
        return [1.0 - p1, p1];
      } else {
        // Multiclass: softmax
        const maxRaw = Math.max(...raw);
        const expRaw = raw.map(r => Math.exp(r - maxRaw));
        const sumExp = expRaw.reduce((a, b) => a + b, 0);
        return expRaw.map(e => e / sumExp);
      }
    });
  }

  public predict(inputs: number[][]): any[] {
    const probas = this.predict_proba(inputs);
    return probas.map(p => {
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
}

export class GradientBoostingRegressor extends TreeModel {
  private trees: TreeNode[][][];
  private learningRate: number;
  private initRaw: number[];

  constructor(params: any, meta: ModelMeta) {
    super(meta);
    this.trees = params.trees;
    this.learningRate = params.learning_rate;
    this.initRaw = params.init_raw_predictions;
  }

  public predict(inputs: number[][]): number[] {
    return inputs.map(input => {
      const raw = [...this.initRaw];
      for (const stageTrees of this.trees) {
        for (let k = 0; k < stageTrees.length; k++) {
          const val = this.traverseTree(stageTrees[k], input);
          const leafValue = Array.isArray(val) ? val[0] : (val as number);
          raw[k] += this.learningRate * leafValue;
        }
      }
      return raw[0];
    });
  }
}
