import { ModelMeta, Predictor } from '../types';

interface TreeNode {
  id: number;
  feature: number;
  threshold: number;
  left: number;
  right: number;
  value: number[] | number;
}

export class TreeModel implements Predictor {
  protected meta: ModelMeta;

  constructor(meta: ModelMeta) {
    this.meta = meta;
  }

  protected traverseTree(nodes: TreeNode[], input: number[]): number[] | number {
    let node = nodes[0];
    while (node.left !== -1 && node.right !== -1 && node.feature !== -2) {
      if (input[node.feature] <= node.threshold) {
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
      const probas = this.traverseTree(this.nodes, input) as number[];
      return probas;
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
      return Array.isArray(val) ? val[0] : val;
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
      const probas = this.trees.map(tree => this.traverseTree(tree, input) as number[]);
      const numClasses = probas[0].length;
      const avgProbas = new Array(numClasses).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        for (let j = 0; j < numClasses; j++) {
          avgProbas[j] += probas[i][j];
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
        sum += Array.isArray(val) ? val[0] : val;
      }
      return sum / this.trees.length;
    });
  }
}
