import { ModelMeta, Predictor } from '../types';

abstract class KNeighborsModel implements Predictor {
  protected training_data: number[][];
  protected n_neighbors: number;
  protected metric: string;
  protected meta: ModelMeta;

  constructor(params: any, meta: ModelMeta) {
    this.training_data = params.training_data;
    this.n_neighbors = params.n_neighbors;
    this.metric = params.metric;
    this.meta = meta;
  }

  protected getDistances(input: number[]): { distance: number; index: number }[] {
    const distances = this.training_data.map((row, index) => {
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        const diff = input[i] - row[i];
        sum += diff * diff;
      }
      return { distance: Math.sqrt(sum), index };
    });
    
    // Sort ascending
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, this.n_neighbors);
  }

  public abstract predict(inputs: number[][]): any[];
}

export class KNeighborsClassifier extends KNeighborsModel {
  private training_labels: any[];
  private classes: any[];

  constructor(params: any, meta: ModelMeta) {
    super(params, meta);
    this.training_labels = params.training_labels;
    this.classes = params.classes || meta.classes;
  }

  public predict_proba(inputs: number[][]): number[][] {
    return inputs.map(input => {
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
      return this.classes[maxIdx];
    });
  }
}

export class KNeighborsRegressor extends KNeighborsModel {
  private training_targets: number[];

  constructor(params: any, meta: ModelMeta) {
    super(params, meta);
    this.training_targets = params.training_targets || params.training_labels; // fallback
  }

  public predict(inputs: number[][]): number[] {
    return inputs.map(input => {
      const nearest = this.getDistances(input);
      let sum = 0;
      for (let i = 0; i < nearest.length; i++) {
        sum += this.training_targets[nearest[i].index];
      }
      return sum / this.n_neighbors;
    });
  }
}
