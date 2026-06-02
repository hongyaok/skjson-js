import { ModelMeta, Transformer } from '../types';

export class StandardScaler implements Transformer {
  private mean: number[];
  private scale: number[];

  constructor(params: any, meta: ModelMeta) {
    this.mean = params.mean || new Array(meta.n_features).fill(0);
    this.scale = params.scale || new Array(meta.n_features).fill(1);
  }

  public transform(inputs: number[][]): number[][] {
    return inputs.map(row => {
      return row.map((val, j) => (val - this.mean[j]) / this.scale[j]);
    });
  }
}

export class MinMaxScaler implements Transformer {
  private min: number[];
  private scale: number[];

  constructor(params: any, meta: ModelMeta) {
    this.min = params.min || new Array(meta.n_features).fill(0);
    this.scale = params.scale || new Array(meta.n_features).fill(1);
  }

  public transform(inputs: number[][]): number[][] {
    return inputs.map(row => {
      return row.map((val, j) => val * this.scale[j] + this.min[j]);
    });
  }
}
