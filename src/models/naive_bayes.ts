import { ModelMeta, Predictor } from '../types';

export class GaussianNB implements Predictor {
  private theta: number[][];
  private var: number[][];
  private class_prior: number[];
  private classes: any[];
  private meta: ModelMeta;

  constructor(params: any, meta: ModelMeta) {
    this.theta = params.theta;
    this.var = params.var;
    this.class_prior = params.class_prior;
    this.classes = params.classes || meta.classes;
    this.meta = meta;
  }

  private calculateLogProb(x: number[], mean: number[], var_: number[]): number {
    let logProb = 0;
    for (let i = 0; i < x.length; i++) {
      logProb -= 0.5 * Math.log(2 * Math.PI * var_[i]);
      logProb -= Math.pow(x[i] - mean[i], 2) / (2 * var_[i]);
    }
    return logProb;
  }

  public predict_proba(inputs: number[][]): number[][] {
    return inputs.map(input => {
      const logProbas = this.classes.map((_, i) => {
        return Math.log(this.class_prior[i]) + this.calculateLogProb(input, this.theta[i], this.var[i]);
      });
      
      // LogSumExp trick for numerical stability
      const maxLogProb = Math.max(...logProbas);
      let sumExp = 0;
      for (let i = 0; i < logProbas.length; i++) {
        sumExp += Math.exp(logProbas[i] - maxLogProb);
      }
      
      const probas = logProbas.map(lp => Math.exp(lp - maxLogProb) / sumExp);
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
