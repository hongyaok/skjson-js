import { ModelMeta, Predictor } from '../types';

export class LinearModel implements Predictor {
  protected coef: number[] | number[][];
  protected intercept: number | number[];
  protected meta: ModelMeta;

  constructor(params: any, meta: ModelMeta) {
    this.coef = params.coef;
    this.intercept = params.intercept;
    this.meta = meta;
  }

  protected dotProduct(input: number[], weights: number[]): number {
    return input.reduce((sum, val, i) => sum + val * weights[i], 0);
  }

  public predict(inputs: number[][]): any[] {
    const isMultiClass = Array.isArray(this.intercept);
    
    return inputs.map(input => {
      if (isMultiClass) {
        // Multi-class or multi-output
        const numOutputs = (this.intercept as number[]).length;
        const outputs = [];
        for (let i = 0; i < numOutputs; i++) {
          const w = (this.coef as number[][])[i];
          outputs.push(this.dotProduct(input, w) + (this.intercept as number[])[i]);
        }
        return this.postProcess(outputs);
      } else {
        // Single output
        const output = this.dotProduct(input, this.coef as number[]) + (this.intercept as number);
        return this.postProcessSingle(output);
      }
    });
  }
  
  protected postProcess(outputs: number[]): any {
    return outputs;
  }
  
  protected postProcessSingle(output: number): any {
    return output;
  }
}

export class LinearRegression extends LinearModel {}

export class Ridge extends LinearModel {}

export class Lasso extends LinearModel {}

export class LogisticRegression extends LinearModel {
  public predict_proba(inputs: number[][]): number[][] {
    const isMultiClass = this.meta.classes && this.meta.classes.length > 2;
    
    return inputs.map(input => {
      if (isMultiClass) {
        const numOutputs = (this.intercept as number[]).length;
        const outputs = [];
        for (let i = 0; i < numOutputs; i++) {
          const w = (this.coef as number[][])[i];
          outputs.push(this.dotProduct(input, w) + (this.intercept as number[])[i]);
        }
        
        // Softmax
        const maxVal = Math.max(...outputs);
        let sumExp = 0;
        const exps = outputs.map(o => {
          const e = Math.exp(o - maxVal);
          sumExp += e;
          return e;
        });
        return exps.map(e => e / sumExp);
      } else {
        // Binary
        let output = 0;
        if (Array.isArray(this.coef[0])) {
           output = this.dotProduct(input, (this.coef as number[][])[0]) + (Array.isArray(this.intercept) ? this.intercept[0] : this.intercept);
        } else {
           output = this.dotProduct(input, this.coef as number[]) + (this.intercept as number);
        }
        const prob = 1 / (1 + Math.exp(-output));
        return [1 - prob, prob];
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
      return this.meta.classes ? this.meta.classes[maxIdx] : maxIdx;
    });
  }
}
