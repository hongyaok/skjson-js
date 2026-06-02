import { ModelMeta, Predictor } from '../types';
import { LinearModel } from './linear';

export class SVC extends LinearModel {
  public predict(inputs: number[][]): any[] {
    const isMultiClass = Array.isArray(this.intercept) && this.intercept.length > 1;
    return inputs.map(input => {
      if (isMultiClass) {
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < (this.intercept as number[]).length; i++) {
          const w = (this.coef as number[][])[i];
          const val = this.dotProduct(input, w) + (this.intercept as number[])[i];
          if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
          }
        }
        return this.meta.classes ? this.meta.classes[maxIdx] : maxIdx;
      } else {
        const coef = Array.isArray(this.coef[0]) ? (this.coef as number[][])[0] : (this.coef as number[]);
        const intercept = Array.isArray(this.intercept) ? this.intercept[0] : this.intercept;
        const output = this.dotProduct(input, coef) + intercept;
        const idx = output > 0 ? 1 : 0;
        return this.meta.classes ? this.meta.classes[idx] : idx;
      }
    });
  }
}

export class SVR extends LinearModel {
  public predict(inputs: number[][]): number[] {
    return inputs.map(input => {
      const coef = Array.isArray(this.coef[0]) ? (this.coef as number[][])[0] : (this.coef as number[]);
      const intercept = Array.isArray(this.intercept) ? this.intercept[0] : this.intercept;
      return this.dotProduct(input, coef) + intercept;
    });
  }
}
