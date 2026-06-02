export interface ModelMeta {
  library: string;
  version?: string;
  model_type: string;
  task: string;
  n_features: number;
  feature_names?: string[] | null;
  classes?: number[] | string[] | null;
}

export interface ModelJson {
  meta: ModelMeta;
  params: any;
}

export interface Predictor {
  predict(inputs: number[][]): any[];
  predict_proba?(inputs: number[][]): number[][];
}

export interface Transformer {
  transform(inputs: number[][]): number[][];
}
