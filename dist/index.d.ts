interface ModelMeta {
    library: string;
    version?: string;
    model_type: string;
    task: string;
    n_features: number;
    feature_names?: string[] | null;
    classes?: number[] | string[] | null;
}
interface ModelJson {
    meta: ModelMeta;
    params: any;
}
interface Predictor {
    predict(inputs: number[][]): any[];
    predict_proba?(inputs: number[][]): number[][];
}
interface Transformer {
    transform(inputs: number[][]): number[][];
}

declare function loadModel(modelJson: ModelJson): Predictor | Transformer;

export { loadModel };
