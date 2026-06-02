import { ModelJson, Predictor, Transformer } from './types';
import { LinearRegression, LogisticRegression, Ridge, Lasso } from './models/linear';
import { DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor } from './models/tree';
import { GaussianNB } from './models/naive_bayes';
import { SVC, SVR } from './models/svm';
import { KNeighborsClassifier, KNeighborsRegressor } from './models/neighbors';
import { StandardScaler, MinMaxScaler } from './models/preprocessing';

export function loadModel(modelJson: ModelJson): Predictor | Transformer {
    if (modelJson?.meta?.library !== 'skjson') {
        throw new Error('Invalid skjson format');
    }

    switch (modelJson.meta.model_type) {
        case 'LinearRegression':
            return new LinearRegression(modelJson.params, modelJson.meta);
        case 'Ridge':
            return new Ridge(modelJson.params, modelJson.meta);
        case 'Lasso':
            return new Lasso(modelJson.params, modelJson.meta);
        case 'LogisticRegression':
            return new LogisticRegression(modelJson.params, modelJson.meta);
        case 'DecisionTreeClassifier':
            return new DecisionTreeClassifier(modelJson.params, modelJson.meta);
        case 'DecisionTreeRegressor':
            return new DecisionTreeRegressor(modelJson.params, modelJson.meta);
        case 'RandomForestClassifier':
            return new RandomForestClassifier(modelJson.params, modelJson.meta);
        case 'RandomForestRegressor':
            return new RandomForestRegressor(modelJson.params, modelJson.meta);
        case 'GaussianNB':
            return new GaussianNB(modelJson.params, modelJson.meta);
        case 'SVC':
            return new SVC(modelJson.params, modelJson.meta);
        case 'SVR':
            return new SVR(modelJson.params, modelJson.meta);
        case 'KNeighborsClassifier':
            return new KNeighborsClassifier(modelJson.params, modelJson.meta);
        case 'KNeighborsRegressor':
            return new KNeighborsRegressor(modelJson.params, modelJson.meta);
        case 'StandardScaler':
            return new StandardScaler(modelJson.params, modelJson.meta);
        case 'MinMaxScaler':
            return new MinMaxScaler(modelJson.params, modelJson.meta);
        default:
            throw new Error(`Model type ${modelJson.meta.model_type} is currently not supported.`);
    }
}

