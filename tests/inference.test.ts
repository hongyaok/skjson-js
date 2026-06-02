import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { loadModel } from '../src/index';

import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURES_DIR = path.join(__dirname, 'fixtures');

describe('Model Inference Parity', () => {
  const files = fs.readdirSync(FIXTURES_DIR);
  const models = files.filter(f => f.endsWith('_model.json')).map(f => f.replace('_model.json', ''));

  for (const modelName of models) {
    it(`Matches Python exact outputs for ${modelName}`, () => {
      const modelJson = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, `${modelName}_model.json`), 'utf-8'));
      const outputsJson = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, `${modelName}_outputs.json`), 'utf-8'));

      const predictor = loadModel(modelJson);
      const inputs = outputsJson.inputs;

      if ('transform' in predictor) {
        // It's a Transformer
        const expected = outputsJson.transformed;
        const result = predictor.transform(inputs);
        
        expect(result.length).toBe(expected.length);
        for (let i = 0; i < result.length; i++) {
          for (let j = 0; j < result[i].length; j++) {
            expect(result[i][j]).toBeCloseTo(expected[i][j], 5);
          }
        }
      } else {
        // It's a Predictor
        const expectedPreds = outputsJson.predictions;
        const result = predictor.predict(inputs);
        
        expect(result.length).toBe(expectedPreds.length);
        for (let i = 0; i < result.length; i++) {
          if (typeof result[i] === 'number' && typeof expectedPreds[i] === 'number') {
             expect(result[i]).toBeCloseTo(expectedPreds[i], 5);
          } else {
             expect(result[i]).toBe(expectedPreds[i]);
          }
        }

        if (outputsJson.probabilities && typeof predictor.predict_proba === 'function') {
          const expectedProbas = outputsJson.probabilities;
          const probas = predictor.predict_proba(inputs);
          
          expect(probas.length).toBe(expectedProbas.length);
          for (let i = 0; i < probas.length; i++) {
            for (let j = 0; j < probas[i].length; j++) {
              expect(probas[i][j]).toBeCloseTo(expectedProbas[i][j], 5);
            }
          }
        }
      }
    });
  }
});
