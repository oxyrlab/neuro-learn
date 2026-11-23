/** neuro-learn adapters. Imports the built package (`dist`), exactly what a consumer installs. */
import type { Dataset } from '../dataset';
import type { Scenario, Subject } from '../spec';
import { BLOCKS, POOL } from '../spec';

const MAX_WEIGHT = 100; // far above anything reached here, so clamping never differs from tf.js

export async function buildNeuro(
  sc: Scenario,
  init: Float32Array,
  data: Dataset,
): Promise<Subject> {
  const lib = await import('../../../../dist/index.js');
  const rng = lib.createSeededRng(1);
  const inputSize = sc.layers[0] as number;
  const outputSize = sc.layers[sc.layers.length - 1] as number;

  if (sc.kind === 'mlp-forward') {
    const network = new lib.FeedforwardNetwork({ layerSizes: [...sc.layers], rng });
    network.restore(init);
    const output = new Float32Array(sc.batch * outputSize);
    const outRows = Array.from({ length: sc.batch }, (_, k) =>
      output.subarray(k * outputSize, (k + 1) * outputSize),
    );
    const inRows = data.blocks.map((block) =>
      Array.from({ length: sc.batch }, (_, k) =>
        block.subarray(k * inputSize, (k + 1) * inputSize),
      ),
    );
    return {
      output,
      weights: () => network.serialize(),
      run(call) {
        const rows = inRows[call & (BLOCKS - 1)] as Float32Array[];
        for (let k = 0; k < sc.batch; k += 1)
          network.forward(rows[k] as Float32Array, outRows[k] as Float32Array);
      },
    };
  }

  if (sc.kind === 'population-forward') {
    const stateSize = init.length / sc.batch;
    const networks = Array.from({ length: sc.batch }, (_, c) => {
      const network = new lib.FeedforwardNetwork({ layerSizes: [...sc.layers], rng });
      network.restore(init.subarray(c * stateSize, (c + 1) * stateSize));
      return network;
    });
    const output = new Float32Array(sc.batch * outputSize);
    const outRows = networks.map((_, c) => output.subarray(c * outputSize, (c + 1) * outputSize));
    const inRows = data.blocks.map((block) =>
      networks.map((_, c) => block.subarray(c * inputSize, (c + 1) * inputSize)),
    );
    return {
      output,
      weights: () => init,
      run(call) {
        const rows = inRows[call & (BLOCKS - 1)] as Float32Array[];
        for (let c = 0; c < sc.batch; c += 1)
          (networks[c] as (typeof networks)[number]).forward(
            rows[c] as Float32Array,
            outRows[c] as Float32Array,
          );
      },
    };
  }

  if (sc.kind === 'mlp-train') {
    const learner = new lib.MlpSgdLearner({
      layerSizes: [...sc.layers],
      rng,
      maxWeight: MAX_WEIGHT,
    });
    learner.restore(init);
    return {
      output: new Float32Array(outputSize),
      weights: () => learner.serialize(),
      run(call) {
        const j = call & (POOL - 1);
        learner.learn(data.inputs[j] as Float32Array, data.targets[j] as Float32Array, sc.rate);
      },
    };
  }

  if (sc.kind === 'delta-rule') {
    const learner = new lib.RescorlaWagnerLearner({
      inputSize,
      outputSize,
      maxStrength: MAX_WEIGHT,
    });
    learner.restore(init);
    return {
      output: new Float32Array(outputSize),
      weights: () => learner.serialize(),
      run(call) {
        const j = call & (POOL - 1);
        learner.learn(data.inputs[j] as Float32Array, data.targets[j] as Float32Array, sc.rate);
      },
    };
  }

  const [, hidden, outputs] = sc.layers as [number, number, number];
  const network = new lib.RecurrentNetwork({
    inputSize,
    hiddenSize: hidden,
    outputSize: outputs,
    rng,
  });
  const state = new Float32Array(init.length + hidden); // weights followed by a zero hidden state
  state.set(init);
  network.restore(state);
  const output = new Float32Array(outputs);
  return {
    output,
    weights: () => network.serialize().subarray(0, init.length),
    run(call) {
      network.forward(data.inputs[call & (POOL - 1)] as Float32Array, output);
    },
  };
}
