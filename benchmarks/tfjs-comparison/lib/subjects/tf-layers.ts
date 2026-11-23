/** TensorFlow.js through the high-level `tf.layers` model API (what most users write). */
import type { Dataset } from '../dataset';
import { denseToMlp, mlpToDense, rnnFromParams, rnnToParams, transpose } from '../layout';
import type { Scenario, Subject } from '../spec';
import { BLOCKS, POOL } from '../spec';
import type { Tf } from './tf-load';

type Tensor = import('@tensorflow/tfjs').Tensor;

type Sequential = ReturnType<Tf['sequential']>;

/** A dense MLP (tanh hidden, linear output) holding the given weights (neuro-learn layout). */
function buildMlp(tf: Tf, sizes: readonly number[], weights: Float32Array): Sequential {
  const model = tf.sequential();
  const params = mlpToDense(sizes, weights);
  params.forEach((p, k) => {
    model.add(
      tf.layers.dense({
        units: p.fanOut,
        activation: k < params.length - 1 ? 'tanh' : 'linear',
        ...(k === 0 ? { inputShape: [sizes[0] as number] } : {}),
      }),
    );
    model.layers[k]?.setWeights([tf.tensor2d(p.kernel, [p.fanIn, p.fanOut]), tf.tensor1d(p.bias)]);
  });
  return model;
}

export function buildTfLayers(tf: Tf, sc: Scenario, init: Float32Array, data: Dataset): Subject {
  const inputSize = sc.layers[0] as number;
  const outputSize = sc.layers[sc.layers.length - 1] as number;
  const tensorCount = (): number => tf.memory().numTensors;

  if (sc.kind === 'rnn-forward') {
    const [, hidden, outputs] = sc.layers as [number, number, number];
    const params = rnnToParams(sc.layers, init);
    const model = tf.sequential();
    model.add(
      tf.layers.simpleRNN({
        units: hidden,
        activation: 'tanh',
        stateful: true,
        batchInputShape: [1, 1, inputSize],
      }),
    );
    model.add(tf.layers.dense({ units: outputs, activation: 'linear' }));
    const [rnn, dense] = model.layers as [
      (typeof model.layers)[number],
      (typeof model.layers)[number],
    ];
    rnn.setWeights([
      tf.tensor2d(params.inputKernel, [inputSize, hidden]),
      tf.tensor2d(params.recurrentKernel, [hidden, hidden]),
      tf.tensor1d(params.hiddenBias),
    ]);
    dense.setWeights([
      tf.tensor2d(params.outputKernel, [hidden, outputs]),
      tf.tensor1d(params.outputBias),
    ]);
    const output = new Float32Array(outputs);
    return {
      output,
      tensorCount,
      weights() {
        const [k, r, b] = rnn.getWeights() as [Tensor, Tensor, Tensor];
        const [v, c] = dense.getWeights() as [Tensor, Tensor];
        return rnnFromParams(sc.layers, {
          inputKernel: k.dataSync() as Float32Array,
          recurrentKernel: r.dataSync() as Float32Array,
          hiddenBias: b.dataSync() as Float32Array,
          outputKernel: v.dataSync() as Float32Array,
          outputBias: c.dataSync() as Float32Array,
        });
      },
      run(call) {
        const x = data.inputs[call & (POOL - 1)] as Float32Array;
        tf.tidy(() => {
          output.set((model.predict(tf.tensor3d(x, [1, 1, inputSize])) as Tensor).dataSync());
        });
      },
    };
  }

  if (sc.kind === 'population-forward') {
    // The straightforward way to give every creature its own brain: one model per creature.
    const stateSize = init.length / sc.batch;
    const models = Array.from({ length: sc.batch }, (_, c) =>
      buildMlp(tf, sc.layers, init.subarray(c * stateSize, (c + 1) * stateSize)),
    );
    const output = new Float32Array(sc.batch * outputSize);
    return {
      output,
      tensorCount,
      weights: () => init,
      run(call) {
        const block = data.blocks[call & (BLOCKS - 1)] as Float32Array;
        tf.tidy(() => {
          models.forEach((model, c) => {
            const x = tf.tensor2d(block.subarray(c * inputSize, (c + 1) * inputSize), [
              1,
              inputSize,
            ]);
            output.set((model.predict(x) as Tensor).dataSync(), c * outputSize);
          });
        });
      },
    };
  }

  let model: Sequential;
  if (sc.kind === 'delta-rule') {
    model = tf.sequential();
    model.add(
      tf.layers.dense({
        units: outputSize,
        useBias: false,
        activation: 'linear',
        inputShape: [inputSize],
      }),
    );
    // neuro-learn layout is outputs x inputs; tf wants inputs x outputs.
    model.layers[0]?.setWeights([
      tf.tensor2d(transpose(init, outputSize, inputSize), [inputSize, outputSize]),
    ]);
  } else {
    model = buildMlp(tf, sc.layers, init);
  }
  const weights = (): Float32Array => {
    if (sc.kind === 'delta-rule') {
      const kernel = (model.layers[0]?.getWeights()[0] as Tensor).dataSync() as Float32Array;
      return transpose(kernel, inputSize, outputSize);
    }
    return denseToMlp(
      model.layers.map((layer, k) => {
        const [kernel, bias] = layer.getWeights() as [Tensor, Tensor];
        return {
          fanIn: sc.layers[k] as number,
          fanOut: sc.layers[k + 1] as number,
          kernel: kernel.dataSync() as Float32Array,
          bias: bias.dataSync() as Float32Array,
        };
      }),
    );
  };

  if (sc.kind === 'mlp-forward') {
    const output = new Float32Array(sc.batch * outputSize);
    return {
      output,
      tensorCount,
      weights,
      run(call) {
        const block = data.blocks[call & (BLOCKS - 1)] as Float32Array;
        tf.tidy(() => {
          // batchSize = whole block: the default (32) would split the call into several passes.
          output.set(
            (
              model.predict(tf.tensor2d(block, [sc.batch, inputSize]), {
                batchSize: sc.batch,
              }) as Tensor
            ).dataSync(),
          );
        });
      },
    };
  }

  const optimizer = tf.train.sgd(sc.rate);
  return {
    output: new Float32Array(outputSize),
    tensorCount,
    weights,
    run(call) {
      const j = call & (POOL - 1);
      tf.tidy(() => {
        const x = tf.tensor2d(data.inputs[j] as Float32Array, [1, inputSize]);
        const t = tf.tensor2d(data.targets[j] as Float32Array, [1, outputSize]);
        // 0.5 * sum of squared errors: the same loss neuro-learn differentiates (tf's built-in MSE is a mean).
        optimizer.minimize(
          () =>
            tf.mul(
              0.5,
              tf.sum(tf.squaredDifference(model.apply(x) as Tensor, t)),
            ) as import('@tensorflow/tfjs').Scalar,
        );
      });
    },
  };
}
