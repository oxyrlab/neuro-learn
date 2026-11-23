/** TensorFlow.js through low-level ops (matMul/add/tanh + autograd), skipping the layers wrapper. */
import type { Dataset } from '../dataset';
import { denseToMlp, mlpToDense, rnnFromParams, rnnToParams, transpose } from '../layout';
import type { Scenario, Subject } from '../spec';
import { BLOCKS, POOL } from '../spec';
import type { Tf } from './tf-load';

type Tensor = import('@tensorflow/tfjs').Tensor;
type Scalar = import('@tensorflow/tfjs').Scalar;
type Variable = import('@tensorflow/tfjs').Variable;

export function buildTfOps(tf: Tf, sc: Scenario, init: Float32Array, data: Dataset): Subject {
  const inputSize = sc.layers[0] as number;
  const outputSize = sc.layers[sc.layers.length - 1] as number;
  const tensorCount = (): number => tf.memory().numTensors;
  if (sc.kind === 'population-forward') {
    // All creatures in one batched op: weights stacked as [creatures, fanIn, fanOut], one batchMatMul per layer.
    const stateSize = init.length / sc.batch;
    const perCreature = Array.from({ length: sc.batch }, (_, c) =>
      mlpToDense(sc.layers, init.subarray(c * stateSize, (c + 1) * stateSize)),
    );
    const layerCount = sc.layers.length - 1;
    const kernels = Array.from({ length: layerCount }, (_, l) => {
      const p = perCreature[0]?.[l] as { fanIn: number; fanOut: number };
      const stacked = new Float32Array(sc.batch * p.fanIn * p.fanOut);
      perCreature.forEach((layers, c) =>
        stacked.set((layers[l] as { kernel: Float32Array }).kernel, c * p.fanIn * p.fanOut),
      );
      return tf.keep(tf.tensor3d(stacked, [sc.batch, p.fanIn, p.fanOut]));
    });
    const biases = Array.from({ length: layerCount }, (_, l) => {
      const fanOut = sc.layers[l + 1] as number;
      const stacked = new Float32Array(sc.batch * fanOut);
      perCreature.forEach((layers, c) =>
        stacked.set((layers[l] as { bias: Float32Array }).bias, c * fanOut),
      );
      return tf.keep(tf.tensor3d(stacked, [sc.batch, 1, fanOut]));
    });
    const output = new Float32Array(sc.batch * outputSize);
    return {
      output,
      tensorCount,
      weights: () => init,
      run(call) {
        const block = data.blocks[call & (BLOCKS - 1)] as Float32Array;
        tf.tidy(() => {
          let h: Tensor = tf.tensor3d(block, [sc.batch, 1, inputSize]);
          kernels.forEach((kernel, l) => {
            h = tf.add(tf.matMul(h, kernel), biases[l] as Tensor);
            if (l < layerCount - 1) h = tf.tanh(h);
          });
          output.set(h.dataSync());
        });
      },
    };
  }

  const trainable = sc.kind === 'mlp-train' || sc.kind === 'delta-rule';
  const make = (values: Float32Array, shape: [number, number] | [number]): Tensor =>
    trainable ? tf.variable(tf.tensor(values, shape)) : tf.keep(tf.tensor(values, shape));

  if (sc.kind === 'rnn-forward') {
    const [, hidden, outputs] = sc.layers as [number, number, number];
    const p = rnnToParams(sc.layers, init);
    const k = make(p.inputKernel, [inputSize, hidden]);
    const r = make(p.recurrentKernel, [hidden, hidden]);
    const b = make(p.hiddenBias, [hidden]);
    const v = make(p.outputKernel, [hidden, outputs]);
    const c = make(p.outputBias, [outputs]);
    let state: Tensor = tf.keep(tf.zeros([1, hidden]));
    const output = new Float32Array(outputs);
    return {
      output,
      tensorCount,
      weights: () =>
        rnnFromParams(sc.layers, {
          inputKernel: k.dataSync() as Float32Array,
          recurrentKernel: r.dataSync() as Float32Array,
          hiddenBias: b.dataSync() as Float32Array,
          outputKernel: v.dataSync() as Float32Array,
          outputBias: c.dataSync() as Float32Array,
        }),
      run(call) {
        const x = data.inputs[call & (POOL - 1)] as Float32Array;
        const [next, y] = tf.tidy(() => {
          const nextState = tf.tanh(
            tf.add(tf.add(tf.matMul(tf.tensor2d(x, [1, inputSize]), k), tf.matMul(state, r)), b),
          );
          return [nextState, tf.add(tf.matMul(nextState, v), c)];
        }) as [Tensor, Tensor];
        output.set(y.dataSync());
        y.dispose();
        state.dispose();
        state = next;
      },
    };
  }

  let parameters: Tensor[];
  let forward: (x: Tensor) => Tensor;
  let weights: () => Float32Array;
  if (sc.kind === 'delta-rule') {
    const w = make(transpose(init, outputSize, inputSize), [inputSize, outputSize]);
    parameters = [w];
    forward = (x) => tf.matMul(x, w);
    weights = () => transpose(w.dataSync() as Float32Array, inputSize, outputSize);
  } else {
    const dense = mlpToDense(sc.layers, init);
    const kernels = dense.map((d) => make(d.kernel, [d.fanIn, d.fanOut]));
    const biases = dense.map((d) => make(d.bias, [d.fanOut]));
    parameters = [...kernels, ...biases];
    forward = (x) => {
      let h = x;
      kernels.forEach((kernel, layer) => {
        h = tf.add(tf.matMul(h, kernel), biases[layer] as Tensor);
        if (layer < kernels.length - 1) h = tf.tanh(h);
      });
      return h;
    };
    weights = () =>
      denseToMlp(
        dense.map((d, layer) => ({
          ...d,
          kernel: (kernels[layer] as Tensor).dataSync() as Float32Array,
          bias: (biases[layer] as Tensor).dataSync() as Float32Array,
        })),
      );
  }

  if (sc.kind === 'mlp-forward') {
    const output = new Float32Array(sc.batch * outputSize);
    return {
      output,
      tensorCount,
      weights,
      run(call) {
        const block = data.blocks[call & (BLOCKS - 1)] as Float32Array;
        tf.tidy(() => {
          output.set(forward(tf.tensor2d(block, [sc.batch, inputSize])).dataSync());
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
        optimizer.minimize(
          () => tf.mul(0.5, tf.sum(tf.squaredDifference(forward(x), t))) as Scalar,
          false,
          parameters as Variable[],
        );
      });
    },
  };
}
