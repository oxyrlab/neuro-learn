/** Loads TensorFlow.js with one backend, in isolation, and verifies the backend really is active. */
import { createRequire } from 'node:module';
import path from 'node:path';
import util from 'node:util';
import { pathToFileURL } from 'node:url';
import type { TfBackend } from '../spec';

export type Tf = typeof import('@tensorflow/tfjs');

const EXPECTED_BACKEND_NAME: Record<TfBackend, string> = {
  cpu: 'cpu',
  wasm: 'wasm',
  node: 'tensorflow',
};

/**
 * @tensorflow/tfjs-node 4.22 (the latest release) calls `util.isNullOrUndefined` and friends,
 * which Node removed in v23. Restoring these trivial helpers is a load-time compatibility shim
 * only; it has no effect on the numeric kernels that are measured.
 */
function restoreRemovedUtilHelpers(): void {
  const helpers: Record<string, (value: unknown) => boolean> = {
    isNullOrUndefined: (v) => v === null || v === undefined,
    isNull: (v) => v === null,
    isUndefined: (v) => v === undefined,
    isFunction: (v) => typeof v === 'function',
    isString: (v) => typeof v === 'string',
    isNumber: (v) => typeof v === 'number',
    isBoolean: (v) => typeof v === 'boolean',
    isObject: (v) => v !== null && typeof v === 'object',
    isArray: Array.isArray,
  };
  const target = util as unknown as Record<string, unknown>;
  for (const [name, helper] of Object.entries(helpers))
    if (!(name in target)) target[name] = helper;
}

export async function loadTf(backend: TfBackend): Promise<Tf> {
  let tf: Tf;
  if (backend === 'node') {
    restoreRemovedUtilHelpers();
    tf = (await import('@tensorflow/tfjs-node')) as unknown as Tf;
  } else {
    tf = await import('@tensorflow/tfjs');
    if (backend === 'wasm') {
      const wasm = await import('@tensorflow/tfjs-backend-wasm');
      const pkg = createRequire(import.meta.url).resolve(
        '@tensorflow/tfjs-backend-wasm/package.json',
      );
      wasm.setWasmPaths(pathToFileURL(path.join(path.dirname(pkg), 'dist') + path.sep).href);
    }
    await tf.setBackend(backend);
  }
  await tf.ready();
  if (tf.getBackend() !== EXPECTED_BACKEND_NAME[backend]) {
    throw new Error(
      `Expected backend "${EXPECTED_BACKEND_NAME[backend]}" but got "${tf.getBackend()}".`,
    );
  }
  return tf;
}
