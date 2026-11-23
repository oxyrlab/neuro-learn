import type { Dataset } from '../dataset';
import type { ImplSpec, Scenario, Subject } from '../spec';
import { buildNeuro } from './neuro';
import { loadTf } from './tf-load';
import { buildTfLayers } from './tf-layers';
import { buildTfOps } from './tf-ops';

export async function buildSubject(
  impl: ImplSpec,
  sc: Scenario,
  init: Float32Array,
  data: Dataset,
): Promise<Subject> {
  if (impl.family === 'neuro') return buildNeuro(sc, init, data);
  const tf = await loadTf(impl.backend as NonNullable<ImplSpec['backend']>);
  return impl.api === 'ops' ? buildTfOps(tf, sc, init, data) : buildTfLayers(tf, sc, init, data);
}
