/* Ducky credits. Hybrid TFJS loader: LayersModel or GraphModel. */
import * as tf from '@tensorflow/tfjs';

let ready = false;
// one of these will be set
let layers: tf.LayersModel | null = null;
let graph: tf.GraphModel | null = null;

function assertInput(x: tf.Tensor4D) {
	const s = x.shape;
	if (s[0] !== 1 || s[1] !== 160 || s[2] !== 160 || s[3] !== 3) {
		throw new Error(
			`[ducky] bad input tensor shape ${JSON.stringify(s)}; expected [1,160,160,3]`
		);
	}
}

export async function initModel() {
	if (ready) return;
	await tf.setBackend('webgl');
	await tf.ready();
	console.info('[ducky] tf backend:', tf.getBackend());

	try {
		console.info('[ducky] trying LayersModel …');
		layers = await tf.loadLayersModel('/model/model.json');
		console.info(
			'[ducky] layers model loaded:',
			layers.inputs[0].shape,
			'->',
			layers.outputs[0].shape
		);
		// warm
		tf.tidy(() => {
			const z = tf.zeros([1, 160, 160, 3]);
			(layers!.predict(z) as tf.Tensor).dataSync();
		});
		console.info('[ducky] warm layers ok');
	} catch (e) {
		console.warn('[ducky] layers load failed, trying GraphModel:', e);
		graph = await tf.loadGraphModel('/model/model.json');
		console.info(
			'[ducky] graph model loaded with signature keys:',
			Object.keys((graph as any).executor?._signature?.inputs || {}),
			'->',
			Object.keys((graph as any).executor?._signature?.outputs || {})
		);
		// warm
		tf.tidy(() => {
			const z = tf.zeros([1, 160, 160, 3]);
			(graph!.predict(z) as tf.Tensor).dataSync();
		});
		console.info('[ducky] warm graph ok');
	}

	ready = true;
}

/** returns 0..1 Walmart confidence */
export async function predictTensor(x: tf.Tensor4D): Promise<number> {
	assertInput(x);
	if (!ready) throw new Error('model not ready');

	if (layers) {
		const y = tf.tidy(() => layers!.predict(x) as tf.Tensor);
		const v = (await y.data())[0];
		y.dispose();
		return v;
	}

	// GraphModel path
	// most SavedModels expose a single float output
	const y = tf.tidy(() => graph!.predict(x) as tf.Tensor);
	const v = (await y.data())[0];
	y.dispose();
	return v;
}

export function isModelReady() {
	return ready;
}
