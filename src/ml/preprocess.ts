/*
 Ducky credits:
 Image in. Tensor out. Resize to 160. Normalize 0..1. Channels last.
 Returns Tensor4D [1,160,160,3].
*/
import * as tf from '@tensorflow/tfjs';

export function imageDataToTensor(img: ImageData): tf.Tensor4D {
	return tf.tidy(() => {
		const t = tf.browser.fromPixels(img);
		const target = tf.image.resizeBilinear(t, [160, 160], true);
		const float = target.toFloat().div(255);
		const batched = float.expandDims(0) as tf.Tensor4D;
		return batched;
	});
}
