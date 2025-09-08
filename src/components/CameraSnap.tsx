/* 
 Ducky credits:
 Camera snap and tiny ML glue by Ducky for Arth.
 Mobile first. Opens camera on user tap, snaps, resizes with Pica, feeds TFJS stub.
*/
import { useEffect, useRef, useState } from 'react';
import Pica from 'pica';
import { imageDataToTensor } from '../ml/preprocess';
import { initModel, predictTensor } from '../ml/model';
import * as tf from '@tensorflow/tfjs';

const pica = new Pica();

export default function CameraSnap() {
	const videoRef = useRef<HTMLVideoElement | null>(null);
	const workCanvas = useRef<HTMLCanvasElement | null>(null);
	const [streamErr, setStreamErr] = useState<string | null>(null);
	const [verdict, setVerdict] = useState<string>('');
	const [tone, setTone] = useState<'green' | 'yellow' | 'red' | null>(null);
	const [armed, setArmed] = useState(false);

	useEffect(() => {
		let live = true;
		async function boot() {
			try {
				await initModel();
				const s = await navigator.mediaDevices.getUserMedia({
					video: { facingMode: 'environment' },
					audio: false,
				});
				if (!live) return;
				if (videoRef.current) {
					videoRef.current.srcObject = s;
					await videoRef.current.play();
				}
			} catch (e: any) {
				setStreamErr(e?.message || 'camera failed');
			}
		}
		if (armed) boot();
		return () => {
			live = false;
			const s = videoRef.current?.srcObject as MediaStream | undefined;
			s?.getTracks().forEach(t => t.stop());
		};
	}, [armed]);

	async function snap() {
		const v = videoRef.current;
		const c = workCanvas.current;
		if (!v || !c) return;

		// draw to canvas first
		const w = v.videoWidth || 1280;
		const h = v.videoHeight || 720;
		c.width = 320;
		c.height = Math.round((320 / w) * h);

		const ctx = c.getContext('2d')!;
		ctx.drawImage(v, 0, 0, c.width, c.height);

		// extract ImageData at 160 for the model
		const tmp = document.createElement('canvas');
		tmp.width = 160;
		tmp.height = 160;
		await pica.resize(c, tmp);
		const tctx = tmp.getContext('2d')!;
		const img = tctx.getImageData(0, 0, 160, 160);

		const x = imageDataToTensor(img);
		const conf = await predictTensor(x as tf.Tensor4D);
		x.dispose();

		const pct = conf * 100;
		if (conf >= 0.5) {
			setTone('green');
			setVerdict(`Good chance that's wal-mart bro • ${pct.toFixed(1)}%`);
		} else if (conf >= 0.4) {
			setTone('yellow');
			setVerdict(`Probably Walmart • ${pct.toFixed(1)}%`);
		} else {
			setTone('red');
			setVerdict(`Not Walmart • ${pct.toFixed(1)}%`);
		}
	}

	return (
		<div style={{ display: 'grid', gap: 12 }}>
			{streamErr && <div style={{ color: 'crimson' }}>{streamErr}</div>}

			{!armed && (
				<button
					onClick={() => setArmed(true)}
					style={{ padding: '0.8rem 1rem' }}
				>
					Start Camera
				</button>
			)}

			<video
				ref={videoRef}
				playsInline
				muted
				style={{ width: '100%', borderRadius: 12 }}
			/>

			<button
				onClick={snap}
				disabled={!armed}
				style={{ padding: '0.8rem 1rem' }}
			>
				Snap
			</button>

			{verdict && (
				<div
					style={{
						marginTop: 8,
						padding: '0.6rem 0.9rem',
						borderRadius: 12,
						textAlign: 'center',
						fontWeight: 700,
						border: `1px solid ${
							tone === 'green' ? 'rgb(20,160,80)'
							: tone === 'yellow' ? 'rgb(180,140,20)'
							: 'rgb(200,60,60)'
						}`,
						background:
							tone === 'green' ? 'rgba(20,160,80,0.12)'
							: tone === 'yellow' ? 'rgba(180,140,20,0.15)'
							: 'rgba(200,60,60,0.12)',
						color:
							tone === 'green' ? 'rgb(10,120,60)'
							: tone === 'yellow' ? 'rgb(130,100,20)'
							: 'rgb(160,40,40)',
					}}
				>
					{verdict}
				</div>
			)}

			<canvas
				ref={workCanvas}
				style={{ display: 'none' }}
			/>
		</div>
	);
}
