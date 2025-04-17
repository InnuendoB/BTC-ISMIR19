import os
import argparse
import numpy as np
import warnings
from btc_model import *
import onnxruntime as ort
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from utils import logger
import mir_eval

warnings.filterwarnings('ignore')
logger.logging_verbosity(1)

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--audio_dir', type=str, default='./test')
parser.add_argument('--save_dir', type=str, default='./test')
parser.add_argument('--onnx_path', type=str, default='btc_model.onnx')
args = parser.parse_args()

config = HParams.load("run_config.yaml")


config.feature['large_voca'] = True
config.model['num_chords'] = 170
idx_to_chord = idx2voca_chord()
logger.info("label type: large voca")


# Load ONNX model
session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load normalization stats
norm_stat = np.load('norm_stat.npz')  # Save mean/std from training
mean = norm_stat['mean']
std = norm_stat['std']

# Process audio files
audio_paths = get_audio_paths(args.audio_dir)

for i, audio_path in enumerate(audio_paths):
    logger.info("======== %d of %d in progress ========" % (i + 1, len(audio_paths)))

    # Feature extraction
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
    logger.info("audio file loaded and feature computation success : %s" % audio_path)

    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep) if (feature.shape[0] % n_timestep) != 0 else 0
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    predictions = []
    for t in range(num_instance):
        input_chunk = feature[t * n_timestep : (t + 1) * n_timestep]
        input_chunk = input_chunk[np.newaxis, :, :]  # Shape: (1, 108, 144)
        debugarray = input_chunk.flatten()
        array = np.array(debugarray)
        np.savetxt("debug.txt", array, fmt="%f")
        ort_inputs = {input_name: input_chunk.astype(np.float32)}
        ort_outs = session.run([output_name], ort_inputs)
        predictions.append(ort_outs[0])

    predictions = np.concatenate(predictions, axis=0).squeeze()  # Shape: (total_time_steps,)

    # Post-processing to lab file
    start_time = 0.0
    lines = []
    for i, pred in enumerate(predictions):
        if i == 0:
            prev_chord = pred
            continue
        if pred != prev_chord:
            lines.append('%.3f %.3f %s\n' % (start_time, time_unit * i, idx_to_chord[int(prev_chord)]))
            start_time = time_unit * i
            prev_chord = pred
    lines.append('%.3f %.3f %s\n' % (start_time, time_unit * len(predictions), idx_to_chord[int(prev_chord)]))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(
        args.save_dir,
        os.path.split(audio_path)[-1].replace('.mp3', '').replace('.wav', '') + '.lab'
    )
    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved : %s" % save_path)
