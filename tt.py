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
n_timestep = config.model['timestep']

x = np.loadtxt("dump_feature.txt").reshape(1, n_timestep, 144)
ort_inputs = {input_name: x.astype(np.float32)}
ort_outs = session.run([output_name], ort_inputs)
print(ort_outs)