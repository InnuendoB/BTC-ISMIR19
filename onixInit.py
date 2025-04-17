import torch
import onnx
from btc_model import *
import os
import argparse
import onnxruntime as ort
print(onnx.__version__)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
config = HParams.load("run_config.yaml")
# 加载训练好的模型
model_file = './test/btc_model_large_voca.pt'
config.feature['large_voca'] = True
config.model['num_chords'] = 170
model = BTC_model(config=config.model).to(device)

# Load model
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    np.savez("norm_stat.npz", mean=checkpoint["mean"], std=checkpoint["std"])
    # 保存为文本文件，方便 C++ 加载
    print("mean:", mean)
    print("std:", std)
    print("mean shape:", np.shape(mean))
    print("std shape:", np.shape(std))
# 创建一个示例输入（确保形状与推理时相同）
dummy_input = torch.randn(1,config.model['timestep'], config.model['feature_size'])  # 例如 (batch_size, feature_dim)


# 导出到 ONNX
onnx_path = "btc_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, 
                  input_names=['CQT'], output_names=['BestIDs','input','input2','input3','input4','input5','input6','input7','input8','input9','input10',
                                                     'input11','input12','input13','input14','input15','input16',"SecondaryIDs"])

print(f"Model exported to {onnx_path}")

session = ort.InferenceSession(onnx_path)
