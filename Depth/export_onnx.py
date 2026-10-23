"""Modellarni ONNX ga eksport: APK ichida ishlatish uchun. py export_onnx.py"""
import torch
from transformers import AutoModelForDepthEstimation, AutoModelForSemanticSegmentation

class Seg(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).logits

class Dep(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(pixel_values=x).predicted_depth

seg = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024").eval()
dep = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").eval()
torch.onnx.export(Seg(seg), torch.randn(1, 3, 160, 352), "../App/Assets/seg.onnx", input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False)
torch.onnx.export(Dep(dep), torch.randn(1, 3, 112, 224), "../App/Assets/depth.onnx", input_names=["x"], output_names=["depth"], opset_version=17, dynamo=False)
