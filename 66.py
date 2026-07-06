import onnx

model = onnx.load("D:\\Anti_depression\\emotion-ferplus-8.onnx")

# Opset
print(f"Opset: {model.opset_import[0].version}")

# 输入
inp = model.graph.input[0]
shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
print(f"Input:  name={inp.name}, shape={shape}")  # 预期: [0, 1, 64, 64]

# 输出
out = model.graph.output[0]
shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
print(f"Output: name={out.name}, shape={shape}")  # 预期: [0, 8]