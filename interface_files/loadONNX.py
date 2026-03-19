import onnx

model = onnx.load("C:\\Users\\Kenneth\\.insightface\\models\\buffalo_sc\\det_500m.onnx")
graph = model.graph

# Add every node's output as a model output
for node in graph.node:
    for output_name in node.output:
        graph.output.extend([
            onnx.ValueInfoProto(name=output_name)
        ])

onnx.save(model, "det_500m_intermediate.onnx")