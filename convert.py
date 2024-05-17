import onnx
import onnx_tf
import tensorflow as tf
import ultralytics

def convert_torch_to_tf(weights_path):
    # Load YOLOv8 model
    model = ultralytics.YOLO(weights_path)  # Replace with your model path

    # Convert to ONNX
    success = model.export(format='onnx')
    if not success:
        raise RuntimeError("ONNX export failed")
    onnx_model = onnx.load(f'{weights_path}.onnx')
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(f"{weights_path}/tf_model")
    
    return tf.saved_model.load(f"{weights_path}/tf_model")
