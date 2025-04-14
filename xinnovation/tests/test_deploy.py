import pytest
import torch
import torch.nn as nn
import lightning.pytorch as L
import os
import tempfile
from xinnovation.src.components.deploy.converter import ONNXConverter
from xinnovation.src.components.deploy.deploy import Deployer

class SimpleModel(L.LightningModule):
    """A simple model for testing deployment"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def input_sample():
    return torch.randn(1, 3, 32, 32)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

def test_onnx_converter(model, input_sample, temp_dir):
    """Test ONNX converter functionality"""
    # Create converter
    converter = ONNXConverter()
    
    # Convert model
    save_path = os.path.join(temp_dir, 'model.onnx')
    output_path = converter.convert(model, input_sample, save_path)
    
    # Check if file exists
    assert os.path.exists(output_path)
    
    # Check if file is valid ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    assert onnx_model is not None
    
    # Check input/output shapes
    assert len(onnx_model.graph.input) == 1
    assert len(onnx_model.graph.output) == 1
    
    # Check dynamic axes
    assert 'batch_size' in str(onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param)
    assert 'batch_size' in str(onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param)

def test_deployer(model, input_sample, temp_dir):
    """Test deployer functionality"""
    # Create deployer config
    onnx_converter_cfg = {
        'type': 'ONNXConverter',
        'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        'opset_version': 12
    }
    
    # Create deployer
    deployer = Deployer(onnx_converter_cfg)
    
    # Deploy model
    output_dir = os.path.join(temp_dir, 'deployed')
    os.makedirs(output_dir, exist_ok=True)
    
    result = deployer.deploy(
        model=model,
        input_sample=input_sample,
        output_dir=output_dir,
        export_formats=['onnx']
    )
    
    # Check if ONNX file exists
    assert os.path.exists(os.path.join(output_dir, 'model.onnx'))
    
    # Check if result contains correct paths
    assert 'onnx' in result
    assert os.path.exists(result['onnx'])

def test_deployer_with_quantization(model, input_sample, temp_dir):
    """Test deployer with quantization"""
    # Create deployer config with quantization
    onnx_converter_cfg = {
        'type': 'ONNXConverter',
        'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        'opset_version': 12
    }
    
    quantizer_cfg = {
        'type': 'PTQQuantizer',
        'calibration_method': 'minmax',
        'num_bits': 8
    }
    
    # Create deployer
    deployer = Deployer(onnx_converter_cfg, quantizer_cfg)
    
    # Deploy model with quantization
    output_dir = os.path.join(temp_dir, 'deployed_quantized')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy calibration data
    calibration_data = [torch.randn(1, 3, 32, 32) for _ in range(10)]
    
    result = deployer.deploy(
        model=model,
        input_sample=input_sample,
        output_dir=output_dir,
        export_formats=['onnx'],
        apply_quantization=True,
        quantization_method='ptq',
        calibration_data=calibration_data
    )
    
    # Check if quantized model exists
    assert os.path.exists(os.path.join(output_dir, 'model_quantized.onnx'))
    assert 'onnx_quantized' in result
    assert os.path.exists(result['onnx_quantized'])

def test_deployer_with_pruning(model, input_sample, temp_dir):
    """Test deployer with pruning"""
    # Create deployer config with pruning
    onnx_converter_cfg = {
        'type': 'ONNXConverter',
        'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        'opset_version': 12
    }
    
    pruner_cfg = {
        'type': 'L1Pruner',
        'amount': 0.3
    }
    
    # Create deployer
    deployer = Deployer(onnx_converter_cfg, pruner_cfg=pruner_cfg)
    
    # Deploy model with pruning
    output_dir = os.path.join(temp_dir, 'deployed_pruned')
    os.makedirs(output_dir, exist_ok=True)
    
    result = deployer.deploy(
        model=model,
        input_sample=input_sample,
        output_dir=output_dir,
        export_formats=['onnx'],
        apply_pruning=True,
        pruning_method='l1',
        pruning_amount=0.3
    )
    
    # Check if pruned model exists
    assert os.path.exists(os.path.join(output_dir, 'model_pruned.onnx'))
    assert 'onnx_pruned' in result
    assert os.path.exists(result['onnx_pruned']) 