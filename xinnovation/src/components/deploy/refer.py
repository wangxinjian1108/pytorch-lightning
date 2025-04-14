import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
import lightning.pytorch as L
from xinnovation.src.core.registry import Registry, CONVERTERS

# Base Converter Classes


class ONNXConverter(BaseConverter):
    """Convert Lightning models to ONNX format"""
    
    def __init__(self, dynamic_axes: Optional[Dict] = None, opset_version: int = 12):
        self.dynamic_axes = dynamic_axes
        self.opset_version = opset_version
    
    def convert(self, model: L.LightningModule, input_sample: Union[torch.Tensor, Tuple], 
                save_path: str) -> str:
        """
        Convert the Lightning model to ONNX format
        
        Args:
            model: The Lightning module to convert
            input_sample: Example input to trace the model
            save_path: Path to save the ONNX model
            
        Returns:
            Path to the saved ONNX model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Ensure model is in eval mode
        model.eval()
        
        # Handle dynamic axes if not provided
        if self.dynamic_axes is None:
            # Default dynamic axes for batch dimension
            self.dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
        # Extract the forward function from the LightningModule
        if hasattr(model, 'forward_onnx'):
            forward_func = model.forward_onnx
        else:
            forward_func = model.forward
        
        # Export the model to ONNX
        torch.onnx.export(
            model,
            input_sample,
            save_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=self.dynamic_axes,
        )
        
        print(f"Model successfully exported to ONNX: {save_path}")
        return save_path

class TorchScriptConverter(BaseConverter):
    """Convert Lightning models to TorchScript format"""
    
    def __init__(self, method: str = 'trace'):
        """
        Initialize TorchScript converter
        
        Args:
            method: Conversion method, either 'trace' or 'script'
        """
        assert method in ['trace', 'script'], "Method must be either 'trace' or 'script'"
        self.method = method
    
    def convert(self, model: L.LightningModule, input_sample: torch.Tensor, 
                save_path: str) -> str:
        """
        Convert the Lightning model to TorchScript format
        
        Args:
            model: The Lightning module to convert
            input_sample: Example input to trace the model (only used if method='trace')
            save_path: Path to save the TorchScript model
            
        Returns:
            Path to the saved TorchScript model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Ensure model is in eval mode
        model.eval()
        
        # Convert the model to TorchScript
        if self.method == 'trace':
            scripted_model = torch.jit.trace(model, input_sample)
        else:  # script
            scripted_model = torch.jit.script(model)
        
        # Save the TorchScript model
        scripted_model.save(save_path)
        
        print(f"Model successfully exported to TorchScript: {save_path}")
        return save_path

# Quantization Classes
class BaseQuantizer(ABC):
    """Base class for model quantizers"""
    
    @abstractmethod
    def quantize(self, model: Union[L.LightningModule, torch.nn.Module], 
                 calibration_data: Optional[Any] = None) -> torch.nn.Module:
        """Quantize the model"""
        pass

class PTQQuantizer(BaseQuantizer):
    """Post-Training Quantization"""
    
    def __init__(self, dtype: str = 'int8', backend: str = 'fbgemm'):
        """
        Initialize PTQ quantizer
        
        Args:
            dtype: Target data type for quantization ('int8' or 'fp16')
            backend: Backend for quantization ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        self.dtype = dtype
        self.backend = backend
    
    def quantize(self, model: Union[L.LightningModule, torch.nn.Module], 
                calibration_data: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module:
        """
        Apply post-training quantization to the model
        
        Args:
            model: The model to quantize
            calibration_data: DataLoader providing calibration data
            
        Returns:
            Quantized PyTorch model
        """
        # Ensure model is in eval mode
        model.eval()
        
        # Configure quantization backend
        if self.backend == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
        elif self.backend == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        
        # Create a quantization configuration
        if self.dtype == 'int8':
            qconfig = torch.quantization.get_default_qconfig(self.backend)
        else:  # fp16
            qconfig = torch.quantization.float_qparams_weight_only_qconfig
        
        # Prepare the model for quantization
        qconfig_dict = {torch.nn.Linear: qconfig, torch.nn.Conv2d: qconfig}
        prepared_model = torch.quantization.prepare_qat(model, qconfig_dict) if isinstance(model, torch.nn.Module) else torch.quantization.prepare_qat(model.model, qconfig_dict)
        
        # Calibrate the model if calibration data is provided
        if calibration_data:
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    prepared_model(inputs)
        
        # Convert the model to quantized version
        quantized_model = torch.quantization.convert(prepared_model)
        
        print(f"Model successfully quantized using {self.dtype} with {self.backend} backend")
        return quantized_model

# Pruning Classes
class BasePruner(ABC):
    """Base class for model pruners"""
    
    @abstractmethod
    def prune(self, model: Union[L.LightningModule, torch.nn.Module], amount: float) -> torch.nn.Module:
        """Prune the model"""
        pass

class L1Pruner(BasePruner):
    """L1-norm based weight pruning"""
    
    def __init__(self, target_modules: List[str] = None):
        """
        Initialize L1 pruner
        
        Args:
            target_modules: List of module types to prune (e.g., ['Linear', 'Conv2d'])
        """
        self.target_modules = target_modules or ['Linear', 'Conv2d']
    
    def prune(self, model: Union[L.LightningModule, torch.nn.Module], amount: float = 0.3) -> torch.nn.Module:
        """
        Apply L1-norm based pruning to the model
        
        Args:
            model: The model to prune
            amount: Fraction of weights to prune (0.0 to 1.0)
            
        Returns:
            Pruned PyTorch model
        """
        import torch.nn.utils.prune as prune
        
        # Make a copy of the model to avoid modifying the original
        pruned_model = model if isinstance(model, torch.nn.Module) else model.model
        
        # Apply L1 pruning to all specified module types
        for name, module in pruned_model.named_modules():
            if any(module_type in module.__class__.__name__ for module_type in self.target_modules):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Make pruning permanent by removing the parameters
                prune.remove(module, 'weight')
        
        print(f"Model successfully pruned with L1 method. Pruning amount: {amount}")
        return pruned_model

# Main Deployer Class
class Deployer:
    """Lightning model deployment manager"""
    
    def __init__(self):
        self.converters = {
            'onnx': ONNXConverter(),
            'torchscript': TorchScriptConverter()
        }
        self.quantizers = {
            'ptq': PTQQuantizer()
        }
        self.pruners = {
            'l1': L1Pruner()
        }
        
    def register_converter(self, name: str, converter: BaseConverter):
        """Register a new converter"""
        self.converters[name] = converter
        
    def register_quantizer(self, name: str, quantizer: BaseQuantizer):
        """Register a new quantizer"""
        self.quantizers[name] = quantizer
        
    def register_pruner(self, name: str, pruner: BasePruner):
        """Register a new pruner"""
        self.pruners[name] = pruner
    
    def deploy(self, 
               model: L.LightningModule,
               input_sample: Any,
               output_dir: str,
               export_formats: List[str] = ['onnx'],
               apply_quantization: bool = False,
               quantization_method: str = 'ptq',
               calibration_data: Optional[Any] = None,
               apply_pruning: bool = False,
               pruning_method: str = 'l1',
               pruning_amount: float = 0.3) -> Dict[str, str]:
        """
        Deploy a Lightning model with various optimizations
        
        Args:
            model: Lightning module to deploy
            input_sample: Example input for tracing/conversion
            output_dir: Directory to save the exported models
            export_formats: List of target formats ('onnx', 'torchscript', etc.)
            apply_quantization: Whether to apply quantization
            quantization_method: Quantization method to use
            calibration_data: Data for quantization calibration
            apply_pruning: Whether to apply pruning
            pruning_method: Pruning method to use
            pruning_amount: Amount of weights to prune
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Make a copy of the model for deployment
        deploy_model = model
        
        # Apply pruning if requested
        if apply_pruning:
            if pruning_method not in self.pruners:
                raise ValueError(f"Pruning method '{pruning_method}' not registered")
            
            pruner = self.pruners[pruning_method]
            deploy_model = pruner.prune(deploy_model, pruning_amount)
            print(f"Model pruned with {pruning_method} method")
        
        # Apply quantization if requested
        if apply_quantization:
            if quantization_method not in self.quantizers:
                raise ValueError(f"Quantization method '{quantization_method}' not registered")
            
            quantizer = self.quantizers[quantization_method]
            deploy_model = quantizer.quantize(deploy_model, calibration_data)
            print(f"Model quantized with {quantization_method} method")
        
        # Export model to requested formats
        output_paths = {}
        for format_name in export_formats:
            if format_name not in self.converters:
                raise ValueError(f"Export format '{format_name}' not registered")
            
            converter = self.converters[format_name]
            save_path = os.path.join(output_dir, f"model.{format_name}")
            
            # Convert and save the model
            output_path = converter.convert(deploy_model, input_sample, save_path)
            output_paths[format_name] = output_path
        
        return output_paths

# Deploy Pipeline for end-to-end deployment workflow
class DeploymentPipeline:
    """End-to-end deployment pipeline for Lightning models"""
    
    def __init__(self, 
                 model: L.LightningModule,
                 input_shapes: Dict[str, List[int]],
                 output_dir: str,
                 device: str = 'cpu'):
        """
        Initialize deployment pipeline
        
        Args:
            model: Lightning module to deploy
            input_shapes: Dictionary mapping input names to shapes
            output_dir: Output directory for deployed models
            device: Device to run deployment on ('cpu' or 'cuda')
        """
        self.model = model
        self.input_shapes = input_shapes
        self.output_dir = output_dir
        self.device = device
        self.deployer = Deployer()
        
    def prepare_model(self):
        """Prepare the model for deployment"""
        # Ensure model is in eval mode
        self.model.eval()
        self.model = self.model.to(self.device)
        return self.model
    
    def generate_input_samples(self):
        """Generate sample inputs for model tracing"""
        input_samples = {}
        for name, shape in self.input_shapes.items():
            input_samples[name] = torch.randn(*shape).to(self.device)
        
        # If only one input, return just that tensor instead of a dict
        if len(input_samples) == 1:
            return list(input_samples.values())[0]
        return input_samples
    
    def deploy(self, 
               export_formats: List[str] = ['onnx', 'torchscript'],
               quantize: bool = False,
               prune: bool = False,
               benchmark: bool = True) -> Dict[str, str]:
        """
        Run the full deployment pipeline
        
        Args:
            export_formats: Formats to export to
            quantize: Whether to apply quantization
            prune: Whether to apply pruning
            benchmark: Whether to benchmark the deployed models
            
        Returns:
            Dictionary of deployed model paths
        """
        # Prepare model and generate input samples
        model = self.prepare_model()
        input_samples = self.generate_input_samples()
        
        # Deploy the model
        output_paths = self.deployer.deploy(
            model=model,
            input_sample=input_samples,
            output_dir=self.output_dir,
            export_formats=export_formats,
            apply_quantization=quantize,
            apply_pruning=prune
        )
        
        # Benchmark if requested
        if benchmark:
            self._benchmark_models(output_paths, input_samples)
        
        return output_paths
    
    def _benchmark_models(self, model_paths: Dict[str, str], input_samples: Any):
        """Benchmark the deployed models"""
        print("\n=== Deployment Benchmarks ===")
        
        # Benchmark original PyTorch model
        self._benchmark_pytorch_model(input_samples)
        
        # TODO: Add benchmarking for other formats (ONNX, TorchScript, etc.)
        # This would require loading the exported models and measuring inference time
        
    def _benchmark_pytorch_model(self, input_samples: Any, num_iterations: int = 100):
        """Benchmark the original PyTorch model"""
        import time
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_samples)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(input_samples)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
        print(f"PyTorch model inference time: {avg_time:.2f} ms (average over {num_iterations} iterations)")


# Example usage
if __name__ == "__main__":
    # Example Lightning model
    class ExampleModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(32, 10)
            )
            
        def forward(self, x):
            return self.model(x)
    
    # Create deployment pipeline
    model = ExampleModel()
    
    # Create deployment pipeline
    pipeline = DeploymentPipeline(
        model=model,
        input_shapes={"image": [1, 3, 224, 224]},
        output_dir="./deployed_models"
    )
    
    # Run deployment
    deployed_models = pipeline.deploy(
        export_formats=["onnx", "torchscript"],
        quantize=True,
        prune=False,
        benchmark=True
    )
    
    print(f"Deployed models: {deployed_models}")