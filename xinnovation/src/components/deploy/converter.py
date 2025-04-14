from xinnovation.src.core.registry import Registry, CONVERTERS
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
import torch
import lightning.pytorch as L
import os

class BaseConverter(ABC):
    """Base class for model format converters"""
    
    @abstractmethod
    def convert(self, model: L.LightningModule, input_sample: Any, save_path: str) -> str:
        """Convert the model to target format"""
        pass

@CONVERTERS.register_module()
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