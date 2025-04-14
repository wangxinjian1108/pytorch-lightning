from xinnovation.src.core.registry import Registry, DEPLOY, CONVERTERS, QUANTIZERS, PRUNERS
import lightning.pytorch as L
from typing import Dict, List, Optional, Any


@DEPLOY.register_module()
class Deployer:
    """Lightning model deployment manager"""
    
    def __init__(self, onnx_converter_cfg: Dict, quantizer_cfg: Dict=None, pruner_cfg: Dict=None):
        self.onnx_converter = CONVERTERS.build(onnx_converter_cfg)
        if quantizer_cfg is not None:
            self.quantizer = QUANTIZERS.build(quantizer_cfg)
        if pruner_cfg is not None:
            self.pruner = PRUNERS.build(pruner_cfg)
            
    def deploy(self, model: L.LightningModule, 
                input_sample: Any, 
                output_dir: str, 
                export_formats: List[str] = ['onnx'], 
                apply_quantization: bool = False, 
                quantization_method: str = 'ptq', 
                calibration_data: Optional[Any] = None, 
                apply_pruning: bool = False, 
                pruning_method: str = 'l1', 
                pruning_amount: float = 0.3) -> Dict[str, str]:
        """Deploy the model"""
        pass
