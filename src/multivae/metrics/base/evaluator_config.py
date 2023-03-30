from pydantic.dataclasses import dataclass


@dataclass
class EvaluatorConfig:
    """
    
    Base config class for the evaluation modules. 
    
    Args :
        batch_size (int) : The batch size to use in the evaluation.
    
    
    """
    batch_size: int = 512
