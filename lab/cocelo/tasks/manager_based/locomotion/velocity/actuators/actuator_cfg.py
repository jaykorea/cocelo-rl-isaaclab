from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import (
    IdealPDActuatorCfg as BaseIdealPDActuatorCfg, DelayedPDActuatorCfg
)
from .gear_actuator import GearDelayedPDActuator
from .actuator_force_zero import ForceZeroActuator

@configclass
class ForceZeroActuatorCfg(BaseIdealPDActuatorCfg):
    """Configuration for LSTM-based actuator model."""

    class_type: type = ForceZeroActuator

@configclass
class GearDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for GearDelayedPDActuator."""

    class_type: type = GearDelayedPDActuator

    gear_ratio: float = MISSING
    
    gamma: float = MISSING

