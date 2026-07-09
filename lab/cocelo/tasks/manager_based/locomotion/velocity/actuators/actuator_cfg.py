from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg as BaseIdealPDActuatorCfg
from .gear_actuator import GearDelayedPDActuator, CoupledDelayedPDActuator
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


@configclass
class CoupledDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = CoupledDelayedPDActuator

    # Signed external gear ratios.
    # ankle: 외부기어 1:2 + 반대방향이면 -2.0
    # torso: 외부기어 없으면 1.0 또는 방향 반대면 -1.0
    gear_ratio_1: float = MISSING
    gear_ratio_2: float = MISSING

    gamma: float = 1.0
