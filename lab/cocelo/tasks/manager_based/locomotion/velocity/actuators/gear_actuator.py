from __future__ import annotations
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils.types import ArticulationActions

class GearDelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with an additional gear ratio and gamma scaling on applied torques."""

    cfg: "GearDelayedPDActuatorCfg"

    def compute(
        self, control_action: ArticulationActions, joint_pos, joint_vel
    ) -> ArticulationActions:
        joint_pos = joint_pos * self.cfg.gear_ratio # OBS_{joint space} -> OBS_{motor space}
        joint_vel = joint_vel * self.cfg.gear_ratio # OBS_{joint space} -> OBS_{motor space}
        # 먼저 DelayedPDActuator의 compute를 호출해서 delayed torque 계산
        control_action = super().compute(control_action, joint_pos, joint_vel) # motor space에서 torque 계산

        # 여기서 gear ratio를 곱해줌
        if self.cfg.gear_ratio != 1.0:
            self.computed_effort = self.computed_effort * self.cfg.gear_ratio * self.cfg.gamma # TORQUE_{motor space} -> TORQUE_{joint space}
            self.applied_effort = self._clip_effort(self.computed_effort)
            control_action.joint_efforts = self.applied_effort

        return control_action
    
class CoupledDelayedPDActuator(DelayedPDActuator):
    """
    2-DOF coupled actuator.

    Virtual joint space:
        q = [pitch, roll]

    Motor space:
        m = [motor_1, motor_2]

    Coupling:
        left ankle:
            m1 = -g1 * (roll - pitch)
            m2 = -g2 * (roll + pitch)
        right ankle:
            m1 = -g1 * (roll + pitch)
            m2 = -g2 * (roll - pitch)
        torso:
            m1 = g1 * (roll - pitch)
            m2 = -g2 * (roll + pitch)

    Torque mapping:
        tau_joint = J^T tau_motor

    Note:
        Action slots are interpreted as motor-space targets.
        For example:
            pitch_joint slot -> motor_1 target
            roll_joint slot  -> motor_2 target
    """

    cfg: "CoupledDelayedPDActuatorCfg"

    def __init__(self, cfg: "CoupledDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        joint_names = getattr(self, "joint_names", None)
        if joint_names is None:
            joint_names = getattr(self, "_joint_names")
        joint_names = list(joint_names)

        def has_joint(name: str) -> bool:
            return name in joint_names

        def get_idx(name: str) -> int:
            if name not in joint_names:
                raise RuntimeError(f"{name} not found in actuator joint names: {joint_names}")
            return joint_names.index(name)

        self._pairs = {}

        # left ankle: [pitch, roll]
        if has_joint("left_ankle_pitch_joint") and has_joint("left_ankle_roll_joint"):
            self._pairs["left_ankle"] = {
                "ids": torch.tensor(
                    [
                        get_idx("left_ankle_pitch_joint"),
                        get_idx("left_ankle_roll_joint"),
                    ],
                    dtype=torch.long,
                    device=self._device,
                ),
                "coupling": "roll_sum",
            }

        # right ankle: [pitch, roll]
        if has_joint("right_ankle_pitch_joint") and has_joint("right_ankle_roll_joint"):
            self._pairs["right_ankle"] = {
                "ids": torch.tensor(
                    [
                        get_idx("right_ankle_pitch_joint"),
                        get_idx("right_ankle_roll_joint"),
                    ],
                    dtype=torch.long,
                    device=self._device,
                ),
                "coupling": "roll_sum",
            }

        # torso: [pitch, roll]
        if has_joint("torso_pitch_joint") and has_joint("torso_roll_joint"):
            self._pairs["torso"] = {
                "ids": torch.tensor(
                    [
                        get_idx("torso_pitch_joint"),
                        get_idx("torso_roll_joint"),
                    ],
                    dtype=torch.long,
                    device=self._device,
                ),
                "coupling": "roll_sum",
            }

        if len(self._pairs) == 0:
            raise RuntimeError(
                f"No coupled joint pair found in actuator joint names: {joint_names}"
            )

        self._g1 = float(self.cfg.gear_ratio_1)
        self._g2 = float(self.cfg.gear_ratio_2)
        self._gamma = float(self.cfg.gamma)

    def _joint_to_motor(self, joint_value: torch.Tensor) -> torch.Tensor:
        motor_value = torch.zeros_like(joint_value)

        for pair_name, pair in self._pairs.items():
            ids = pair["ids"]
            coupling = pair["coupling"]

            q_pair = joint_value[:, ids]
            pitch = q_pair[:, 0]
            roll = q_pair[:, 1]

            if coupling == "roll_sum":
                if pair_name == "left_ankle":
                    motor_1 = -self._g1 * (roll - pitch)
                    motor_2 = -self._g2 * (roll + pitch)
                elif pair_name == "right_ankle":
                    motor_1 = -self._g1 * (roll + pitch)
                    motor_2 = -self._g2 * (roll - pitch)
                elif pair_name == "torso":
                    motor_1 = self._g1 * (roll - pitch)
                    motor_2 = -self._g2 * (roll + pitch)
                else:
                    raise RuntimeError(f"Unsupported coupled actuator pair: {pair_name}")
            else:
                raise RuntimeError(f"Unsupported coupled actuator mapping: {coupling}")

            motor_value[:, ids[0]] = motor_1
            motor_value[:, ids[1]] = motor_2

        return motor_value

    def _motor_to_joint_torque(self, motor_tau: torch.Tensor) -> torch.Tensor:
        joint_tau = torch.zeros_like(motor_tau)

        for pair_name, pair in self._pairs.items():
            ids = pair["ids"]
            coupling = pair["coupling"]

            tau_pair = motor_tau[:, ids]
            tau_m1 = tau_pair[:, 0]
            tau_m2 = tau_pair[:, 1]

            if coupling == "roll_sum":
                if pair_name == "left_ankle":
                    tau_pitch = self._g1 * tau_m1 - self._g2 * tau_m2
                    tau_roll = -self._g1 * tau_m1 - self._g2 * tau_m2
                elif pair_name == "right_ankle":
                    tau_pitch = -self._g1 * tau_m1 + self._g2 * tau_m2
                    tau_roll = -self._g1 * tau_m1 - self._g2 * tau_m2
                elif pair_name == "torso":
                    tau_pitch = -self._g1 * tau_m1 - self._g2 * tau_m2
                    tau_roll = self._g1 * tau_m1 - self._g2 * tau_m2
                else:
                    raise RuntimeError(f"Unsupported coupled actuator pair: {pair_name}")
            else:
                raise RuntimeError(f"Unsupported coupled actuator mapping: {coupling}")

            joint_tau[:, ids[0]] = tau_pitch
            joint_tau[:, ids[1]] = tau_roll

        return joint_tau

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:

        # Isaac virtual joint observation -> motor-space observation
        motor_pos = self._joint_to_motor(joint_pos)
        motor_vel = self._joint_to_motor(joint_vel)

        # DelayedPDActuator computes motor-space torque.
        # control_action.joint_positions is assumed to be motor-space target.
        control_action = super().compute(control_action, motor_pos, motor_vel)

        # motor torque -> Isaac virtual joint torque
        motor_tau = self.computed_effort
        joint_tau = self._motor_to_joint_torque(motor_tau) * self._gamma

        self.computed_effort = joint_tau
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort

        return control_action
