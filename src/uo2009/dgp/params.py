from dataclasses import dataclass

@dataclass(frozen=True)
class EfficientPriceParams:
    T: float = 23400.0              # 6.5 hours in seconds :contentReference[oaicite:6]{index=6}
    dt_internal: float = 1.0        # internal Euler step (seconds)
    lambda1: float = 0.5
    lambda2: float = 0.5
    kappa1: float = 0.006
    kappa2: float = 0.037
    theta1: float = 2.719
    theta2: float = 2.527
    omega1: float = 0.0382
    omega2: float = 0.216
    # correlation process
    kappa3: float = 0.051
    theta3: float = 0.200
    omega3: float = 0.130

@dataclass(frozen=True)
class NoiseParamsMA2:
    nsr1: float = 0.005             # noise-to-signal ratios :contentReference[oaicite:7]{index=7}
    nsr2: float = 0.002
