"""Anonymization controls shared by CASA prompts and the conversion backends."""
import hashlib
import json
import math

VC_DEFAULTS = {
    "formant_lpc": {"alpha": 0.8},
    "formant_spec": {"scale": 1.1},
    "pitch": {"n_steps": -3.0},
    "NeuralVC": {
        "diffusion_steps": 30,
        "length_adjust": 1.0,
        "intelligibility_cfg_rate": 0.7,
        "similarity_cfg_rate": 0.7,
    },
    "Resyn": {"fishaudio_emotion": "Suffocating"},
}

VC_PARAMETER_GUIDE = (
    "vc_params is an optional JSON object. Use the defaults unless the request "
    "or evaluation suggests an adjustment. Include only keys for the chosen method. "
    "n_steps is the pitch shift in semitones (speaker metadata may set the direction), "
    "alpha is the McAdams coefficient, and scale is the spectral formant multiplier. "
    "NeuralVC accepts the diffusion step count, duration multiplier and guidance "
    "strengths for intelligibility and speaker similarity. Resyn accepts a Fish Speech "
    "emotion label or an empty string to disable the tag. Defaults by method:\n"
    + json.dumps(VC_DEFAULTS, indent=2)
)


def resolve_vc_params(method, overrides=None):
    """Validate model output and fill omissions with the actual backend defaults."""
    if method not in VC_DEFAULTS:
        raise ValueError(f"Unknown CASA voice conversion method: {method!r}")
    values = VC_DEFAULTS[method].copy()
    if overrides is None:
        return values
    if not isinstance(overrides, dict):
        raise ValueError("vc_params must be a JSON object")
    for name, value in overrides.items():
        if name not in values:
            raise ValueError(f"{method} has no parameter {name!r}. Available: {', '.join(values)}")
        default = values[name]
        if isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        elif type(default) is int:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name in {"alpha", "scale", "formant_scale", "length_adjust"} and value <= 0:
                raise ValueError(f"{name} must be positive")
            value = float(value)
        values[name] = value
    return values


def vc_output_name(method, params):
    """Keep the standard directory for defaults and isolate each tuned variant."""
    if not params or params == VC_DEFAULTS[method]:
        return method
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return f"{method}_{hashlib.sha256(encoded).hexdigest()[:16]}"
