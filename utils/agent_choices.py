"""Validate finite agent choices before running any tools."""
from utils.vc_names import METHODS, canonical_method


def choice_error(label, value, choices):
    if not isinstance(value, str) or value not in choices:
        return f"Invalid {label}: {value!r}. Available choices: {', '.join(choices)}. Please select one."
    return None


def validate_tool_choices(name, params, tool_names, asr_models):
    problem = choice_error("tool", name, tool_names)
    if problem:
        return problem
    if name == "run_transcription":
        model = params.get("asr_model", "")
        if model != "":
            problem = choice_error("ASR model (asr_model)", model, asr_models)
            if problem:
                return problem + " You may also omit asr_model or use an empty string for language-based selection."
    if name == "run_voice_conversion":
        method = params.get("method")
        if isinstance(method, str):
            method = canonical_method(method)
        return choice_error("VC method", method, METHODS)
    options = {
        "run_mask": {"mask_method": ("noise", "mute"), "detect_method": ("llm", "ner")},
        "check_directory": {"file_type": ("audio", "json", "any")},
    }
    for field, choices in options.get(name, {}).items():
        if field in params:
            problem = choice_error(f"{name}.{field}", params[field], choices)
            if problem:
                return problem
    return None
