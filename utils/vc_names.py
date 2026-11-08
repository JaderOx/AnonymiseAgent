"""Canonical public method names; legacy CLI names remain accepted."""
ALIASES = {"mcadams": "formant_lpc", "formant": "formant_spec",
           "seedvc": "NeuralVC", "fishaudio_tts": "Resyn"}


def canonical_method(name):
    return ALIASES.get(name, name)
