"""Public method names and input aliases for historical plans and results."""
METHODS = ("formant_lpc", "formant_spec", "pitch", "NeuralVC", "Resyn")
ALIASES = {
    "mcadams": "formant_lpc",
    "formant": "formant_spec",
    "seedvc": "NeuralVC",
    "fishaudio_tts": "Resyn",
}


def canonical_method(name):
    return ALIASES.get(name, name)
