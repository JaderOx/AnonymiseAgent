def final_attempt_from(decision, history):
    """Select the evaluated attempt matching the decision, including its parameters."""
    if not history:
        return None
    evaluated = [attempt for attempt in history if attempt.get("metrics")] or history
    if decision:
        method = decision.get("best_vc_method") or decision.get("vc_method")
        for attempt in reversed(evaluated):
            if attempt["vc_method"] != method:
                continue
            if "best_vc_params" not in decision or decision["best_vc_params"] == attempt.get("vc_params", {}):
                return attempt
    return evaluated[-1]


def final_vc_from(decision, history) -> str | None:
    attempt = final_attempt_from(decision, history)
    return attempt["vc_method"] if attempt else None
