def final_vc_from(decision, history) -> str | None:
    """最终方法：LLM done=true 的裁决（best_vc_method，或回退分支选的已试方法）
    优先；无裁决或裁决不在历史中时回落最后一次尝试。

    裁决必须是已尝试过的方法（其指标在 history 中），否则视为 LLM 幻觉回落。
    """
    if not history:
        return None
    last = history[-1]["vc_method"]
    if not decision:
        return last
    best = decision.get("best_vc_method") or decision.get("vc_method")
    if best and any(h["vc_method"] == best for h in history):
        return best
    return last


