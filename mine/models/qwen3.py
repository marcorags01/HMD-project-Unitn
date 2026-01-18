from typing import Dict, List, Optional, Any
from transformers import PreTrainedTokenizer, BatchEncoding

def _flatten_token_ids(x: Any) -> List[int]:
    # torch tensor -> python list
    if hasattr(x, "detach") and hasattr(x, "tolist"):
        x = x.detach().cpu().tolist()

    # batched [[...]] -> take first row
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = x[0]

    if isinstance(x, (list, tuple)):
        out: List[int] = []
        for t in x:
            out.append(int(t))  # int-like scalars supported
        return out

    return [int(x)]  # scalar int-like

def prepare_text(
    prompt,
    tokenizer: PreTrainedTokenizer,
    messages: Optional[List[Dict[str, str]]] = None,
    n_exchanges: int = 2,
):
    if messages is None:
        messages = []

    prompt = "" if prompt is None else str(prompt)
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages[-n_exchanges * 2 :],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Expected path
    if isinstance(text, str):
        return text

    # Some remote-code implementations might return pre-tokenized objects
    if isinstance(text, BatchEncoding):
        ids = text.data.get("input_ids", None)
        if ids is None:
            raise TypeError("apply_chat_template returned BatchEncoding without input_ids")
        return tokenizer.decode(_flatten_token_ids(ids), skip_special_tokens=False)

    if isinstance(text, dict) and "input_ids" in text:
        return tokenizer.decode(_flatten_token_ids(text["input_ids"]), skip_special_tokens=False)

    # Token-id lists/tuples (possibly nested)
    if isinstance(text, (list, tuple)):
        try:
            return tokenizer.decode(_flatten_token_ids(text), skip_special_tokens=False)
        except Exception as e:
            raise TypeError(f"apply_chat_template returned an unsupported list/tuple shape: {type(text)}") from e

    # Fail loudly (do NOT silently str() unknown objects)
    raise TypeError(f"apply_chat_template returned unsupported type: {type(text)}")
