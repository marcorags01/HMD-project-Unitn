from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer



def prepare_text(
    prompt,
    tokenizer: PreTrainedTokenizer,
    messages: Optional[List[Dict[str, str]]] = None,
    n_exchanges: int = 2,
):
    if messages is None:
        messages = []

    # Ensure prompt is a string
    prompt = "" if prompt is None else str(prompt)

    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages[-n_exchanges * 2 :],
        tokenize=False,
        add_generation_prompt=True,
    )

    # HARDEN: guarantee a string return type
    if isinstance(text, str):
        return text

    # Some implementations may return token ids (list[int]) or something else.
    if isinstance(text, (list, tuple)) and all(isinstance(t, int) for t in text):
        return tokenizer.decode(text, skip_special_tokens=False)

    return str(text)
