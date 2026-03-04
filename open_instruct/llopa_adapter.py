from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def install_llopa_modeling(modeling_path: str, model_family: str = "llama") -> None:
    """Load Capsule TRI modeling into the corresponding transformers module path."""
    path = Path(modeling_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LLoPA modeling file not found: {path}")

    family = (model_family or "llama").strip().lower()
    if family == "llama":
        import transformers.models.llama  # noqa: F401

        target_name = "transformers.models.llama.modeling_llama"
        expected = ("LlamaModel", "LlamaForCausalLM")
    elif family == "qwen3":
        import transformers.models.qwen3  # noqa: F401

        target_name = "transformers.models.qwen3.modeling_qwen3"
        expected = ("Qwen3Model", "Qwen3ForCausalLM")
    elif family == "mistral":
        import transformers.models.mistral  # noqa: F401

        target_name = "transformers.models.mistral.modeling_mistral"
        expected = ("MistralModel", "MistralForCausalLM")
    else:
        raise ValueError(f"Unsupported modeling_family for LLoPA: {model_family}")

    if target_name in sys.modules:
        del sys.modules[target_name]
    spec = importlib.util.spec_from_file_location(target_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load TRI modeling spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    spec.loader.exec_module(module)

    for klass in expected:
        if not hasattr(module, klass):
            raise RuntimeError(f"{path} does not define required class: {klass}")
    logger.info("Installed LLoPA modeling from %s into %s", path, target_name)


def _unwrap_model(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    return m


def _get_llopa_step_fn(model):
    candidates = []
    m = _unwrap_model(model)
    candidates.append(m)
    if hasattr(m, "base_model"):
        candidates.append(getattr(m, "base_model"))
    if hasattr(m, "get_base_model"):
        try:
            candidates.append(m.get_base_model())
        except Exception:
            pass
    if hasattr(m, "model"):
        candidates.append(getattr(m, "model"))

    for cand in candidates:
        if cand is not None and hasattr(cand, "llopa_step_logits"):
            return getattr(cand, "llopa_step_logits")
    raise RuntimeError("LLoPA step function not found (missing llopa_step_logits on model).")


def normalize_prompt_messages(messages: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        tmpl = getattr(tokenizer, "chat_template", "") or ""
        if add_generation_prompt and "<|start_header_id|>" in tmpl:
            rendered += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return rendered


def _tokens_from_messages(tokenizer, messages: list[dict[str, str]], device, add_generation_prompt: bool = False):
    rendered = _apply_chat_template(tokenizer, messages, add_generation_prompt)
    ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
    return ids.to(device)


def _build_segments(tokenizer, prefix_messages: list[dict[str, str]], device):
    msgs = normalize_prompt_messages(prefix_messages)
    if not msgs:
        empty = torch.empty((1, 0), dtype=torch.long, device=device)
        return msgs, empty, empty, empty, empty

    su_ids = _tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=False)
    su_gen = _tokens_from_messages(tokenizer, msgs, device, add_generation_prompt=True)

    if msgs[0]["role"] == "system":
        s_ids = _tokens_from_messages(tokenizer, [msgs[0]], device, add_generation_prompt=False)
        user_ids = su_ids[:, s_ids.size(1) :]
    else:
        s_ids = su_ids[:, :0]
        user_ids = su_ids
    assistant_header_delta = su_gen[:, su_ids.size(1) :]
    return msgs, s_ids, user_ids, su_gen, assistant_header_delta


def _assistant_content_delta(tokenizer, prefix_messages: list[dict[str, str]], assistant_text: str, su_gen, device):
    msgs_ass = list(prefix_messages) + [{"role": "assistant", "content": assistant_text}]
    full_ids = _tokens_from_messages(tokenizer, msgs_ass, device, add_generation_prompt=False)
    if full_ids.size(1) <= su_gen.size(1):
        return full_ids[:, :0]
    return full_ids[:, su_gen.size(1) :]


def _llopa_split_system(system_ids: torch.Tensor, system_prefill: str):
    mode = (system_prefill or "full").strip().lower()
    if mode == "full":
        return system_ids, system_ids[:, :0]
    if mode == "no_system":
        if system_ids.size(1) <= 1:
            return system_ids, system_ids[:, :0]
        return system_ids[:, :1], system_ids[:, 1:]
    # "no_bos_system" and fallback: no system tokens in upper path.
    return system_ids[:, :0], system_ids


def _llopa_merge_user(system_ids: torch.Tensor, user_ids: torch.Tensor, system_prefill: str):
    sys_upper, sys_lower_extra = _llopa_split_system(system_ids, system_prefill)
    if sys_lower_extra.size(1) > 0:
        user_llopa = torch.cat([sys_lower_extra, user_ids], dim=1)
    else:
        user_llopa = user_ids
    return sys_upper, user_llopa


class LLOPADataCollator:
    """Token collator + raw message passthrough for LLoPA-specific training."""

    def __init__(self, tokenizer, model, messages_key: str = "messages"):
        self.base = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        self.messages_key = messages_key

    def __call__(self, features):
        messages = [f.get(self.messages_key) for f in features]
        token_keys = {"input_ids", "attention_mask", "labels"}
        stripped = [{k: v for k, v in f.items() if k in token_keys} for f in features]
        batch = self.base(stripped)
        batch[self.messages_key] = messages
        return batch


def compute_llopa_batch_loss(
    model,
    tokenizer,
    batch: dict[str, Any],
    *,
    lower_k: int,
    prefill_mode: str,
    prefill_attn: str,
    system_prefill: str,
    user_prefill: str,
    no_upper_attn: bool,
    loss_scope: str,
    messages_key: str = "messages",
):
    if prefill_mode != "lower":
        raise ValueError("LLoPA requires prefill_mode='lower'.")
    if prefill_attn not in {"causal", "full"}:
        raise ValueError("LLoPA requires prefill_attn in {'causal', 'full'}.")

    messages_batch = batch.get(messages_key)
    if not isinstance(messages_batch, list):
        raise RuntimeError(f"LLoPA batch is missing '{messages_key}' list.")

    step_fn = _get_llopa_step_fn(model)
    device = batch["input_ids"].device
    sample_losses = []

    for sample_messages in messages_batch:
        msgs = normalize_prompt_messages(sample_messages)
        assistant_turns = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not assistant_turns:
            continue
        if loss_scope == "last_turn":
            turn_indices = [assistant_turns[-1]]
        elif loss_scope == "all_assistant":
            turn_indices = assistant_turns
        else:
            raise ValueError(f"Unsupported llopa_loss_scope: {loss_scope}")

        turn_losses = []
        for turn_idx in turn_indices:
            assistant_text = str(msgs[turn_idx].get("content") or "").strip()
            if not assistant_text:
                continue
            prefix_msgs = msgs[:turn_idx]
            _, system_ids, user_ids, su_gen, assistant_header_delta = _build_segments(tokenizer, prefix_msgs, device)
            assistant_delta = _assistant_content_delta(tokenizer, prefix_msgs, assistant_text, su_gen, device)
            if assistant_delta.size(1) < 1:
                continue

            # For generic chat messages we do not have explicit doc/question split.
            if (user_prefill or "full").strip().lower() == "no_question":
                user_prefill_ids = user_ids
                prefix_delta = assistant_header_delta
            else:
                user_prefill_ids = user_ids
                prefix_delta = assistant_header_delta

            sys_upper, user_llopa = _llopa_merge_user(system_ids, user_prefill_ids, system_prefill)
            assistant_ids = torch.cat([prefix_delta, assistant_delta], dim=1)
            if assistant_ids.size(1) < 2:
                continue

            labels = assistant_ids.clone()
            if prefix_delta.size(1) > 0:
                labels[:, : prefix_delta.size(1)] = -100

            out = step_fn(
                system_ids=sys_upper,
                user_ids=user_llopa,
                assistant_ids=assistant_ids,
                lower_k=int(lower_k),
                logits_to_keep=assistant_ids.size(1),
                labels=labels,
                prefill_mode=prefill_mode,
                prefill_attn=prefill_attn,
                no_upper_attn=bool(no_upper_attn),
            )
            if out.loss is not None:
                turn_losses.append(out.loss)

        if turn_losses:
            sample_losses.append(torch.stack(turn_losses).mean())

    if not sample_losses:
        raise RuntimeError("No valid LLoPA losses in current batch (check message formatting).")
    return torch.stack(sample_losses).mean()
