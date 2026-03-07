from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


LLOPA_SYSTEM_IDS_KEY = "llopa_system_ids"
LLOPA_SYSTEM_MASK_KEY = "llopa_system_attention_mask"
LLOPA_USER_IDS_KEY = "llopa_user_ids"
LLOPA_USER_MASK_KEY = "llopa_user_attention_mask"
LLOPA_ASSISTANT_IDS_KEY = "llopa_assistant_ids"
LLOPA_ASSISTANT_MASK_KEY = "llopa_assistant_attention_mask"
LLOPA_LABELS_KEY = "llopa_labels"
_WARNED_BATCHED_LAST_TURN_FALLBACK = False


def _warn_batched_last_turn_fallback_once(message: str) -> None:
    global _WARNED_BATCHED_LAST_TURN_FALLBACK
    if _WARNED_BATCHED_LAST_TURN_FALLBACK:
        return
    _WARNED_BATCHED_LAST_TURN_FALLBACK = True
    logger.warning(message)


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


def _zero_proxy_loss(model, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Return a differentiable zero scalar while preserving distributed collectives.

    For ZeRO/DDP, ranks must execute a compatible backward graph. When a local rank
    has no valid LLoPA sample, run one standard forward on the batch and zero it out
    so every rank still participates in expected gradient communications.
    """
    token_batch: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "position_ids", "labels"):
        value = batch.get(key)
        if value is not None:
            token_batch[key] = value
    try:
        if token_batch:
            outputs = model(**token_batch, use_cache=False)
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                return torch.nan_to_num(loss.float(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
    except Exception:
        pass

    # Last-resort fallback.
    try:
        base = _unwrap_model(model)
        p = next(base.parameters())
        return p.sum() * 0.0
    except Exception:
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)


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


def _pad_segment_batch(tensors: list[torch.Tensor], pad_value: int):
    batch_size = len(tensors)
    max_len = max((int(t.size(1)) for t in tensors), default=0)
    dtype = tensors[0].dtype if tensors else torch.long
    padded = torch.full((batch_size, max_len), pad_value, dtype=dtype)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, tensor in enumerate(tensors):
        width = int(tensor.size(1))
        if width <= 0:
            continue
        padded[i, :width] = tensor.squeeze(0)
        attention_mask[i, :width] = 1
    return padded, attention_mask


def _build_llopa_last_turn_example(
    tokenizer,
    sample_messages: Any,
    *,
    system_prefill: str,
    user_prefill: str,
):
    device = torch.device("cpu")
    msgs = normalize_prompt_messages(sample_messages)
    assistant_turns = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not assistant_turns:
        return None

    turn_idx = assistant_turns[-1]
    assistant_text = str(msgs[turn_idx].get("content") or "").strip()
    if not assistant_text:
        return None

    prefix_msgs = msgs[:turn_idx]
    _, system_ids, user_ids, su_gen, assistant_header_delta = _build_segments(tokenizer, prefix_msgs, device)
    assistant_delta = _assistant_content_delta(tokenizer, prefix_msgs, assistant_text, su_gen, device)
    if assistant_delta.size(1) < 1:
        return None

    if (user_prefill or "full").strip().lower() == "no_question":
        user_prefill_ids = user_ids
        prefix_delta = assistant_header_delta
    else:
        user_prefill_ids = user_ids
        prefix_delta = assistant_header_delta

    sys_upper, user_llopa = _llopa_merge_user(system_ids, user_prefill_ids, system_prefill)
    assistant_ids = torch.cat([prefix_delta, assistant_delta], dim=1)
    if assistant_ids.size(1) < 2:
        return None

    labels = assistant_ids.clone()
    if prefix_delta.size(1) > 0:
        labels[:, : prefix_delta.size(1)] = -100

    return {
        LLOPA_SYSTEM_IDS_KEY: sys_upper,
        LLOPA_USER_IDS_KEY: user_llopa,
        LLOPA_ASSISTANT_IDS_KEY: assistant_ids,
        LLOPA_LABELS_KEY: labels,
    }


def _batch_llopa_last_turn_examples(tokenizer, messages_batch: list[Any], *, system_prefill: str, user_prefill: str):
    examples = []
    for sample_messages in messages_batch:
        example = _build_llopa_last_turn_example(
            tokenizer,
            sample_messages,
            system_prefill=system_prefill,
            user_prefill=user_prefill,
        )
        if example is None:
            return None
        examples.append(example)

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    system_ids, system_mask = _pad_segment_batch([ex[LLOPA_SYSTEM_IDS_KEY] for ex in examples], pad_token_id)
    user_ids, user_mask = _pad_segment_batch([ex[LLOPA_USER_IDS_KEY] for ex in examples], pad_token_id)
    assistant_ids, assistant_mask = _pad_segment_batch([ex[LLOPA_ASSISTANT_IDS_KEY] for ex in examples], pad_token_id)
    labels, _ = _pad_segment_batch([ex[LLOPA_LABELS_KEY] for ex in examples], -100)
    return {
        LLOPA_SYSTEM_IDS_KEY: system_ids,
        LLOPA_SYSTEM_MASK_KEY: system_mask,
        LLOPA_USER_IDS_KEY: user_ids,
        LLOPA_USER_MASK_KEY: user_mask,
        LLOPA_ASSISTANT_IDS_KEY: assistant_ids,
        LLOPA_ASSISTANT_MASK_KEY: assistant_mask,
        LLOPA_LABELS_KEY: labels,
    }


class LLOPADataCollator:
    """Token collator + raw message passthrough for LLoPA-specific training."""

    def __init__(
        self,
        tokenizer,
        model,
        messages_key: str = "messages",
        *,
        enable_batched_last_turn: bool = False,
        system_prefill: str = "full",
        user_prefill: str = "full",
    ):
        self.base = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        self.tokenizer = tokenizer
        self.messages_key = messages_key
        self.enable_batched_last_turn = bool(enable_batched_last_turn)
        self.system_prefill = str(system_prefill)
        self.user_prefill = str(user_prefill)

    def __call__(self, features):
        messages = [f.get(self.messages_key) for f in features]
        token_keys = {"input_ids", "attention_mask", "labels"}
        stripped = [{k: v for k, v in f.items() if k in token_keys} for f in features]
        batch = self.base(stripped)
        batch[self.messages_key] = messages
        if self.enable_batched_last_turn:
            batched_inputs = _batch_llopa_last_turn_examples(
                self.tokenizer,
                messages,
                system_prefill=self.system_prefill,
                user_prefill=self.user_prefill,
            )
            if batched_inputs is not None:
                batch.update(batched_inputs)
            else:
                _warn_batched_last_turn_fallback_once(
                    "Falling back to generic LLoPA last_turn path because batched segment construction failed."
                )
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

    batched_system_ids = batch.get(LLOPA_SYSTEM_IDS_KEY)
    if loss_scope == "last_turn" and batched_system_ids is not None and prefill_attn == "causal":
        out = step_fn(
            system_ids=batched_system_ids.to(device=device),
            system_attention_mask=batch[LLOPA_SYSTEM_MASK_KEY].to(device=device),
            user_ids=batch[LLOPA_USER_IDS_KEY].to(device=device),
            user_attention_mask=batch[LLOPA_USER_MASK_KEY].to(device=device),
            assistant_ids=batch[LLOPA_ASSISTANT_IDS_KEY].to(device=device),
            assistant_attention_mask=batch[LLOPA_ASSISTANT_MASK_KEY].to(device=device),
            lower_k=int(lower_k),
            logits_to_keep=batch[LLOPA_ASSISTANT_IDS_KEY].size(1),
            labels=batch[LLOPA_LABELS_KEY].to(device=device),
            prefill_mode=prefill_mode,
            prefill_attn=prefill_attn,
            no_upper_attn=bool(no_upper_attn),
        )
        if out.loss is not None:
            return out.loss
    elif loss_scope == "last_turn" and prefill_attn == "causal":
        _warn_batched_last_turn_fallback_once(
            "Using generic LLoPA last_turn loss path because prebatched segment tensors are unavailable."
        )

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
        logger.warning("No valid LLoPA losses in current batch; skipping this batch.")
        return _zero_proxy_loss(model, batch, device)
    return torch.stack(sample_losses).mean()


def compute_llopa_batch_loss_streaming_backward(
    model,
    tokenizer,
    batch: dict[str, Any],
    *,
    backward_fn,
    lower_k: int,
    prefill_mode: str,
    prefill_attn: str,
    system_prefill: str,
    user_prefill: str,
    no_upper_attn: bool,
    loss_scope: str,
    messages_key: str = "messages",
):
    """Compute all_assistant loss with per-turn backward to lower peak memory.

    This preserves the original weighting:
      mean_i( mean_t( loss_{i,t} ) )
    by scaling each backward term with 1 / (num_valid_samples * turns_in_sample_i).
    """
    if prefill_mode != "lower":
        raise ValueError("LLoPA requires prefill_mode='lower'.")
    if prefill_attn not in {"causal", "full"}:
        raise ValueError("LLoPA requires prefill_attn in {'causal', 'full'}.")
    if loss_scope != "all_assistant":
        raise ValueError("Streaming backward path only supports loss_scope='all_assistant'.")

    messages_batch = batch.get(messages_key)
    if not isinstance(messages_batch, list):
        raise RuntimeError(f"LLoPA batch is missing '{messages_key}' list.")

    step_fn = _get_llopa_step_fn(model)
    device = batch["input_ids"].device
    prepared_samples: list[list[dict[str, torch.Tensor]]] = []

    # First pass: build all valid turn tensors and count valid samples.
    for sample_messages in messages_batch:
        msgs = normalize_prompt_messages(sample_messages)
        assistant_turns = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not assistant_turns:
            continue

        prepared_turns: list[dict[str, torch.Tensor]] = []
        for turn_idx in assistant_turns:
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

            prepared_turns.append(
                {
                    "system_ids": sys_upper,
                    "user_ids": user_llopa,
                    "assistant_ids": assistant_ids,
                    "labels": labels,
                }
            )

        if prepared_turns:
            prepared_samples.append(prepared_turns)

    if not prepared_samples:
        logger.warning("No valid LLoPA losses in current batch; skipping this batch.")
        return _zero_proxy_loss(model, batch, device)

    num_valid_samples = len(prepared_samples)
    total_loss_detached = torch.zeros((), device=device, dtype=torch.float32)

    # Second pass: backward per turn with exact objective scaling.
    for sample_turns in prepared_samples:
        turns_in_sample = len(sample_turns)
        sample_loss_detached = torch.zeros((), device=device, dtype=torch.float32)
        for turn in sample_turns:
            out = step_fn(
                system_ids=turn["system_ids"],
                user_ids=turn["user_ids"],
                assistant_ids=turn["assistant_ids"],
                lower_k=int(lower_k),
                logits_to_keep=turn["assistant_ids"].size(1),
                labels=turn["labels"],
                prefill_mode=prefill_mode,
                prefill_attn=prefill_attn,
                no_upper_attn=bool(no_upper_attn),
            )
            if out.loss is None:
                continue
            scaled_loss = out.loss / (num_valid_samples * turns_in_sample)
            backward_fn(scaled_loss)
            sample_loss_detached = sample_loss_detached + (out.loss.detach().float() / turns_in_sample)
        total_loss_detached = total_loss_detached + sample_loss_detached

    return total_loss_detached / num_valid_samples
