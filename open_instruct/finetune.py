# !/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
import contextlib
import importlib.util
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import datasets
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, get_scheduler
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, utils
from open_instruct.dataset_transformation import (
    ASSISTANT_HEADER_START_KEY,
    ASSISTANT_HEADER_START_MASK_KEY,
    ASSISTANT_HEADER_STARTS_KEY,
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.llopa_adapter import (
    LLOPADataCollator,
    PREFILL_LOWER_SYSTEM_LEN_KEY,
    compute_prefill_lower_freeze_batch_loss,
    compute_prefill_lower_solo_bos_batch_loss,
    compute_prefill_lower_solo_batch_loss,
    compute_llopa_batch_loss,
    compute_llopa_batch_loss_streaming_backward,
    get_prefill_lower_system_len,
    install_llopa_modeling,
    normalize_system_prefill,
)
from open_instruct.model_utils import push_folder_to_hub, save_lora_adapter_from_zero_checkpoint, save_with_accelerate
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    truncate_wandb_tag,
)

logger = get_logger(__name__)


FUSION_TOKEN_TEMPLATE = "<|FUSION{}|>"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
QUESTION_ID_KEY = "question_id"


class QuestionGroupedBatchSampler:
    """Keep all rows from the same question in the same batch."""

    def __init__(
        self,
        question_ids,
        *,
        max_batch_responses: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.max_batch_responses = int(max_batch_responses)
        if self.max_batch_responses <= 0:
            raise ValueError("max_batch_responses must be >= 1.")

        groups: dict[str, list[int]] = {}
        order: list[str] = []
        for row_index, raw_question_id in enumerate(question_ids):
            question_id = str(raw_question_id)
            if question_id not in groups:
                groups[question_id] = []
                order.append(question_id)
            groups[question_id].append(int(row_index))

        self.packed_batches: list[list[int]] = []
        current_batch: list[int] = []
        current_size = 0
        largest_group = 0
        for question_id in order:
            group = groups[question_id]
            group_size = len(group)
            largest_group = max(largest_group, group_size)
            if group_size > self.max_batch_responses:
                raise ValueError(
                    "question-grouped batching requires every question group to fit in one batch: "
                    f"largest_group={largest_group}, max_batch_responses={self.max_batch_responses}."
                )
            if current_batch and current_size + group_size > self.max_batch_responses:
                self.packed_batches.append(list(current_batch))
                current_batch = []
                current_size = 0
            current_batch.extend(group)
            current_size += group_size
        if current_batch:
            self.packed_batches.append(list(current_batch))

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        batch_order = list(range(len(self.packed_batches)))
        if self.shuffle and len(batch_order) > 1:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            permutation = torch.randperm(len(batch_order), generator=generator).tolist()
            batch_order = [batch_order[idx] for idx in permutation]
        for batch_index in batch_order:
            yield list(self.packed_batches[batch_index])

    def __len__(self) -> int:
        return len(self.packed_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _infer_checkpoint_vocab_size_from_load_error(exc: RuntimeError) -> int | None:
    message = str(exc)
    patterns = []
    if "model.embed_tokens.weight" in message:
        patterns.append(
            r"model\.embed_tokens\.weight: copying a param with shape torch\.Size\(\[(\d+),\s*\d+\]\) from checkpoint, "
            r"the shape in current model is torch\.Size\(\[(\d+),\s*\d+\]\)"
        )
    if "lm_head.weight" in message:
        patterns.append(
            r"lm_head\.weight: copying a param with shape torch\.Size\(\[(\d+),\s*\d+\]\) from checkpoint, "
            r"the shape in current model is torch\.Size\(\[(\d+),\s*\d+\]\)"
        )
    if "Error(s) in loading state_dict for Embedding" in message:
        patterns.append(
            r"Error\(s\) in loading state_dict for Embedding:\s*size mismatch for weight: copying a param with shape "
            r"torch\.Size\(\[(\d+),\s*\d+\]\) from checkpoint, the shape in current model is torch\.Size\(\[(\d+),\s*\d+\]\)"
        )

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.DOTALL)
        if match is None:
            continue
        checkpoint_vocab = int(match.group(1))
        current_vocab = int(match.group(2))
        if checkpoint_vocab > current_vocab:
            return checkpoint_vocab
    return None


def _build_fusion_token_strings(num_suffix_specials: int) -> list[str]:
    count = max(0, int(num_suffix_specials or 0))
    return [FUSION_TOKEN_TEMPLATE.format(i) for i in range(1, count + 1)]


def _tokenizer_entry_to_str(token: Any) -> str:
    content = getattr(token, "content", None)
    if isinstance(content, str) and content:
        return content
    return str(token)


def _ensure_suffix_special_tokens(tokenizer, num_suffix_specials: int) -> tuple[list[str], list[int]]:
    fusion_tokens = _build_fusion_token_strings(num_suffix_specials)
    if not fusion_tokens:
        return [], []

    vocab = tokenizer.get_vocab()
    existing_additional = list(tokenizer.special_tokens_map_extended.get("additional_special_tokens", []) or [])
    existing_token_strings = {_tokenizer_entry_to_str(token) for token in existing_additional}
    updated_additional = list(existing_additional)
    added = False
    for token in fusion_tokens:
        if token not in vocab and token not in existing_token_strings:
            updated_additional.append(token)
            added = True
    if added:
        tokenizer.add_special_tokens({"additional_special_tokens": updated_additional})

    token_ids: list[int] = []
    for token in fusion_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            raise ValueError(f"Failed to register suffix fusion token: {token}")
        token_ids.append(int(token_id))
    return fusion_tokens, token_ids


def _feature_value_to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.view(-1).tolist()]
    return [int(v) for v in list(value)]


def _extract_assistant_turn_starts(feature_dict: dict[str, Any]) -> list[int]:
    turn_starts_raw = feature_dict.pop(ASSISTANT_HEADER_STARTS_KEY, None)
    split_start_raw = feature_dict.pop(ASSISTANT_HEADER_START_KEY, None)

    turn_starts: list[int] = []
    if turn_starts_raw is not None:
        for raw_start in _feature_value_to_int_list(turn_starts_raw):
            if raw_start >= 0:
                turn_starts.append(int(raw_start))
    if not turn_starts and split_start_raw is not None:
        if isinstance(split_start_raw, torch.Tensor):
            split_start = int(split_start_raw.item())
        else:
            split_start = int(split_start_raw)
        if split_start >= 0:
            turn_starts.append(split_start)
    return turn_starts


def _insert_vanilla_suffix_specials_into_feature(
    feature: dict[str, Any],
    *,
    suffix_token_ids: list[int],
) -> dict[str, Any]:
    feature_dict = dict(feature)
    turn_starts = _extract_assistant_turn_starts(feature_dict)
    if not turn_starts:
        raise ValueError("assistant_header_start(s) must be present for vanilla suffix-special batches.")

    input_ids = _feature_value_to_int_list(feature_dict["input_ids"])
    labels = _feature_value_to_int_list(feature_dict["labels"])
    attention_mask_raw = feature_dict.get("attention_mask")
    if attention_mask_raw is None:
        attention_mask = [1] * len(input_ids)
    else:
        attention_mask = _feature_value_to_int_list(attention_mask_raw)

    if not (len(input_ids) == len(labels) == len(attention_mask)):
        raise ValueError("input_ids, labels, and attention_mask must have the same length.")

    label_pad = [-100] * len(suffix_token_ids)
    mask_pad = [1] * len(suffix_token_ids)
    offset = 0
    for raw_start in turn_starts:
        insert_at = min(max(int(raw_start) + offset, 0), len(input_ids))
        input_ids[insert_at:insert_at] = suffix_token_ids
        labels[insert_at:insert_at] = label_pad
        attention_mask[insert_at:insert_at] = mask_pad
        offset += len(suffix_token_ids)

    feature_dict["input_ids"] = input_ids
    feature_dict["labels"] = labels
    feature_dict["attention_mask"] = attention_mask
    return feature_dict


def _infer_transformer_layers_pattern(model: torch.nn.Module) -> str:
    for candidate in ("layers", "h", "blocks", "block"):
        needle = f".{candidate}."
        if any(name.startswith(f"{candidate}.") or needle in name for name, _ in model.named_modules()):
            return candidate
    raise ValueError("Unable to infer transformer layer pattern for train_upper_only LoRA placement.")


def _split_lora_target_modules(raw_targets: Any) -> list[str]:
    if raw_targets is None:
        return []
    if isinstance(raw_targets, str):
        values = [raw_targets]
    else:
        values = list(raw_targets)

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for piece in text.replace(",", " ").split():
            token = piece.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


def _infer_all_linear_lora_targets(model: torch.nn.Module) -> list[str]:
    transformer_root = model
    for attr in ("model", "transformer"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, torch.nn.Module):
            transformer_root = candidate
            break

    def _collect(root: torch.nn.Module) -> list[str]:
        skip_leaf_names = {"lm_head", "embed_out", "output", "output_projection"}
        found: list[str] = []
        seen: set[str] = set()
        for name, module in root.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name in skip_leaf_names or leaf_name in seen:
                continue
            seen.add(leaf_name)
            found.append(leaf_name)
        return found

    found = _collect(transformer_root)
    if not found and transformer_root is not model:
        found = _collect(model)
    if not found:
        raise ValueError("Unable to infer LoRA target modules for all_linear.")
    return found


def _resolve_lora_target_modules(model: torch.nn.Module, raw_targets: Any) -> list[str]:
    targets = _split_lora_target_modules(raw_targets)
    if not targets:
        return list(DEFAULT_LORA_TARGET_MODULES)

    normalized = {target.lower().replace("-", "_") for target in targets}
    if normalized & {"all_linear", "alllinear"}:
        if len(targets) != 1:
            raise ValueError("lora_target_modules cannot mix all_linear with explicit module names.")
        return _infer_all_linear_lora_targets(model)
    return targets


def _resolve_transformer_layer_container(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, str, torch.nn.ModuleList, str]:
    candidates: list[tuple[torch.nn.Module, str, torch.nn.ModuleList, str]] = []
    for root_name in ("model", "transformer", ""):
        root = model if root_name == "" else getattr(model, root_name, None)
        if not isinstance(root, torch.nn.Module):
            continue
        for layer_attr in ("layers", "h", "blocks", "block"):
            layer_container = getattr(root, layer_attr, None)
            if isinstance(layer_container, torch.nn.ModuleList):
                qualified_name = f"{root_name}.{layer_attr}" if root_name else layer_attr
                candidates.append((root, layer_attr, layer_container, qualified_name))
    if not candidates:
        raise ValueError("Unable to locate transformer layer container for no_upper_layers pruning.")
    return candidates[0]


def _prune_upper_layers_inplace(model: transformers.PreTrainedModel, keep_layers: int) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("no_upper_layers pruning requires model.config to be present.")
    original_num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if original_num_hidden_layers is None:
        raise ValueError("no_upper_layers pruning requires config.num_hidden_layers to be defined.")

    original_num_hidden_layers = int(original_num_hidden_layers)
    keep_layers = int(keep_layers)
    if keep_layers <= 0:
        raise ValueError(f"no_upper_layers requires keep_layers > 0, got {keep_layers}.")
    if keep_layers > original_num_hidden_layers:
        raise ValueError(
            f"no_upper_layers requested keep_layers={keep_layers}, but model only has "
            f"{original_num_hidden_layers} layers."
        )

    root_module, layer_attr, layer_container, qualified_name = _resolve_transformer_layer_container(model)
    if len(layer_container) < keep_layers:
        raise ValueError(
            f"Layer container '{qualified_name}' has only {len(layer_container)} entries, "
            f"cannot keep {keep_layers} layers."
        )

    if keep_layers < len(layer_container):
        setattr(root_module, layer_attr, torch.nn.ModuleList(list(layer_container[:keep_layers])))
    config.num_hidden_layers = keep_layers
    setattr(config, "capsule_no_upper_layers", True)
    setattr(config, "capsule_original_num_hidden_layers", original_num_hidden_layers)
    setattr(config, "capsule_retained_num_hidden_layers", keep_layers)

    base_model = getattr(model, "model", None)
    if isinstance(base_model, torch.nn.Module) and getattr(base_model, "config", None) is not None:
        base_model.config.num_hidden_layers = keep_layers
        llopa_specials = getattr(base_model, "llopa_specials", None)
        if isinstance(llopa_specials, torch.nn.ParameterList) and len(llopa_specials) > keep_layers:
            base_model.llopa_specials = torch.nn.ParameterList(list(llopa_specials[:keep_layers]))

    if keep_layers < original_num_hidden_layers:
        logger.info(
            "Enabled no_upper_layers: pruned model from %s to %s transformer layers via '%s'.",
            original_num_hidden_layers,
            keep_layers,
            qualified_name,
        )
    else:
        logger.info(
            "Enabled no_upper_layers, but keep_layers=%s matches the full model depth; topology unchanged.",
            keep_layers,
        )


def _unwrap_model_for_metadata(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    if hasattr(current, "module"):
        current = current.module
    return current


def _iter_model_configs(model: torch.nn.Module):
    seen: set[int] = set()
    queue = [_unwrap_model_for_metadata(model)]
    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        config = getattr(current, "config", None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            yield config
        for attr in ("model", "base_model", "language_model"):
            child = getattr(current, attr, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            with contextlib.suppress(Exception):
                base = get_base_model()
                if isinstance(base, torch.nn.Module):
                    queue.append(base)


def _maybe_wait_for_everyone(accelerator: Accelerator, *, reason: str) -> None:
    if getattr(accelerator, "num_processes", 1) <= 1:
        logger.info(
            "Skipping accelerator.wait_for_everyone() at %s because num_processes=%s",
            reason,
            getattr(accelerator, "num_processes", 1),
        )
        return
    accelerator.wait_for_everyone()


def _set_capsule_runtime_metadata(model: torch.nn.Module, args) -> None:
    for config in _iter_model_configs(model):
        setattr(config, "capsule_attention_gate_mode", str(getattr(args, "attention_gate_mode", "off")))
        if not bool(getattr(args, "unified_llopa", False)):
            continue
        setattr(config, "capsule_llopa_enabled", True)
        setattr(config, "capsule_lower_layers", int(args.lower_layers))
        setattr(config, "capsule_prefill_mode", str(args.prefill_mode))
        setattr(config, "capsule_prefill_attn", str(args.prefill_attn))
        setattr(config, "capsule_system_prefill", str(args.system_prefill))
        setattr(config, "capsule_user_prefill", str(args.user_prefill))
        setattr(config, "capsule_no_upper_attn", bool(args.no_upper_attn))
        setattr(config, "capsule_replay_module", str(getattr(args, "replay_module", "none")))
        setattr(config, "capsule_last_layer_module", str(getattr(args, "replay_module", "none")))
        setattr(config, "capsule_replay_per_layers", int(getattr(args, "replay_per_layers", -1) or -1))
        setattr(config, "capsule_num_suffix_specials", int(getattr(args, "num_suffix_specials", 0) or 0))
        setattr(config, "capsule_fusion_mode", _normalize_fusion_mode(getattr(args, "fusion_mode", "upper_only")))
        suffix_tokens = getattr(config, "capsule_suffix_special_tokens", None)
        suffix_token_ids = getattr(config, "capsule_suffix_special_token_ids", None)
        if suffix_tokens is not None:
            setattr(config, "capsule_suffix_special_tokens", list(suffix_tokens))
        if suffix_token_ids is not None:
            setattr(config, "capsule_suffix_special_token_ids", list(suffix_token_ids))


def _write_capsule_tri_info(output_dir: str, args) -> None:
    gate_mode = str(getattr(args, "attention_gate_mode", "off"))
    if not output_dir or (
        not bool(getattr(args, "unified_llopa", False))
        and gate_mode == "off"
    ):
        return
    lines = [
        f"attention_gate_mode={gate_mode}",
    ]
    if not bool(getattr(args, "unified_llopa", False)):
        with open(os.path.join(output_dir, "tri_info.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return
    lines.extend([
        f"lower_k={int(args.lower_layers)}",
        f"prefill_mode={str(args.prefill_mode)}",
        f"prefill_attn={str(args.prefill_attn)}",
        f"system_prefill={str(args.system_prefill)}",
        f"user_prefill={str(args.user_prefill)}",
        f"no_upper_attn={int(bool(args.no_upper_attn))}",
        f"replay_module={str(getattr(args, 'replay_module', 'none'))}",
        f"replay_per_layers={int(getattr(args, 'replay_per_layers', -1) or -1)}",
        f"last_layer_module={str(getattr(args, 'replay_module', 'none'))}",
        f"num_suffix_specials={int(getattr(args, 'num_suffix_specials', 0) or 0)}",
        f"fusion_mode={_normalize_fusion_mode(getattr(args, 'fusion_mode', 'upper_only'))}",
        "capsule_llopa_enabled=1",
    ])
    with open(os.path.join(output_dir, "tri_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _should_prepare_capsule_hf_repo(args) -> bool:
    gate_mode = str(getattr(args, "attention_gate_mode", "off"))
    return bool(
        bool(getattr(args, "unified_llopa", False))
        or bool(getattr(args, "llopa", False))
        or int(getattr(args, "prefill_lower_layers", 0) or 0) > 0
        or gate_mode != "off"
    )


def _resolve_capsule_packaging_module():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "Capsule" / "llopa_train.py",
        repo_root / "llopa_train.py",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("capsule_llopa_train_runtime", str(candidate))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def _prepare_capsule_hf_repo(output_dir: str, args) -> None:
    if not output_dir or not _should_prepare_capsule_hf_repo(args):
        return

    module = _resolve_capsule_packaging_module()
    if module is None or not hasattr(module, "_prepare_hf_repo"):
        logger.warning("Capsule HF packaging helper not found; leaving raw training output at %s", output_dir)
        return

    try:
        module._prepare_hf_repo(
            Path(output_dir),
            str(getattr(args, "model_name_or_path", "") or ""),
            str(getattr(args, "modeling_family", "llama") or "llama"),
            str(getattr(args, "llopa_modeling_path", "") or ""),
            getattr(args, "cache_dir", None),
            getattr(args, "model_revision", None),
            getattr(args, "token", None),
            bool(getattr(args, "local_files_only", False)),
        )
        logger.info("Prepared HF-friendly Capsule repo at %s", output_dir)
    except Exception:
        logger.exception("Failed to prepare HF-friendly Capsule repo at %s", output_dir)


def _maybe_suffix_exp_name_for_system_prefill(args) -> None:
    if not bool(getattr(args, "unified_llopa", False)):
        return
    normalized_system_prefill = normalize_system_prefill(str(args.system_prefill))
    args.system_prefill = normalized_system_prefill
    if normalized_system_prefill == "no_system" and not str(args.exp_name).endswith("-no_system"):
        args.exp_name = f"{args.exp_name}-no_system"


def _use_vanilla_suffix_specials(args) -> bool:
    return bool(
        int(getattr(args, "num_suffix_specials", 0) or 0) > 0
        and not bool(getattr(args, "unified_llopa", False))
        and not bool(getattr(args, "llopa", False))
        and int(getattr(args, "prefill_lower_layers", 0) or 0) <= 0
    )


def _normalize_fusion_mode(mode: Any) -> str:
    normalized = str(mode or "upper_only").strip().lower()
    return normalized or "upper_only"


def _normalize_attention_gate_mode(mode: Any) -> str:
    normalized = str(mode or "off").strip().lower()
    aliases = {
        "": "off",
        "none": "off",
        "disabled": "off",
        "disable": "off",
        "false": "off",
        "0": "off",
        "paper": "sdpa_sigmoid",
        "sdpa_gate": "sdpa_sigmoid",
        "sdpa-gate": "sdpa_sigmoid",
        "sigmoid_after_sdpa": "sdpa_sigmoid",
        "sdpa_elementwise_sigmoid": "sdpa_sigmoid",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"off", "sdpa_sigmoid"}:
        raise ValueError("attention_gate_mode must be one of {'off', 'sdpa_sigmoid'}.")
    return normalized


def _normalize_last_layer_module(mode: Any) -> str:
    normalized = str(mode or "none").strip().lower()
    aliases = {
        "": "none",
        "off": "none",
        "disabled": "none",
        "disable": "none",
        "self-attention": "self",
        "self_attention": "self",
        "selfattn": "self",
        "self_attn": "self",
        "cross-attention": "cross",
        "cross_attention": "cross",
        "crossattn": "cross",
        "cross_attn": "cross",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized or "none"


def _normalize_replay_module(mode: Any) -> str:
    return _normalize_last_layer_module(mode)


def _normalize_replay_per_layers(value: Any) -> int:
    try:
        normalized = int(value)
    except Exception as exc:
        raise ValueError("replay_per_layers must be an integer.") from exc
    if normalized == -1 or normalized >= 1:
        return normalized
    raise ValueError("replay_per_layers must be -1 or a positive integer.")


class VanillaSuffixSpecialDataCollator:
    def __init__(self, *, tokenizer, model, suffix_token_ids: list[int]):
        if not suffix_token_ids:
            raise ValueError("VanillaSuffixSpecialDataCollator requires at least one suffix token id.")
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        self.suffix_token_ids = [int(token_id) for token_id in suffix_token_ids]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        stripped_features = [
            _insert_vanilla_suffix_specials_into_feature(feature, suffix_token_ids=self.suffix_token_ids)
            for feature in features
        ]
        return self.base_collator(stripped_features)


class PrefillLowerDataCollator:
    def __init__(
        self,
        *,
        tokenizer,
        model,
        messages_key: str = "messages",
        system_prefill: str = "no_bos_system",
        enable_batched_last_turn: bool = False,
    ):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        self.tokenizer = tokenizer
        self.messages_key = messages_key
        self.system_prefill = normalize_system_prefill(system_prefill)
        self.enable_batched_last_turn = bool(enable_batched_last_turn)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        split_starts = []
        assistant_turn_starts = []
        system_lens = []
        stripped_features = []
        for feature in features:
            feature_dict = dict(feature)
            split_start = feature_dict.pop(ASSISTANT_HEADER_START_KEY, None)
            turn_starts = feature_dict.pop(ASSISTANT_HEADER_STARTS_KEY, None)
            messages = feature_dict.pop(self.messages_key, None)
            if split_start is not None:
                if isinstance(split_start, torch.Tensor):
                    split_start = int(split_start.item())
                else:
                    split_start = int(split_start)
                if split_start < 0:
                    split_start = None
            split_starts.append(split_start)
            if turn_starts is None:
                assistant_turn_starts.append(None)
            else:
                if isinstance(turn_starts, torch.Tensor):
                    turn_starts = turn_starts.tolist()
                turn_starts = [int(v) for v in list(turn_starts) if int(v) >= 0]
                assistant_turn_starts.append(turn_starts)
            system_len = 0
            if messages is not None and self.system_prefill in {"full", "no_system"}:
                raw_input_ids = feature_dict.get("input_ids")
                sequence_len = None
                if isinstance(raw_input_ids, torch.Tensor):
                    sequence_len = int(raw_input_ids.numel())
                elif raw_input_ids is not None:
                    sequence_len = int(len(raw_input_ids))
                system_len = get_prefill_lower_system_len(
                    self.tokenizer,
                    messages,
                    split_start=split_start,
                    sequence_len=sequence_len,
                )
            system_lens.append(system_len)
            stripped_features.append(feature_dict)

        batch = self.base_collator(stripped_features)
        batch[self.messages_key] = [dict(feature).get(self.messages_key) for feature in features]
        if any(split_start is not None for split_start in split_starts):
            if not all(split_start is not None for split_start in split_starts):
                raise ValueError("assistant_header_start must be present for every example in a prefill-lower batch.")
            batch[ASSISTANT_HEADER_START_KEY] = torch.tensor(split_starts, dtype=torch.long)
        if any(turn_starts is not None for turn_starts in assistant_turn_starts):
            if not all(turn_starts is not None for turn_starts in assistant_turn_starts):
                raise ValueError("assistant_header_starts must be present for every example in a prefill-lower batch.")
            max_turns = max((len(turn_starts) for turn_starts in assistant_turn_starts), default=0)
            padded_turn_starts = torch.full((len(assistant_turn_starts), max_turns), -1, dtype=torch.long)
            turn_mask = torch.zeros((len(assistant_turn_starts), max_turns), dtype=torch.bool)
            for row_idx, turn_starts in enumerate(assistant_turn_starts):
                width = len(turn_starts)
                if width <= 0:
                    continue
                padded_turn_starts[row_idx, :width] = torch.tensor(turn_starts, dtype=torch.long)
                turn_mask[row_idx, :width] = True
            batch[ASSISTANT_HEADER_STARTS_KEY] = padded_turn_starts
            batch[ASSISTANT_HEADER_START_MASK_KEY] = turn_mask
        batch[PREFILL_LOWER_SYSTEM_LEN_KEY] = torch.tensor(system_lens, dtype=torch.long)
        if self.enable_batched_last_turn:
            from open_instruct.llopa_adapter import _batch_prefill_last_turn_examples

            batched_inputs = _batch_prefill_last_turn_examples(self.tokenizer, batch[self.messages_key])
            if batched_inputs is not None:
                batch.update(batched_inputs)
        return batch


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    # Note: the suggested ellipses typing above causes errors on python 3.10, so they are omitted.
    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    do_not_randomize_output_dir: bool = False
    """By default the output directory will be randomized"""
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use flash attention in the model training"}
    )
    model_revision: str | None = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: dict | str | None = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: dict | None = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: str | None = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    system_prompt_override: str | None = field(
        default="You are a helpful assistant.",
        metadata={"help": "Replace/prepend every SFT sample's leading system message with this prompt."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW optimizer."})
    logging_steps: int | None = field(
        default=None, metadata={"help": "Log the training loss and learning rate every logging_steps steps."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of lora."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of lora modules."})
    lora_target_modules: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional LoRA target modules. Pass explicit module names or the special alias "
                "'all_linear' (or 'all-linear') to target all transformer linear layers except the LM head."
            )
        },
    )
    train_upper_only: int = field(
        default=0,
        metadata={
            "help": (
                "If > 0 and use_lora=True, freeze the lowest K transformer blocks by applying LoRA only to "
                "upper layers [K, num_hidden_layers)."
            )
        },
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    group_responses_by_question: bool = field(
        default=False,
        metadata={
            "help": (
                "Keep all responses from the same question in the same batch. "
                "When enabled, per_device_train_batch_size is interpreted as a response-row budget."
            )
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False, metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."}
    )
    unified_llopa: bool = field(
        default=False,
        metadata={"help": "Enable the unified direct-compatible LLoPA training path."},
    )
    lower_layers: int = field(
        default=0,
        metadata={"help": "Number of lower layers K used by unified LLoPA."},
    )
    prefill_mode: str = field(
        default="lower",
        metadata={"help": "Unified LLoPA prefill mode. Only 'lower' is currently supported."},
    )
    prefill_attn: str = field(
        default="causal",
        metadata={"help": "Unified LLoPA prefill attention mode: causal or full."},
    )
    system_prefill: str = field(
        default="no_bos_system",
        metadata={"help": "Unified LLoPA system-prefix visibility: full | no_system | no_bos_system."},
    )
    user_prefill: str = field(
        default="full",
        metadata={"help": "Unified LLoPA user prefill mode. Only 'full' is supported in the main path."},
    )
    no_upper_attn: bool = field(
        default=False,
        metadata={"help": "Unified LLoPA decode optimization: skip upper-layer attention."},
    )
    attention_gate_mode: str = field(
        default="off",
        metadata={
            "help": (
                "Attention gate mode for the whole backbone. "
                "'sdpa_sigmoid' enables the paper-style head-specific elementwise sigmoid gate after SDPA "
                "on every layer; 'off' disables it."
            )
        },
    )
    replay_module: str = field(
        default="none",
        metadata={
            "help": (
                "Replay lower-prefill hidden states in upper layers: "
                "none | self | cross."
            )
        },
    )
    replay_per_layers: int = field(
        default=-1,
        metadata={
            "help": (
                "Replay schedule across upper layers: -1 for the last upper layer only, "
                "or positive N for every N upper layers counted from the upper start."
            )
        },
    )
    last_layer_module: str | None = field(
        default=None,
        metadata={"help": "Deprecated alias for --replay_module."},
    )
    num_suffix_specials: int = field(
        default=0,
        metadata={
            "help": (
                "Number of learnable fusion special tokens (<|FUSION1|>...) inserted at the "
                "assistant boundary. Unified LLoPA/prefill-lower insert them in the upper path; "
                "plain vanilla inserts them directly into the token sequence before assistant turns."
            )
        },
    )
    fusion_mode: str = field(
        default="upper_only",
        metadata={
            "help": (
                "Suffix-fusion behavior for unified LLoPA. "
                "'upper_only' inserts fusion specials only in the upper path; "
                "'inband' inserts them into the token stream before lower-layer processing."
            )
        },
    )
    llopa: bool = field(
        default=False,
        metadata={"help": "Enable Capsule LLoPA training path (system/user segmented prefill + assistant decode)."},
    )
    llopa_prefill_layers: int = field(
        default=32, metadata={"help": "Number of lower layers used during LLoPA prefill (K)."}
    )
    llopa_prefill_mode: str = field(
        default="lower", metadata={"help": "LLoPA prefill mode. Currently only 'lower' is supported."}
    )
    llopa_prefill_attn: str = field(
        default="causal", metadata={"help": "LLoPA prefill attention mode: causal or full."}
    )
    llopa_system_prefill: str = field(
        default="no_bos_system",
        metadata={
            "help": (
                "System-prefix visibility policy for LLoPA and prefill_lower upper layers: "
                "full | no_system | no_bos_system."
            )
        },
    )
    llopa_user_prefill: str = field(
        default="full", metadata={"help": "LLoPA user prefill mode: full | no_question."}
    )
    llopa_no_upper_attn: bool = field(
        default=False, metadata={"help": "LLoPA decode optimization: skip upper-layer attention."}
    )
    prefill_lower_layers: int = field(
        default=0,
        metadata={
            "help": "Vanilla-compatible split path: prefill prompt tokens with lower K layers only, then decode the suffix with full layers."
        },
    )
    prefill_lower_attn: str = field(
        default="causal",
        metadata={"help": "Prefill attention mode for prefill_lower_layers: causal or full."},
    )
    no_upper_layers: bool = field(
        default=False,
        metadata={
            "help": "Physically prune all transformer blocks above prefill_lower_layers and train/save only the "
            "retained lower stack plus lm_head."
        },
    )
    skip_upper_attention_layers: int = field(
        default=0,
        metadata={"help": "For full-sequence training, preserve attention in the first K layers and skip attention above them."},
    )
    solo_attention_layers: int = field(
        default=0,
        metadata={"help": "For full-sequence training, preserve standard attention in the first K layers and use self-only attention above them."},
    )
    prefill_lower_freeze: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze baseline: lower K layers are computed normally, then system/user prefix hidden states are "
                "frozen at layer K while BOS and assistant continue through upper layers."
            )
        },
    )
    prefill_lower_solo_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Solo-attention baseline: lower K layers are computed normally, then system/user prefix tokens use "
                "self-only attention in upper layers while BOS and assistant keep normal upper attention."
            )
        },
    )
    prefill_lower_solo_bos_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Solo-BOS-attention baseline: lower K layers are computed normally, then system/user prefix tokens "
                "use BOS-or-self-only attention in upper layers while BOS and assistant keep normal upper attention."
            )
        },
    )
    llopa_loss_scope: str = field(
        default="last_turn", metadata={"help": "LLoPA assistant loss scope: last_turn | all_assistant."}
    )
    use_single_only: bool = field(
        default=False,
        metadata={"help": "Use only single-turn samples (one user-assistant exchange, optional system messages)."},
    )
    llopa_stream_backward: bool = field(
        default=True,
        metadata={
            "help": "When llopa_loss_scope=all_assistant, run per-turn backward with exact loss scaling "
            "to reduce peak memory."
        },
    )
    llopa_profile_memory_steps: int = field(
        default=0,
        metadata={"help": "Profile LLoPA/vanilla CUDA memory for the first N optimizer steps."},
    )
    llopa_modeling_path: str = field(
        default="", metadata={"help": "Path to Capsule TRI modeling file (e.g., tri_llama3_modeling.py)."}
    )
    lopa_modeling_path: str = field(
        default="", metadata={"help": "Deprecated alias for --llopa_modeling_path."}
    )
    modeling_family: str = field(
        default="llama", metadata={"help": "Model family for TRI modeling injection: llama | qwen3 | mistral."}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    final_lr_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Set the final lr value at the end of training to be final_lr_ratio * learning_rate."
            " Only for linear schedulers, currently."
        },
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    report_to: str | list[str] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: str | None = field(
        default=None, metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."}
    )
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    max_train_steps: int | None = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: str | None = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3, metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."}
    )
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW or not."})
    load_balancing_loss: bool = field(
        default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE) or not."}
    )
    load_balancing_weight: float = field(
        default=0.5, metadata={"help": "Weight for load balancing loss if applicable."}
    )
    clean_checkpoints_at_end: bool = field(
        default=True, metadata={"help": "Whether to clean up all previous checkpoints at the end of the run."}
    )

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    sync_each_batch: bool = False
    """Optionaly sync grads every batch when using grad accumulation. Can significantly reduce memory costs."""
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use packing/padding-free collation via TensorDataCollatorWithFlattening"},
    )
    verbose: bool = field(
        default=False, metadata={"help": "Optionally print additional statistics at each reporting period"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or dataset mixer list.")
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")
        if self.final_lr_ratio is not None:
            if self.lr_scheduler_type != "linear":
                raise NotImplementedError("final_lr_ratio only currently implemented for linear schedulers")
            if not (1.0 >= self.final_lr_ratio >= 0.0):
                raise ValueError(f"final_lr_ratio must be between 0 and 1, not {self.final_lr_ratio=}")
        if self.prefill_lower_layers < 0:
            raise ValueError("prefill_lower_layers must be >= 0.")
        if self.prefill_lower_attn not in {"causal", "full"}:
            raise ValueError("prefill_lower_attn must be one of {'causal', 'full'}.")
        if self.lower_layers < 0:
            raise ValueError("lower_layers must be >= 0.")
        if self.num_suffix_specials < 0:
            raise ValueError("num_suffix_specials must be >= 0.")
        if self.num_suffix_specials > 0 and self.use_lora:
            raise ValueError("num_suffix_specials requires full-model training (use_lora=False).")
        if self.num_suffix_specials > 0 and str(self.modeling_family or "llama").strip().lower() != "llama":
            raise NotImplementedError("num_suffix_specials currently supports modeling_family='llama' only.")
        self.attention_gate_mode = _normalize_attention_gate_mode(self.attention_gate_mode)
        self.fusion_mode = _normalize_fusion_mode(self.fusion_mode)
        if self.last_layer_module is not None and _normalize_replay_module(self.replay_module) == "none":
            self.replay_module = self.last_layer_module
        self.replay_module = _normalize_replay_module(self.replay_module)
        if self.replay_module not in {"none", "self", "cross"}:
            raise ValueError("replay_module must be one of {'none', 'self', 'cross'}.")
        self.replay_per_layers = _normalize_replay_per_layers(self.replay_per_layers)
        self.last_layer_module = self.replay_module
        if self.fusion_mode not in {"upper_only", "inband"}:
            raise ValueError("fusion_mode must be one of {'upper_only', 'inband'}.")
        if self.prefill_attn not in {"causal", "full"}:
            raise ValueError("prefill_attn must be one of {'causal', 'full'}.")
        self.llopa_system_prefill = normalize_system_prefill(self.llopa_system_prefill)
        self.system_prefill = normalize_system_prefill(self.system_prefill)
        if self.num_suffix_specials > 0 and self.packing:
            raise ValueError("num_suffix_specials does not support packing.")
        if self.prefill_lower_layers > 0 and self.packing:
            raise ValueError("prefill_lower_layers path does not support packing.")
        if self.prefill_lower_freeze:
            if self.prefill_lower_layers <= 0:
                raise ValueError("prefill_lower_freeze requires prefill_lower_layers > 0.")
            if self.unified_llopa:
                raise ValueError("prefill_lower_freeze cannot be combined with --unified_llopa.")
            if self.llopa:
                raise ValueError("prefill_lower_freeze cannot be combined with --llopa.")
            if self.no_upper_layers:
                raise ValueError("prefill_lower_freeze cannot be combined with --no_upper_layers.")
            if self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0:
                raise ValueError("prefill_lower_freeze cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
            if self.load_balancing_loss:
                raise ValueError("prefill_lower_freeze does not support load_balancing_loss.")
        if self.prefill_lower_solo_attention:
            if self.prefill_lower_layers <= 0:
                raise ValueError("prefill_lower_solo_attention requires prefill_lower_layers > 0.")
            if self.unified_llopa:
                raise ValueError("prefill_lower_solo_attention cannot be combined with --unified_llopa.")
            if self.llopa:
                raise ValueError("prefill_lower_solo_attention cannot be combined with --llopa.")
            if self.no_upper_layers:
                raise ValueError("prefill_lower_solo_attention cannot be combined with --no_upper_layers.")
            if self.prefill_lower_freeze:
                raise ValueError("prefill_lower_solo_attention cannot be combined with --prefill_lower_freeze.")
            if self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0:
                raise ValueError("prefill_lower_solo_attention cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
            if self.load_balancing_loss:
                raise ValueError("prefill_lower_solo_attention does not support load_balancing_loss.")
        if self.prefill_lower_solo_bos_attention:
            if self.prefill_lower_layers <= 0:
                raise ValueError("prefill_lower_solo_bos_attention requires prefill_lower_layers > 0.")
            if self.unified_llopa:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --unified_llopa.")
            if self.llopa:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --llopa.")
            if self.no_upper_layers:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --no_upper_layers.")
            if self.prefill_lower_freeze:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --prefill_lower_freeze.")
            if self.prefill_lower_solo_attention:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --prefill_lower_solo_attention.")
            if self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0:
                raise ValueError("prefill_lower_solo_bos_attention cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
            if self.load_balancing_loss:
                raise ValueError("prefill_lower_solo_bos_attention does not support load_balancing_loss.")
        if self.replay_module != "none":
            if str(self.modeling_family or "llama").strip().lower() != "llama":
                raise NotImplementedError("replay_module currently supports modeling_family='llama' only.")
            if self.no_upper_layers:
                raise ValueError("replay_module cannot be combined with --no_upper_layers.")
            if self.prefill_lower_freeze:
                raise ValueError("replay_module cannot be combined with --prefill_lower_freeze.")
            if self.prefill_lower_solo_attention:
                raise ValueError("replay_module cannot be combined with --prefill_lower_solo_attention.")
            if self.prefill_lower_solo_bos_attention:
                raise ValueError("replay_module cannot be combined with --prefill_lower_solo_bos_attention.")
        if self.unified_llopa:
            if self.lower_layers <= 0:
                raise ValueError("unified_llopa requires lower_layers > 0.")
            if self.prefill_mode != "lower":
                raise NotImplementedError("unified_llopa currently requires prefill_mode='lower'.")
            if self.llopa_loss_scope not in {"last_turn", "all_assistant"}:
                raise ValueError("unified_llopa requires llopa_loss_scope in {'last_turn', 'all_assistant'}.")
            if self.user_prefill != "full":
                raise ValueError("unified_llopa currently supports only user_prefill='full'.")
            if self.packing:
                raise ValueError("unified_llopa does not support packing.")
            if self.load_balancing_loss:
                raise ValueError("unified_llopa does not support load_balancing_loss.")
            if self.llopa:
                raise ValueError("unified_llopa cannot be combined with legacy --llopa.")
            if self.prefill_lower_layers > 0:
                raise ValueError("unified_llopa cannot be combined with legacy --prefill_lower_layers.")
            if self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0:
                raise ValueError("unified_llopa cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
            if self.no_upper_layers:
                raise ValueError("unified_llopa cannot be combined with --no_upper_layers.")
            if self.replay_module != "none" and self.no_upper_attn:
                raise ValueError("replay_module cannot be combined with --no_upper_attn.")
        elif self.fusion_mode == "inband":
            raise ValueError("fusion_mode='inband' is supported with --unified_llopa only.")
        elif self.replay_module != "none" and self.prefill_lower_layers <= 0:
            raise ValueError("replay_module requires --unified_llopa or --prefill_lower_layers > 0.")
        if self.no_upper_layers:
            if self.prefill_lower_layers <= 0:
                raise ValueError("no_upper_layers requires prefill_lower_layers > 0.")
            if self.llopa:
                raise ValueError("no_upper_layers cannot be combined with --llopa.")
            if self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0:
                raise ValueError("no_upper_layers cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
            if self.use_lora:
                raise ValueError("no_upper_layers currently requires full-model training (use_lora=False).")
            if self.train_upper_only > 0:
                raise ValueError("no_upper_layers cannot be combined with --train_upper_only.")
        if self.llopa:
            if self.llopa_prefill_mode != "lower":
                raise ValueError("LLoPA currently requires llopa_prefill_mode='lower'.")
            if self.llopa_prefill_attn not in {"causal", "full"}:
                raise ValueError("LLoPA requires llopa_prefill_attn in {'causal', 'full'}.")
            if self.llopa_loss_scope not in {"last_turn", "all_assistant"}:
                raise ValueError("LLoPA requires llopa_loss_scope in {'last_turn', 'all_assistant'}.")
            if self.packing:
                raise ValueError("LLoPA path does not support packing.")
            if self.load_balancing_loss:
                raise ValueError("LLoPA path does not support load_balancing_loss.")
        if self.llopa and self.prefill_lower_layers > 0:
            raise ValueError("prefill_lower_layers cannot be combined with --llopa.")
        if (self.skip_upper_attention_layers > 0 or self.solo_attention_layers > 0) and self.prefill_lower_layers > 0:
            raise ValueError("prefill_lower_layers cannot be combined with --skip_upper_attention_layers or --solo_attention_layers.")
        if self.llopa_modeling_path and self.lopa_modeling_path:
            if self.llopa_modeling_path != self.lopa_modeling_path:
                raise ValueError(
                    "Both --llopa_modeling_path and deprecated --lopa_modeling_path were provided with different values."
                )
        if not self.llopa_modeling_path and self.lopa_modeling_path:
            self.llopa_modeling_path = self.lopa_modeling_path

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)


def main(args: FlatArguments, tc: TokenizerConfig):
    if args.train_upper_only > 0 and not args.use_lora:
        raise ValueError("train_upper_only currently requires --use_lora True.")
    if args.group_responses_by_question and args.per_device_train_batch_size <= 0:
        raise ValueError("group_responses_by_question requires per_device_train_batch_size >= 1.")
    _maybe_suffix_exp_name_for_system_prefill(args)

    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps, sync_each_batch=args.sync_each_batch
        ),
    )

    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer
    if tc.chat_template_source_name_or_path:
        logger.info(
            "using chat template copied from %s (revision=%s) with tokenizer %s",
            tc.chat_template_source_name_or_path,
            tc.chat_template_source_revision or tc.tokenizer_revision,
            tc.tokenizer_name_or_path,
        )
    fusion_tokens, fusion_token_ids = _ensure_suffix_special_tokens(tokenizer, int(args.num_suffix_specials))
    if fusion_tokens:
        logger.info(
            "Enabled suffix fusion specials | count=%s | tokens=%s",
            len(fusion_tokens),
            ", ".join(fusion_tokens),
        )

    # ------------------------------------------------------------
    # Set up runtime variables

    if not args.do_not_randomize_output_dir:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("using the output directory: %s", args.output_dir)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

    def _safe_wandb_url(tracker):
        run = getattr(tracker, "run", None)
        return getattr(run, "url", None)

    # ------------------------------------------------------------
    # Initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    wandb_tracker = None
    wandb_url = None
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if accelerator.is_main_process and is_beaker_job():
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.exp_name,
                    "entity": args.wandb_entity,
                    "tags": [truncate_wandb_tag(args.exp_name)] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_url = _safe_wandb_url(wandb_tracker)
        if accelerator.is_main_process:
            maybe_update_beaker_description(wandb_url=wandb_url)

    if accelerator.is_main_process:
        pprint([args, tc])

    # Make one log on every process with the configuration for debugging.
    logger_utils.setup_logger()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    _maybe_wait_for_everyone(accelerator, reason="output directory setup")

    if args.dataset_mixer is not None:
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
    # Keep raw messages whenever SFT filtering needs them.
    # This also enforces identical sample filtering between LLoPA and vanilla runs.
    needs_messages_for_sft_filter = "sft_tulu_filter_v1" in args.dataset_transform_fn
    prefill_lower_needs_messages = bool(args.prefill_lower_layers > 0 or args.unified_llopa)
    vanilla_suffix_specials_enabled = _use_vanilla_suffix_specials(args)
    if needs_messages_for_sft_filter or args.use_single_only or args.llopa or prefill_lower_needs_messages:
        if tc.sft_messages_key not in args.dataset_target_columns:
            args.dataset_target_columns = [*args.dataset_target_columns, tc.sft_messages_key]
    prefill_lower_boundary_ready = any(
        fn_name in {"sft_tulu_tokenize_and_truncate_v1", "last_turn_tulu_tokenize_and_truncate_v1"}
        for fn_name in args.dataset_transform_fn
    )
    if (
        prefill_lower_needs_messages
        and prefill_lower_boundary_ready
        and ASSISTANT_HEADER_START_KEY not in args.dataset_target_columns
    ):
        args.dataset_target_columns = [*args.dataset_target_columns, ASSISTANT_HEADER_START_KEY]
    if (
        args.unified_llopa
        and prefill_lower_boundary_ready
        and ASSISTANT_HEADER_STARTS_KEY not in args.dataset_target_columns
    ):
        args.dataset_target_columns = [*args.dataset_target_columns, ASSISTANT_HEADER_STARTS_KEY]
    if vanilla_suffix_specials_enabled and prefill_lower_boundary_ready:
        if ASSISTANT_HEADER_START_KEY not in args.dataset_target_columns:
            args.dataset_target_columns = [*args.dataset_target_columns, ASSISTANT_HEADER_START_KEY]
        if ASSISTANT_HEADER_STARTS_KEY not in args.dataset_target_columns:
            args.dataset_target_columns = [*args.dataset_target_columns, ASSISTANT_HEADER_STARTS_KEY]
    if args.group_responses_by_question and QUESTION_ID_KEY not in args.dataset_target_columns:
        args.dataset_target_columns = [*args.dataset_target_columns, QUESTION_ID_KEY]
    needs_capsule_modeling = bool(
        args.unified_llopa
        or args.llopa
        or args.prefill_lower_layers > 0
        or args.skip_upper_attention_layers > 0
        or args.solo_attention_layers > 0
        or args.llopa_profile_memory_steps > 0
    )
    if needs_capsule_modeling:
        if args.llopa_modeling_path:
            modeling_path = args.llopa_modeling_path
        else:
            repo_root = Path(__file__).resolve().parents[2]
            default_name = {
                "llama": "tri_llama3_modeling.py",
                "qwen3": "tri_qwen3_modeling.py",
                "mistral": "tri_mistral_modeling.py",
            }.get(str(args.modeling_family or "llama").strip().lower(), "tri_llama3_modeling.py")
            modeling_path = str((repo_root / "Capsule" / default_name).resolve())
        install_llopa_modeling(modeling_path=modeling_path, model_family=args.modeling_family)
        if args.unified_llopa:
            logger.info(
                "Unified LLoPA enabled | lower_k=%s | prefill_mode=%s | prefill_attn=%s | system_prefill=%s | no_upper_attn=%s | replay_module=%s | replay_per_layers=%s | fusion_mode=%s | num_suffix_specials=%s | attention_gate_mode=%s | modeling=%s | family=%s",
                args.lower_layers,
                args.prefill_mode,
                args.prefill_attn,
                args.system_prefill,
                args.no_upper_attn,
                args.replay_module,
                args.replay_per_layers,
                _normalize_fusion_mode(args.fusion_mode),
                args.num_suffix_specials,
                args.attention_gate_mode,
                modeling_path,
                args.modeling_family,
            )
        if args.llopa:
            logger.warning("Legacy --llopa path enabled. Prefer --unified_llopa for new training runs.")
            logger.info("LLoPA enabled | modeling=%s | family=%s", modeling_path, args.modeling_family)
        if args.skip_upper_attention_layers > 0:
            logger.info(
                "Full-sequence upper-attention skip enabled | skip_from_layer=%s | modeling=%s | family=%s",
                args.skip_upper_attention_layers,
                modeling_path,
                args.modeling_family,
            )
        if args.solo_attention_layers > 0:
            logger.info(
                "Full-sequence solo-attention enabled | self_only_from_layer=%s | modeling=%s | family=%s",
                args.solo_attention_layers,
                modeling_path,
                args.modeling_family,
            )
        if args.prefill_lower_solo_attention:
            logger.info(
                "Prefill-lower solo-attention baseline enabled | lower_k=%s | modeling=%s | family=%s",
                args.prefill_lower_layers,
                modeling_path,
                args.modeling_family,
            )
        if args.prefill_lower_solo_bos_attention:
            logger.info(
                "Prefill-lower solo-BOS-attention baseline enabled | lower_k=%s | modeling=%s | family=%s",
                args.prefill_lower_layers,
                modeling_path,
                args.modeling_family,
            )
        if args.prefill_lower_layers > 0:
            logger.warning(
                "Legacy --prefill_lower_layers path enabled. Prefer --unified_llopa for direct-compatible training."
            )
            logger.info(
                "Vanilla-compatible prefill-lower path enabled | lower_k=%s | prefill_attn=%s | system_prefill=%s | modeling=%s | family=%s",
                args.prefill_lower_layers,
                args.prefill_lower_attn,
                args.llopa_system_prefill,
                modeling_path,
                args.modeling_family,
            )
        if args.prefill_lower_freeze:
            logger.info(
                "Freeze baseline enabled | lower_k=%s | prefill_attn=%s | system_prefill=%s | modeling=%s | family=%s",
                args.prefill_lower_layers,
                args.prefill_lower_attn,
                args.llopa_system_prefill,
                modeling_path,
                args.modeling_family,
            )
    elif vanilla_suffix_specials_enabled:
        logger.info(
            "Plain vanilla suffix-special path enabled | count=%s | assistant-boundary insertion in collator",
            args.num_suffix_specials,
        )

    question_ids_for_batching: list[str] | None = None
    with accelerator.main_process_first():
        transform_fn_args = []
        for fn_name in args.dataset_transform_fn:
            if fn_name == "sft_tulu_tokenize_and_truncate_v1":
                transform_fn_args.append(
                    {
                        "max_seq_length": args.max_seq_length,
                        "system_prompt_override": args.system_prompt_override,
                    }
                )
            elif fn_name == "last_turn_tulu_tokenize_and_truncate_v1":
                transform_fn_args.append(
                    {
                        "max_seq_length": args.max_seq_length,
                        "system_prompt_override": args.system_prompt_override,
                    }
                )
            elif fn_name == "sft_tulu_filter_v1":
                filter_args = {
                    "use_single_only": bool(args.use_single_only),
                    # Apply the same validity filter in both vanilla and LLoPA runs
                    # so dataset membership remains identical.
                    "llopa_require_valid_assistant": True,
                    "llopa_loss_scope": str(args.llopa_loss_scope),
                }
                transform_fn_args.append(filter_args)
            else:
                transform_fn_args.append({})
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        if vanilla_suffix_specials_enabled and (
            ASSISTANT_HEADER_STARTS_KEY not in train_dataset.column_names
            and ASSISTANT_HEADER_START_KEY not in train_dataset.column_names
        ):
            raise ValueError(
                "Plain vanilla num_suffix_specials requires assistant header boundary metadata in the dataset. "
                "Use a tokenization transform that emits assistant_header_start(s)."
            )
        if args.group_responses_by_question:
            if QUESTION_ID_KEY not in train_dataset.column_names:
                raise ValueError(
                    "group_responses_by_question requires a dataset column named 'question_id'. "
                    "Make sure the dataset preparation preserves it."
                )
            question_ids_for_batching = [str(question_id) for question_id in train_dataset[QUESTION_ID_KEY]]
            train_dataset = train_dataset.remove_columns([QUESTION_ID_KEY])
            logger.info(
                "Question-grouped response batching enabled | response_budget_per_device=%s | questions=%s | rows=%s",
                args.per_device_train_batch_size,
                len(set(question_ids_for_batching)),
                len(question_ids_for_batching),
            )
        else:
            train_dataset = train_dataset.shuffle(seed=args.seed)
        if args.use_single_only and not args.llopa and not prefill_lower_needs_messages and tc.sft_messages_key in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([tc.sft_messages_key])
        if args.llopa or prefill_lower_needs_messages:
            tensor_columns = list(TOKENIZED_SFT_DATASET_KEYS)
            if ASSISTANT_HEADER_START_KEY in train_dataset.column_names:
                tensor_columns.append(ASSISTANT_HEADER_START_KEY)
            if ASSISTANT_HEADER_STARTS_KEY in train_dataset.column_names:
                tensor_columns.append(ASSISTANT_HEADER_STARTS_KEY)
            train_dataset.set_format(type="pt", columns=tensor_columns, output_all_columns=True)
        else:
            train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)

    if args.cache_dataset_only:
        return

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    setattr(config, "capsule_num_suffix_specials", int(args.num_suffix_specials))
    setattr(config, "capsule_suffix_special_tokens", list(fusion_tokens))
    setattr(config, "capsule_suffix_special_token_ids", list(fusion_token_ids))
    setattr(config, "capsule_fusion_mode", _normalize_fusion_mode(args.fusion_mode))
    setattr(config, "capsule_attention_gate_mode", str(args.attention_gate_mode))
    if (
        args.num_suffix_specials > 0
        and args.no_upper_attn
        and _normalize_fusion_mode(args.fusion_mode) == "upper_only"
    ):
        logger.warning(
            "num_suffix_specials=%s is enabled, but no_upper_attn=True disables upper-layer attention, "
            "so fusion specials will not influence assistant tokens.",
            args.num_suffix_specials,
        )

    if args.model_name_or_path:
        def _load_model_with_vocab_retry(loader, **loader_kwargs):
            try:
                return loader(**loader_kwargs)
            except RuntimeError as exc:
                inferred_vocab_size = _infer_checkpoint_vocab_size_from_load_error(exc)
                if inferred_vocab_size is None:
                    raise
                previous_vocab_size = int(getattr(config, "vocab_size", 0) or 0)
                logger.warning(
                    "Retrying model load after checkpoint vocab mismatch: config vocab_size=%s, checkpoint vocab_size=%s",
                    previous_vocab_size,
                    inferred_vocab_size,
                )
                config.vocab_size = inferred_vocab_size
                loader_kwargs["config"] = config
                return loader(**loader_kwargs)

        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = _load_model_with_vocab_retry(
                AutoModelForCausalLM.from_pretrained,
                pretrained_model_name_or_path=args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            )
        elif args.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: PLC0415

            logger.info("Attempting to apply liger-kernel. fused_linear_cross_entropy=True")

            # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
            model = _load_model_with_vocab_retry(
                AutoLigerKernelForCausalLM.from_pretrained,
                pretrained_model_name_or_path=args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                fused_linear_cross_entropy=True,
            )
        else:
            model = _load_model_with_vocab_retry(
                AutoModelForCausalLM.from_pretrained,
                pretrained_model_name_or_path=args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if args.no_upper_layers:
        _prune_upper_layers_inplace(model, keep_layers=int(args.prefill_lower_layers))

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # Under ZeRO-3 parameters may be partitioned, so gather before reading the embedding size.
    embeddings = model.get_input_embeddings()
    embedding_gather_ctx = (
        deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None)
        if accelerator.distributed_type == DistributedType.DEEPSPEED
        else contextlib.nullcontext()
    )
    with embedding_gather_ctx:
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    embedding_gather_ctx = (
        deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None)
        if accelerator.distributed_type == DistributedType.DEEPSPEED
        else contextlib.nullcontext()
    )
    with embedding_gather_ctx:
        embedding_size = embeddings.weight.shape[0]

    if args.use_lora:
        if args.train_upper_only < 0:
            raise ValueError(f"train_upper_only must be >= 0, got {args.train_upper_only}")
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        elif args.gradient_checkpointing:
            # Enable gradient checkpointing for LoRA (non-QLoRA) too
            model.gradient_checkpointing_enable()

        logger.info("Initializing LORA model...")
        target_modules = _resolve_lora_target_modules(model, getattr(args, "lora_target_modules", []))
        logger.info("LoRA target_modules=%s", target_modules)
        peft_config_kwargs = dict(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        if args.train_upper_only > 0:
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
            if num_hidden_layers is None:
                raise ValueError("train_upper_only requires config.num_hidden_layers to be defined.")
            if args.train_upper_only >= num_hidden_layers:
                raise ValueError(
                    f"train_upper_only={args.train_upper_only} leaves no trainable upper layers "
                    f"(num_hidden_layers={num_hidden_layers})."
                )
            layers_pattern = _infer_transformer_layers_pattern(model)
            layers_to_transform = list(range(args.train_upper_only, num_hidden_layers))
            logger.info(
                "Applying LoRA only to upper layers: freezing lower [%s, %s], training upper [%s, %s] via pattern '%s'.",
                0,
                args.train_upper_only - 1,
                args.train_upper_only,
                num_hidden_layers - 1,
                layers_pattern,
            )
            peft_config_kwargs["layers_pattern"] = layers_pattern
            peft_config_kwargs["layers_to_transform"] = layers_to_transform

        peft_config = LoraConfig(**peft_config_kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # DataLoaders creation:
    if args.packing:
        collate_fn = TensorDataCollatorWithFlattening()
    elif args.unified_llopa:
        collate_fn = PrefillLowerDataCollator(
            tokenizer=tokenizer,
            model=model,
            messages_key=tc.sft_messages_key,
            system_prefill=str(args.system_prefill),
        )
    elif args.llopa:
        collate_fn = LLOPADataCollator(
            tokenizer=tokenizer,
            model=model,
            messages_key=tc.sft_messages_key,
            enable_batched_last_turn=bool(args.use_single_only and args.llopa_loss_scope == "last_turn"),
            system_prefill=str(args.llopa_system_prefill),
            user_prefill=str(args.llopa_user_prefill),
        )
    elif args.prefill_lower_layers > 0:
        collate_fn = PrefillLowerDataCollator(
            tokenizer=tokenizer,
            model=model,
            messages_key=tc.sft_messages_key,
            system_prefill=str(args.llopa_system_prefill),
            enable_batched_last_turn=bool(
                args.prefill_lower_solo_attention
                or args.prefill_lower_solo_bos_attention
                or args.prefill_lower_freeze
            ),
        )
    elif _use_vanilla_suffix_specials(args):
        collate_fn = VanillaSuffixSpecialDataCollator(
            tokenizer=tokenizer,
            model=model,
            suffix_token_ids=fusion_token_ids,
        )
    else:
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    accelerator.print("Creating dataloader")
    if args.group_responses_by_question:
        train_batch_sampler = QuestionGroupedBatchSampler(
            question_ids_for_batching or [],
            max_batch_responses=args.per_device_train_batch_size,
            shuffle=True,
            seed=args.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW  # noqa: PLC0415

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )

    num_warmup_steps = int(num_training_steps_for_scheduler * args.warmup_ratio)
    if args.final_lr_ratio is not None and args.lr_scheduler_type == "linear":
        # Correct num_training_steps_for_scheduler to respect final_lr_ratio for a linear scheduler
        num_training_steps_for_scheduler = (
            num_training_steps_for_scheduler - args.final_lr_ratio * num_warmup_steps
        ) / (1 - args.final_lr_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps,
    )
    if args.group_responses_by_question and accelerator.distributed_type == DistributedType.DEEPSPEED:
        micro_batch_size = int(args.per_device_train_batch_size)
        global_batch_size = micro_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None:
            deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = micro_batch_size
            deepspeed_plugin.deepspeed_config["train_batch_size"] = global_batch_size
            logger.info(
                "Question-grouped batching with DeepSpeed: setting train_micro_batch_size_per_gpu=%s, train_batch_size=%s",
                micro_batch_size,
                global_batch_size,
            )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.group_responses_by_question:
        logger.info("  Question-grouped batching = enabled (batch size units = response rows)")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_batch_idx = 0
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_batch_idx = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_batch_idx // len(train_dataloader)
            completed_steps = resume_batch_idx // args.gradient_accumulation_steps
            resume_batch_idx -= starting_epoch * len(train_dataloader)

    else:
        resume_batch_idx = 0

    resume_step = resume_batch_idx // args.gradient_accumulation_steps

    print(f"Starting {starting_epoch=}, {resume_batch_idx=}, {resume_step=}, {completed_steps=}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_pred_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_total_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_pred_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    start_time = time.perf_counter()
    skipped_batches = False
    skipped_oom_batches = 0

    def _is_cuda_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, torch.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    def _iter_llopa_profile_targets(obj):
        seen: set[int] = set()
        stack = [obj]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            yield current
            for attr in ("module", "model", "base_model"):
                with contextlib.suppress(Exception):
                    child = getattr(current, attr)
                    if child is not None:
                        stack.append(child)
            if hasattr(current, "get_base_model"):
                with contextlib.suppress(Exception):
                    child = current.get_base_model()
                    if child is not None:
                        stack.append(child)

    def _set_llopa_profile_state(obj, *, enabled: bool, step: int) -> None:
        for target in _iter_llopa_profile_targets(obj):
            with contextlib.suppress(Exception):
                setattr(target, "_llopa_profile_memory_enabled", bool(enabled))
            with contextlib.suppress(Exception):
                setattr(target, "_llopa_profile_memory_step", int(step))

    def _batch_profile_lengths(batch_data: dict[str, Any]) -> tuple[int, int, int, int]:
        sequence_len = int(batch_data["input_ids"].size(1)) if "input_ids" in batch_data else -1
        system_len = user_len = assistant_len = -1
        if "llopa_system_attention_mask" in batch_data:
            system_len = int(batch_data["llopa_system_attention_mask"].sum(dim=1, dtype=torch.long).max().item())
        if "llopa_user_attention_mask" in batch_data:
            user_len = int(batch_data["llopa_user_attention_mask"].sum(dim=1, dtype=torch.long).max().item())
        if "llopa_assistant_attention_mask" in batch_data:
            assistant_len = int(batch_data["llopa_assistant_attention_mask"].sum(dim=1, dtype=torch.long).max().item())
        return system_len, user_len, assistant_len, sequence_len

    def _log_memory_profile(stage: str, batch_data: dict[str, Any], step: int) -> None:
        if accelerator.device.type != "cuda":
            return
        system_len, user_len, assistant_len, sequence_len = _batch_profile_lengths(batch_data)
        logger.info(
            "[LLOPA_MEM] step=%s rank=%s stage=%s peak_scope=since_reset system_len=%s user_len=%s assistant_len=%s sequence_len=%s "
            "allocated_GiB=%.3f reserved_GiB=%.3f max_allocated_GiB=%.3f max_reserved_GiB=%.3f",
            step,
            accelerator.process_index,
            stage,
            system_len,
            user_len,
            assistant_len,
            sequence_len,
            torch.cuda.memory_allocated(device=accelerator.device) / 2**30,
            torch.cuda.memory_reserved(device=accelerator.device) / 2**30,
            torch.cuda.max_memory_allocated(device=accelerator.device) / 2**30,
            torch.cuda.max_memory_reserved(device=accelerator.device) / 2**30,
        )

    def _reset_memory_profile_peak() -> None:
        if accelerator.device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(device=accelerator.device)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_batch_idx and not skipped_batches:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint.
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch_idx)
            # Only perform this skip once
            skipped_batches = True
        else:
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            pred_tokens_in_batch = (batch["labels"] != -100).sum()
            tokens_including_padding_in_batch = 0
            if "attention_mask" in batch:
                tokens_in_batch = batch["attention_mask"].sum()
                tokens_including_padding_in_batch = batch["attention_mask"].numel()
                total_token_including_padding += tokens_including_padding_in_batch
            elif "position_ids" in batch:
                tokens_in_batch = batch["position_ids"].numel()
                tokens_including_padding_in_batch = tokens_in_batch
                total_token_including_padding += tokens_including_padding_in_batch
            elif "cu_seq_lens_q" in batch:
                tokens_in_batch = batch["cu_seq_lens_q"][-1]
                tokens_including_padding_in_batch = tokens_in_batch
                total_token_including_padding += tokens_including_padding_in_batch
            else:
                raise ValueError(f"Expected attention_mask or position_ids or cu_seq_lens_q in batch, found {batch=}")
            local_total_tokens += tokens_in_batch
            local_total_tokens_this_log_period += tokens_in_batch
            local_pred_tokens += pred_tokens_in_batch
            local_pred_tokens_this_log_period += pred_tokens_in_batch

            loss = None
            loss_already_backwarded = False
            local_oom = torch.zeros(1, dtype=torch.int32, device=accelerator.device)
            using_stream_backward = args.llopa and args.llopa_loss_scope == "all_assistant" and args.llopa_stream_backward
            profile_this_step = bool(args.llopa_profile_memory_steps > 0 and completed_steps < args.llopa_profile_memory_steps)
            _set_llopa_profile_state(
                model,
                enabled=bool(profile_this_step),
                step=completed_steps + 1,
            )
            if profile_this_step and accelerator.device.type == "cuda":
                _reset_memory_profile_peak()
                _log_memory_profile("batch_received", batch, completed_steps + 1)
            try:
                with accelerator.accumulate(model):
                    if profile_this_step:
                        _reset_memory_profile_peak()
                    if args.unified_llopa:
                        outputs = model(
                            **batch,
                            use_cache=False,
                            prefill_lower_layers=int(args.lower_layers),
                            prefill_lower_attn=str(args.prefill_attn),
                            prefill_lower_system_prefill=str(args.system_prefill),
                            prefill_lower_no_upper_attn=bool(args.no_upper_attn),
                            prefill_lower_replay_module=str(args.replay_module),
                            prefill_lower_replay_per_layers=int(args.replay_per_layers),
                        )
                        loss = outputs.loss
                        del outputs
                    elif args.llopa:
                        if using_stream_backward:
                            loss = compute_llopa_batch_loss_streaming_backward(
                                model=model,
                                tokenizer=tokenizer,
                                batch=batch,
                                backward_fn=accelerator.backward,
                                lower_k=int(args.llopa_prefill_layers),
                                prefill_mode=str(args.llopa_prefill_mode),
                                prefill_attn=str(args.llopa_prefill_attn),
                                system_prefill=str(args.llopa_system_prefill),
                                user_prefill=str(args.llopa_user_prefill),
                                no_upper_attn=bool(args.llopa_no_upper_attn),
                                loss_scope=str(args.llopa_loss_scope),
                                messages_key=tc.sft_messages_key,
                            )
                            loss_already_backwarded = True
                        else:
                            loss = compute_llopa_batch_loss(
                                model=model,
                                tokenizer=tokenizer,
                                batch=batch,
                                lower_k=int(args.llopa_prefill_layers),
                                prefill_mode=str(args.llopa_prefill_mode),
                                prefill_attn=str(args.llopa_prefill_attn),
                                system_prefill=str(args.llopa_system_prefill),
                                user_prefill=str(args.llopa_user_prefill),
                                no_upper_attn=bool(args.llopa_no_upper_attn),
                                loss_scope=str(args.llopa_loss_scope),
                                messages_key=tc.sft_messages_key,
                            )
                    elif args.load_balancing_loss:
                        outputs = model(
                            **batch,
                            use_cache=False,
                            output_router_logits=True,
                            prefill_lower_layers=int(args.prefill_lower_layers),
                            prefill_lower_attn=str(args.prefill_lower_attn),
                            prefill_lower_system_prefill=str(args.llopa_system_prefill),
                            prefill_lower_solo_attention=bool(args.prefill_lower_solo_attention),
                            prefill_lower_solo_bos_attention=bool(args.prefill_lower_solo_bos_attention),
                            prefill_lower_replay_module=str(args.replay_module),
                            prefill_lower_replay_per_layers=int(args.replay_per_layers),
                            skip_upper_attention_layers=int(args.skip_upper_attention_layers),
                            solo_attention_layers=int(args.solo_attention_layers),
                        )
                        total_aux_loss += outputs.aux_loss.detach().float()
                        loss = outputs.loss
                        del outputs
                    elif args.prefill_lower_freeze:
                        loss = compute_prefill_lower_freeze_batch_loss(
                            model=model,
                            batch=batch,
                            lower_k=int(args.prefill_lower_layers),
                            prefill_attn=str(args.prefill_lower_attn),
                            system_prefill=str(args.llopa_system_prefill),
                        )
                    elif args.prefill_lower_solo_attention:
                        loss = compute_prefill_lower_solo_batch_loss(
                            model=model,
                            batch=batch,
                            lower_k=int(args.prefill_lower_layers),
                            prefill_attn=str(args.prefill_lower_attn),
                            system_prefill=str(args.llopa_system_prefill),
                        )
                    elif args.prefill_lower_solo_bos_attention:
                        loss = compute_prefill_lower_solo_bos_batch_loss(
                            model=model,
                            batch=batch,
                            lower_k=int(args.prefill_lower_layers),
                            prefill_attn=str(args.prefill_lower_attn),
                            system_prefill=str(args.llopa_system_prefill),
                        )
                    else:
                        # Standard forward pass
                        outputs = model(
                            **batch,
                            use_cache=False,
                            prefill_lower_layers=int(args.prefill_lower_layers),
                            prefill_lower_attn=str(args.prefill_lower_attn),
                            prefill_lower_system_prefill=str(args.llopa_system_prefill),
                            prefill_lower_solo_attention=bool(args.prefill_lower_solo_attention),
                            prefill_lower_solo_bos_attention=bool(args.prefill_lower_solo_bos_attention),
                            prefill_lower_replay_module=str(args.replay_module),
                            prefill_lower_replay_per_layers=int(args.replay_per_layers),
                            skip_upper_attention_layers=int(args.skip_upper_attention_layers),
                            solo_attention_layers=int(args.solo_attention_layers),
                        )
                        loss = outputs.loss
                        del outputs
                    if profile_this_step:
                        _log_memory_profile("after_forward", batch, completed_steps + 1)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                if using_stream_backward:
                    raise RuntimeError(
                        "CUDA OOM occurred during LLoPA streaming-backward path. "
                        "This path cannot safely skip OOM batches in distributed mode."
                    ) from exc
                local_oom.fill_(1)
                if accelerator.is_main_process:
                    logger.warning("CUDA OOM detected. Marking this batch to skip across all ranks.")
                with contextlib.suppress(Exception):
                    optimizer.zero_grad()
                with contextlib.suppress(Exception):
                    model.zero_grad()
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_oom, op=torch.distributed.ReduceOp.MAX)

            if local_oom.item() > 0:
                # Roll back token counters for a skipped batch to keep logging stats meaningful.
                local_total_tokens -= tokens_in_batch
                local_total_tokens_this_log_period -= tokens_in_batch
                local_pred_tokens -= pred_tokens_in_batch
                local_pred_tokens_this_log_period -= pred_tokens_in_batch
                total_token_including_padding -= tokens_including_padding_in_batch
                skipped_oom_batches += 1
                with contextlib.suppress(Exception):
                    optimizer.zero_grad()
                with contextlib.suppress(Exception):
                    model.zero_grad()
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                continue

            if loss is None:
                raise RuntimeError("Loss is None after forward pass without OOM.")

            # Backward is done after OOM synchronization.
            # This prevents other ranks from entering NCCL collectives when one rank already hit OOM in forward.
            if not loss_already_backwarded:
                if profile_this_step:
                    _reset_memory_profile_peak()
                try:
                    accelerator.backward(loss)
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if _is_cuda_oom_error(exc):
                        raise RuntimeError(
                            "CUDA OOM occurred during backward. "
                            "Skipping is not safe once distributed collectives may have started; "
                            "reduce per-device batch/sequence length."
                        ) from exc
                    raise
                if profile_this_step:
                    _log_memory_profile("after_backward", batch, completed_steps + 1)
            elif profile_this_step:
                _log_memory_profile("after_backward", batch, completed_steps + 1)

            # We keep track of the loss at each logged step
            total_loss += loss.detach().float()
            # clip gradient norm. don't do this with deepspeed
            if accelerator.sync_gradients and args.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if profile_this_step:
                _reset_memory_profile_peak()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            if profile_this_step:
                _log_memory_profile("after_optimizer", batch, completed_steps + 1)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    sum_loss = accelerator.gather(total_loss).sum().item()
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_pred_tokens = accelerator.gather(local_pred_tokens).sum().item()
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()
                    total_tokens_this_log_period = accelerator.gather(local_total_tokens_this_log_period).sum().item()
                    local_total_tokens_this_log_period.zero_()
                    accelerator.gather(local_pred_tokens_this_log_period).sum().item()
                    local_pred_tokens_this_log_period.zero_()

                    avg_tokens_per_batch = (
                        total_tokens
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    avg_tokens_per_batch_including_padding = (
                        total_tokens_including_padding
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    avg_pred_tokens_per_batch = (
                        total_pred_tokens
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "total_tokens": total_tokens,
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "total_pred_tokens": total_pred_tokens,
                        "total_tokens_this_log_period": total_tokens_this_log_period,
                        "avg_tokens_per_batch": avg_tokens_per_batch,
                        "avg_tokens_per_batch_including_padding": avg_tokens_per_batch_including_padding,
                        "avg_pred_tokens_per_batch": avg_pred_tokens_per_batch,
                        "per_device_tps": total_tokens
                        / accelerator.num_processes
                        / (time.perf_counter() - start_time),
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.perf_counter() - start_time),
                        "reserved_mem_GiB": torch.cuda.max_memory_reserved(device=torch.cuda.current_device()) / 2**30,
                        "allocated_mem_GiB": torch.cuda.max_memory_allocated(device=torch.cuda.current_device())
                        / 2**30,
                    }

                    # [Loss Reporting]
                    #
                    # It is useful to handle loss-reporting for the "mean" and "sum" loss cases
                    # differently.  Cases:
                    #
                    # 1) "mean" loss: `sum_loss` takes individual losses which were *averaged* over
                    #    the toks in their sequence and sums them over all fwd passes in the logging
                    #    period.  We instead want the avg over these passes. Report avg_loss =
                    #    sum_loss / total_fwd_passes, which is roughly independent of global batch
                    #    size.
                    #
                    # 2) "sum" loss: `sum_loss` takes individual losses which were *summed* over the
                    #    toks in their sequence and sums them over all fwd passes in the logging
                    #    period.  We want the avg over each optimizer step (which scales with the
                    #    global batch size), and the average loss per token and per prediction
                    #    token (which are roughly independent of global batch size).
                    total_fwd_passes = (
                        args.logging_steps * args.gradient_accumulation_steps * accelerator.num_processes
                    )
                    avg_loss = sum_loss / total_fwd_passes
                    metrics_to_log["train_loss"] = avg_loss
                    if args.verbose:
                        sec_per_step = (time.perf_counter() - start_time) / (completed_steps - resume_step)
                        steps_remaining = args.max_train_steps - completed_steps
                        secs_remaining = steps_remaining * sec_per_step
                        accelerator.print(
                            f"Approx. time remaining: {timedelta(seconds=secs_remaining)}. {args.max_train_steps=}, {completed_steps=}, {steps_remaining=}"
                        )

                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"
                        )
                    if args.verbose:
                        accelerator.print(f"{metrics_to_log=}")
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                        if skipped_oom_batches:
                            accelerator.log({"skipped_oom_batches": skipped_oom_batches}, step=completed_steps)
                    maybe_update_beaker_description(
                        current_step=completed_steps,
                        total_steps=args.max_train_steps,
                        start_time=start_time,
                        wandb_url=wandb_url,
                    )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                        f.write("COMPLETED")
                    if accelerator.is_local_main_process:
                        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                    _maybe_wait_for_everyone(accelerator, reason=f"checkpoint step_{completed_steps}")

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            _maybe_wait_for_everyone(accelerator, reason=f"checkpoint epoch_{epoch}")

    _set_capsule_runtime_metadata(model, args)

    if args.output_dir is not None:
        final_zero_checkpoint_dir = get_last_checkpoint_path(args) if bool(getattr(args, "unified_llopa", False)) else None
        save_with_offline_zero3_lora = (
            args.use_lora
            and bool(getattr(args, "unified_llopa", False))
            and accelerator.distributed_type == DistributedType.DEEPSPEED
            and getattr(accelerator, "num_processes", 1) == 1
            and final_zero_checkpoint_dir is not None
        )
        if save_with_offline_zero3_lora:
            if accelerator.is_main_process:
                logger.info(
                    "Saving LoRA adapter via offline ZeRO checkpoint package from %s to %s",
                    final_zero_checkpoint_dir,
                    args.output_dir,
                )
                target_modules = _resolve_lora_target_modules(model, getattr(args, "lora_target_modules", []))
                offline_result = save_lora_adapter_from_zero_checkpoint(
                    final_zero_checkpoint_dir,
                    args.output_dir,
                    tokenizer=tokenizer,
                    base_model_name_or_path=args.model_name_or_path,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=target_modules,
                    model_revision=args.model_revision,
                )
                logger.info(
                    "Finished offline LoRA adapter package at %s | checkpoint=%s | adapter_keys=%s | include_embedding_layers=%s",
                    args.output_dir,
                    offline_result["checkpoint_path"],
                    offline_result["adapter_keys"],
                    offline_result["include_embedding_layers"],
                )
        else:
            save_with_accelerate(
                accelerator,
                model,
                tokenizer,
                args.output_dir,
                args.use_lora,
                chat_template_name=tc.chat_template_name,
                zero3_checkpoint_dir=final_zero_checkpoint_dir,
                prefer_zero3_offline_merge=bool(getattr(args, "unified_llopa", False)),
            )
        if accelerator.is_main_process:
            _write_capsule_tri_info(args.output_dir, args)
            _prepare_capsule_hf_repo(args.output_dir, args)

    # remove all checkpoints to save space
    if args.clean_checkpoints_at_end and accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_url,
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
        )
    if args.push_to_hub and accelerator.is_main_process:
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    _maybe_wait_for_everyone(accelerator, reason="train end")
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
