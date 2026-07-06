"""Clean CAPLA model adapter with checkpoint compatibility.

This module reimplements the active CAPLA architecture while removing side
effects present in the original research code, such as saving attention tensors
or moving tensors to CPU inside `forward`.
"""

from pathlib import Path
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .common import save_torch
from .data_utils import CHAR_SMI_SET_LEN, PT_FEATURE_SIZE

CheckpointLike = Union[str, Path, Mapping[str, Any]]


class Squeeze(nn.Module):
    """Squeeze only the last dimension, preserving the batch axis."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A003 - keep original name for compatibility
        return input.squeeze(-1)


class DilatedConv(nn.Module):
    def __init__(self, nIn: int, nOut: int, kSize: int, stride: int = 1, d: int = 1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class DilatedConvBlockA(nn.Module):
    def __init__(self, nIn: int, nOut: int, add: bool = True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.d16 = DilatedConv(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        self.add = add and (nIn == nOut)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output1 = self.br1(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
        if self.add:
            combine = input + combine
        return self.br2(combine)


class DilatedConvBlockB(nn.Module):
    def __init__(self, nIn: int, nOut: int, add: bool = True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        self.add = add and (nIn == nOut)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output1 = self.br1(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        combine = torch.cat([d1, add1, add2, add3], dim=1)
        if self.add:
            combine = input + combine
        return self.br2(combine)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.gelu(self.layer1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dropout_rate: float, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        q = q * self.scale

        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        if x.size() != orig_q_size:
            raise RuntimeError(f"Unexpected attention output shape {x.size()}, expected {orig_q_size}")
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float, attention_dropout_rate: float, num_heads: int):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class CAPLA(nn.Module):
    """Clean CAPLA model compatible with the original state dict."""

    def __init__(self):
        super().__init__()
        smi_embed_size = 128
        seq_embed_size = 128
        seq_oc = 128
        pkt_oc = 64
        smi_oc = 128
        td_oc = 32

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedConvBlockA(ic, oc))
            ic = oc
        conv_seq.extend([nn.AdaptiveMaxPool1d(1), Squeeze()])
        self.conv_seq = nn.Sequential(*conv_seq)

        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.extend([nn.Conv1d(ic, oc, 3), nn.BatchNorm1d(oc), nn.PReLU()])
            ic = oc
        conv_pkt.extend([nn.AdaptiveMaxPool1d(1), Squeeze()])
        self.conv_pkt = nn.Sequential(*conv_pkt)

        td_conv = []
        ic = 1
        for oc in [16, 32, td_oc * 2]:
            td_conv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        td_conv.extend([nn.AdaptiveMaxPool1d(1), Squeeze()])
        self.td_conv = nn.Sequential(*td_conv)

        td_onlyconv = []
        ic = 1
        for oc in [16, 32, td_oc]:
            td_onlyconv.append(DilatedConvBlockA(ic, oc))
            ic = oc
        self.td_onlyconv = nn.Sequential(*td_onlyconv)

        self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2)
        self.tdpoc_attention_tdlig = EncoderLayer(32, 64, 0.1, 0.1, 1)
        self.adaptmaxpool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = Squeeze()

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedConvBlockB(ic, oc))
            ic = oc
        conv_smi.extend([nn.AdaptiveMaxPool1d(1), Squeeze()])
        self.conv_smi = nn.Sequential(*conv_smi)

        self.cat_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + pkt_oc + smi_oc, 256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, seq: torch.Tensor, pkt: torch.Tensor, smi: torch.Tensor) -> torch.Tensor:
        seq_embed = self.seq_embed(seq)
        seq_embed = torch.transpose(seq_embed, 1, 2)
        seq_conv = self.conv_seq(seq_embed)

        pkt_embed = self.seq_embed(pkt)
        smi_embed = self.smi_embed(smi)
        smi_attention = smi_embed
        smi_embed = self.smi_attention_poc(smi_embed, pkt_embed)
        pkt_embed = self.smi_attention_poc(pkt_embed, smi_attention)

        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)

        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)

        concat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)
        concat = self.cat_dropout(concat)
        return self.classifier(concat)


def build_capla_model() -> CAPLA:
    """Return a fresh clean CAPLA model."""
    return CAPLA()


def _is_tensor_mapping(obj: Any) -> bool:
    if not isinstance(obj, ABCMapping) or not obj:
        return False
    for key, value in obj.items():
        if not isinstance(key, str):
            return False
        if not (torch.is_tensor(value) or isinstance(value, nn.Parameter)):
            return False
    return True


def _to_plain_tensor(value: Any) -> Any:
    if isinstance(value, nn.Parameter):
        return value.detach()
    return value


def _find_state_dict_payload(obj: Any, path: str = "root") -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """Recursively locate the tensor mapping that represents a state dict."""
    if _is_tensor_mapping(obj):
        return obj, path

    if isinstance(obj, ABCMapping):
        preferred_keys = (
            "state_dict",
            "model_state_dict",
            "model",
            "weights",
            "network",
            "net",
            "checkpoint",
        )
        for key in preferred_keys:
            if key in obj:
                payload, payload_path = _find_state_dict_payload(obj[key], f"{path}.{key}")
                if payload is not None:
                    return payload, payload_path

        for key, value in obj.items():
            if isinstance(value, ABCMapping):
                payload, payload_path = _find_state_dict_payload(value, f"{path}.{key}")
                if payload is not None:
                    return payload, payload_path

    return None, None


def _strip_known_prefixes(key: str, prefixes: Tuple[str, ...]) -> str:
    previous = None
    current = key
    while previous != current:
        previous = current
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                break
    return current


def adapt_state_dict_keys(
    state_dict: Mapping[str, Any],
    target_state_dict: Optional[Mapping[str, Any]] = None,
    prefixes: Tuple[str, ...] = ("module.", "model.", "network.", "net.", "capla."),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Adapt minor checkpoint key-prefix differences.

    The shipped CAPLA checkpoints already match 1:1, but this helper keeps the
    loader robust against wrappers such as ``DataParallel``.
    """
    target_keys = set(target_state_dict.keys()) if target_state_dict is not None else None
    cleaned: Dict[str, Any] = {}
    renamed: Dict[str, str] = {}
    collisions = {}  # type: Dict[str, List[str]]

    for original_key, value in state_dict.items():
        tensor_value = _to_plain_tensor(value)
        candidates = [original_key]
        stripped = _strip_known_prefixes(original_key, prefixes)
        if stripped != original_key:
            candidates.append(stripped)

        selected = None
        if target_keys is not None:
            for candidate in candidates:
                if candidate in target_keys:
                    selected = candidate
                    break
        if selected is None:
            selected = candidates[-1]

        if selected in cleaned and selected != original_key:
            collisions.setdefault(selected, []).append(original_key)
            continue

        cleaned[selected] = tensor_value
        if selected != original_key:
            renamed[original_key] = selected

    info = {
        "renamed_keys": renamed,
        "num_renamed": len(renamed),
        "collisions": collisions,
        "num_collisions": sum(len(v) for v in collisions.values()),
        "prefixes_checked": list(prefixes),
    }
    return cleaned, info


def compare_state_dict_compatibility(
    candidate_state_dict: Mapping[str, Any],
    target_state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare keys and tensor shapes between a checkpoint and the clean model."""
    candidate_keys = set(candidate_state_dict.keys())
    target_keys = set(target_state_dict.keys())

    missing_keys = sorted(target_keys - candidate_keys)
    unexpected_keys = sorted(candidate_keys - target_keys)

    shape_mismatches = []
    for key in sorted(candidate_keys & target_keys):
        candidate_shape = tuple(candidate_state_dict[key].shape)
        target_shape = tuple(target_state_dict[key].shape)
        if candidate_shape != target_shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(candidate_shape),
                    "model_shape": list(target_shape),
                }
            )

    return {
        "checkpoint_num_keys": len(candidate_keys),
        "model_num_keys": len(target_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "num_missing_keys": len(missing_keys),
        "num_unexpected_keys": len(unexpected_keys),
        "num_shape_mismatches": len(shape_mismatches),
        "is_compatible_strict": not missing_keys and not unexpected_keys and not shape_mismatches,
    }


def verify_capla_checkpoint_compatibility(
    checkpoint: CheckpointLike,
    model: Optional[CAPLA] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Inspect checkpoint compatibility without mutating a model."""
    if model is None:
        model = build_capla_model()

    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        loaded = torch.load(checkpoint_path, map_location=map_location)
        checkpoint_source = str(checkpoint_path)
    else:
        loaded = dict(checkpoint)
        checkpoint_source = "<in-memory-mapping>"

    if not isinstance(loaded, ABCMapping):
        raise ValueError("CAPLA checkpoint must be a mapping or a path to a serialized mapping.")

    payload, payload_path = _find_state_dict_payload(loaded, "root")
    if payload is None:
        raise ValueError("Could not find a state_dict-like tensor mapping inside the CAPLA checkpoint.")

    target_state_dict = model.state_dict()
    adapted_state_dict, adaptation = adapt_state_dict_keys(payload, target_state_dict=target_state_dict)
    compatibility = compare_state_dict_compatibility(adapted_state_dict, target_state_dict)

    metadata_keys = []
    if isinstance(loaded, ABCMapping) and not _is_tensor_mapping(loaded):
        metadata_keys = sorted([str(key) for key in loaded.keys() if key not in {"state_dict", "model_state_dict"}])

    return {
        "checkpoint_source": checkpoint_source,
        "checkpoint_container_path": payload_path,
        "checkpoint_format": "bundle" if "state_dict" in loaded else "legacy_state_dict",
        "metadata_keys": metadata_keys,
        "key_adaptation": adaptation,
        "compatibility": compatibility,
    }


def _build_load_error(report: Dict[str, Any]) -> str:
    compatibility = report["compatibility"]
    parts = ["CAPLA checkpoint is not compatible with the clean model."]
    if compatibility["num_missing_keys"]:
        parts.append(f"Missing keys: {compatibility['num_missing_keys']}")
    if compatibility["num_unexpected_keys"]:
        parts.append(f"Unexpected keys: {compatibility['num_unexpected_keys']}")
    if compatibility["num_shape_mismatches"]:
        first = compatibility["shape_mismatches"][0]
        parts.append(
            "Shape mismatch example: "
            f"{first['key']} checkpoint={tuple(first['checkpoint_shape'])} "
            f"model={tuple(first['model_shape'])}"
        )
    return " ".join(parts)


def load_capla_checkpoint(
    checkpoint: CheckpointLike,
    model: Optional[CAPLA] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[CAPLA, Dict[str, Any]]:
    """Load either the original ``best_model.pt`` or the new bundle format.

    Compatibility notes
    -------------------
    - Original research checkpoint: plain ``OrderedDict`` state dict.
    - New bundle format: mapping with metadata plus ``state_dict``.
    - Prefix adaptation is supported for common wrappers such as ``module.``.
    """
    if model is None:
        model = build_capla_model()

    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        loaded = torch.load(checkpoint_path, map_location=map_location)
        checkpoint_source = str(checkpoint_path)
    else:
        loaded = dict(checkpoint)
        checkpoint_source = "<in-memory-mapping>"

    if not isinstance(loaded, ABCMapping):
        raise ValueError("CAPLA checkpoint must be a mapping or a path to a serialized mapping.")

    payload, payload_path = _find_state_dict_payload(loaded, "root")
    if payload is None:
        raise ValueError("Could not find a state_dict-like tensor mapping inside the CAPLA checkpoint.")

    target_state_dict = model.state_dict()
    adapted_state_dict, adaptation = adapt_state_dict_keys(payload, target_state_dict=target_state_dict)
    compatibility = compare_state_dict_compatibility(adapted_state_dict, target_state_dict)

    if strict and not compatibility["is_compatible_strict"]:
        raise RuntimeError(_build_load_error({"compatibility": compatibility}))

    incompatible = model.load_state_dict(adapted_state_dict, strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "CAPLA checkpoint load reported incompatible keys after adaptation. "
            f"Missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}."
        )

    metadata: Dict[str, Any] = {}
    if isinstance(loaded, ABCMapping) and not _is_tensor_mapping(loaded):
        metadata = {key: value for key, value in loaded.items() if key not in {"state_dict", "model_state_dict"}}

    metadata["_checkpoint_source"] = checkpoint_source
    metadata["_checkpoint_container_path"] = payload_path
    metadata["_checkpoint_format"] = "bundle" if "state_dict" in loaded else "legacy_state_dict"
    metadata["_key_adaptation"] = adaptation
    metadata["_compatibility_report"] = compatibility
    return model, metadata


def load_capla_model(
    checkpoint: CheckpointLike,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[CAPLA, Dict[str, Any]]:
    """Convenience wrapper that builds the clean model and loads a checkpoint."""
    model = build_capla_model()
    return load_capla_checkpoint(checkpoint=checkpoint, model=model, map_location=map_location, strict=strict)


def save_capla_bundle(
    model: CAPLA,
    path: Union[str, Path],
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> Path:
    """Save the clean CAPLA bundle format."""
    bundle = dict(metadata or {})
    bundle["state_dict"] = model.state_dict()
    bundle.setdefault("checkpoint_format", "CAPLA.pt")
    bundle.setdefault("model_name", "CAPLA")
    bundle.setdefault("repo_variant", "clean_adapter")
    return save_torch(bundle, path)
