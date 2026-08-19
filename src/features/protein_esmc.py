"""ESM-C 600M enzyme encoder (N1 — generational protein LM upgrade).

ESM-C 600M (EvolutionaryScale, 2024) rivals ESM2-3B, a substantial feature-quality jump over the
same-generation ESM2/ProtT5 used in v1. Loaded via the official `esm` SDK. Produces per-residue
(1152-d) and pooled embeddings, cached per unique sequence (same interface as the ESM2/ProtT5
encoders so it drops into the GBDT and cross-attention NN pipelines).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..paths import PRETRAINED_DIR

ESMC_DIM = 1152
ESMC_MAX_RESIDUES = 1022


def _seq_key(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def _truncate(seq: str, max_res: int = ESMC_MAX_RESIDUES) -> str:
    if len(seq) <= max_res:
        return seq
    half = max_res // 2
    return seq[:half] + seq[-half:]


# ESM-C 600M weights (HF-format safetensors). Ships under pretrained/ESMC-600M/ (see
# pretrained/README.md for the download link); override the location with ENZYRANK_PRETRAINED.
ESMC_LOCAL_WEIGHTS = str(PRETRAINED_DIR / "ESMC-600M" / "model.safetensors")


def _remap_hf_to_sdk(hf_key: str):
    """Map an HF-format ESM-C state_dict key to the esm-SDK ESMC key (None to drop)."""
    if hf_key.endswith("_extra_state") or hf_key.startswith("lm_head"):
        return None
    k = hf_key[len("esmc."):] if hf_key.startswith("esmc.") else hf_key
    k = k.replace("attn.layernorm_qkv.layer_norm_weight", "attn.layernorm_qkv.0.weight")
    k = k.replace("attn.layernorm_qkv.layer_norm_bias", "attn.layernorm_qkv.0.bias")
    k = k.replace("attn.layernorm_qkv.weight", "attn.layernorm_qkv.1.weight")
    k = k.replace("ffn.layer_norm_weight", "ffn.0.weight")
    k = k.replace("ffn.layer_norm_bias", "ffn.0.bias")
    k = k.replace("ffn.fc1_weight", "ffn.1.weight")
    k = k.replace("ffn.fc2_weight", "ffn.3.weight")
    return k


class ESMCEncoder:
    """ESM-C 600M encoder: builds the esm-SDK ESMC architecture and loads local HF-format weights
    (avoids the SDK's network download), then extracts pooled/per-residue embeddings with caching."""

    def __init__(self, weights: str = ESMC_LOCAL_WEIGHTS, device: str | None = None):
        import torch

        self.weights = weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from safetensors.torch import load_file
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer

        model = ESMC(d_model=1152, n_heads=18, n_layers=36,
                     tokenizer=EsmSequenceTokenizer()).to(self.device).eval()
        hf = load_file(self.weights)
        sd = {sk: v for hk, v in hf.items() if (sk := _remap_hf_to_sdk(hk)) is not None}
        model.load_state_dict(sd, strict=False)  # only sequence_head (unused) is left uninit
        self._model = model
        torch.set_grad_enabled(False)

    def _embed(self, seq: str) -> np.ndarray:
        """Return (L, 1152) per-residue embeddings (BOS/EOS removed) for one sequence, fp16."""
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig

        self._lazy()
        prot = ESMProtein(sequence=_truncate(seq))
        tns = self._model.encode(prot)
        with torch.no_grad():
            out = self._model.logits(tns, LogitsConfig(sequence=True, return_embeddings=True))
        emb = out.embeddings[0]  # (L+2, 1152) incl. BOS/EOS
        return emb[1:-1].float().cpu().numpy().astype(np.float16)

    def encode_residue_to_h5(self, sequences: list[str], h5_path: Path | str,
                             batch_tokens: int = 12288, log_every: int = 2000) -> None:
        """Per-residue embeddings for unique sequences -> h5 (keyed by md5, fp16, batched). Resumable."""
        import h5py

        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "a") as h:
            todo = [(s, _seq_key(s)) for s in dict.fromkeys(sequences)]
            todo = [(s, k) for s, k in todo if k not in h]
            todo.sort(key=lambda x: len(x[0]))
            i = done = 0
            while i < len(todo):
                batch, maxlen = [], 0
                while i < len(todo):
                    L = min(len(todo[i][0]), ESMC_MAX_RESIDUES) + 2
                    if batch and (len(batch) + 1) * max(maxlen, L) > batch_tokens:
                        break
                    batch.append(todo[i]); maxlen = max(maxlen, L); i += 1
                embs = self._embed_batch([s for s, _ in batch])
                for (s, k), e in zip(batch, embs):
                    h.create_dataset(k, data=e)
                done += len(batch)
                if done % log_every < len(batch):
                    print(f"  [esmc-residue] {done}/{len(todo)}", flush=True)

    def _lazy_tok(self):
        if getattr(self, "_tok", None) is None:
            from esm.tokenization import EsmSequenceTokenizer
            self._tok = EsmSequenceTokenizer()
        return self._tok

    def _embed_batch(self, seqs: list[str]):
        """Return (per_residue list, ) embeddings for a batch via a single forward pass."""
        import torch

        self._lazy()
        tok = self._lazy_tok()
        seqs = [_truncate(s) for s in seqs]
        enc = tok(seqs, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"]
        with torch.no_grad():
            emb = self._model(sequence_tokens=ids).embeddings.float().cpu()  # (B,L,1152)
        outs = []
        for i in range(len(seqs)):
            L = int(am[i].sum())
            outs.append(emb[i, 1:L - 1].numpy().astype(np.float16))  # drop BOS/EOS
        return outs

    def encode_pooled_to_h5(self, sequences: list[str], h5_path: Path | str,
                            batch_tokens: int = 16384, log_every: int = 2000) -> None:
        """mean+max pooled (2304-d) per unique sequence -> small h5 (batched forward)."""
        import h5py

        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "a") as h:
            todo = [(s, _seq_key(s)) for s in dict.fromkeys(sequences)]
            todo = [(s, k) for s, k in todo if k not in h]
            todo.sort(key=lambda x: len(x[0]))
            i = done = 0
            while i < len(todo):
                batch, maxlen = [], 0
                while i < len(todo):
                    L = min(len(todo[i][0]), ESMC_MAX_RESIDUES) + 2
                    if batch and (len(batch) + 1) * max(maxlen, L) > batch_tokens:
                        break
                    batch.append(todo[i]); maxlen = max(maxlen, L); i += 1
                embs = self._embed_batch([s for s, _ in batch])
                for (s, k), e in zip(batch, embs):
                    e = e.astype(np.float32)
                    h.create_dataset(k, data=np.concatenate([e.mean(0), e.max(0)]).astype(np.float16))
                done += len(batch)
                if done % log_every < len(batch):
                    print(f"  [esmc-pooled] {done}/{len(todo)}", flush=True)
