import os
import sys
import torch
import math
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F


def check_deps():
    try:
        import demucs
        import onnx
        import onnxruntime
        import demucs.pretrained
        print(f"[V] Demucs version: {demucs.__version__}")
        return True
    except ImportError as e:
        print(f"\n[!] Missing dependency: {e}")
        return False


def make_stft_basis(n_fft, device='cpu'):
    window = torch.hann_window(n_fft, device=device)
    t = torch.arange(n_fft, device=device, dtype=torch.float32)
    k = torch.arange(n_fft // 2 + 1, device=device, dtype=torch.float32).unsqueeze(1)
    phases = 2.0 * math.pi * k * t / n_fft
    basis_real = (torch.cos(phases) * window).unsqueeze(1)
    basis_imag = (-torch.sin(phases) * window).unsqueeze(1)
    return basis_real, basis_imag


# ---------------------------------------------------------------------------
# ONNX-safe replacement for nn.MultiheadAttention
# ---------------------------------------------------------------------------
# The standard nn.MultiheadAttention internally calls _in_projection_packed
# which uses tensor.chunk() followed by operations the ONNX tracer cannot
# resolve (transpose on tensor of unknown rank).  This replacement does the
# Q/K/V projection manually with explicit, fully-shaped linear operations.
# ---------------------------------------------------------------------------
class OnnxSafeMHA(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention that exports to ONNX."""

    def __init__(self, orig: nn.MultiheadAttention):
        super().__init__()
        embed_dim = orig.embed_dim
        num_heads = orig.num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = orig.batch_first

        # Split the packed in_proj_weight [3*E, E] into three [E, E] matrices
        w = orig.in_proj_weight.data          # [3*E, E]
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_proj.weight = nn.Parameter(w[:embed_dim])
        self.k_proj.weight = nn.Parameter(w[embed_dim:2 * embed_dim])
        self.v_proj.weight = nn.Parameter(w[2 * embed_dim:])

        if orig.in_proj_bias is not None:
            b = orig.in_proj_bias.data
            self.q_proj.bias = nn.Parameter(b[:embed_dim])
            self.k_proj.bias = nn.Parameter(b[embed_dim:2 * embed_dim])
            self.v_proj.bias = nn.Parameter(b[2 * embed_dim:])
        else:
            self.q_proj.bias = None
            self.k_proj.bias = None
            self.v_proj.bias = None

        # Output projection
        self.out_proj = orig.out_proj

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, is_causal=False,
                **kwargs):
        # Ensure batch-first: input shape is (B, T, E)
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T_q, E = query.shape
        _, T_k, _ = key.shape
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V  →  (B, T, H, D)  →  (B, H, T, D)
        q = self.q_proj(query).reshape(B, T_q, H, D).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, T_k, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, T_k, H, D).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = float(D) ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale    # (B, H, T_q, T_k)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)          # (B, H, T_q, D)

        # Merge heads → (B, T_q, E)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T_q, E)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Return format matches nn.MultiheadAttention: (output, weights)
        return attn_output, None


def _replace_mha_modules(model):
    """Walk the model and replace every nn.MultiheadAttention with OnnxSafeMHA."""
    replaced = 0
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            setattr(model, name, OnnxSafeMHA(module))
            replaced += 1
        else:
            replaced += _replace_mha_modules(module)
    return replaced


class OnnxSafeHTDemucs(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.n_fft = inner.nfft
        self.hop = inner.hop_length
        self.n_freq = self.n_fft // 2 + 1
        print(f"[i] n_fft={self.n_fft}, hop={self.hop}, n_freq={self.n_freq}")

        br, bi = make_stft_basis(self.n_fft, device='cpu')
        self.register_buffer('basis_real', br)
        self.register_buffer('basis_imag', bi)

        # Replace all MultiheadAttention modules with ONNX-safe versions
        count = _replace_mha_modules(self.inner)
        print(f"[i] Replaced {count} MultiheadAttention module(s) with OnnxSafeMHA")

    def _stft_real(self, x):
        B, C, T = x.shape
        pad = self.n_fft // 2
        x_flat = x.reshape(B * C, 1, T)
        x_pad = F.pad(x_flat, (pad, pad), mode='reflect')
        real = F.conv1d(x_pad, self.basis_real, stride=self.hop)
        imag = F.conv1d(x_pad, self.basis_imag, stride=self.hop)
        frames = real.shape[-1]
        real = real.reshape(B, C, self.n_freq, frames)
        imag = imag.reshape(B, C, self.n_freq, frames)
        return torch.stack([real, imag], dim=-1)

    def _istft_real(self, z_real, length):
        *leading, Fr, frames, _ = z_real.shape
        real = z_real[..., 0]
        imag = z_real[..., 1]

        n_fft = self.n_fft
        hop = self.hop
        win = torch.hann_window(n_fft, device=z_real.device)

        B_flat = 1
        for d in leading:
            B_flat *= d
        r = real.reshape(B_flat, Fr, frames)
        i = imag.reshape(B_flat, Fr, frames)

        t = torch.arange(n_fft, device=z_real.device, dtype=torch.float32)
        k = torch.arange(Fr, device=z_real.device, dtype=torch.float32).unsqueeze(1)
        phases = 2.0 * math.pi * k * t / n_fft
        syn_real = torch.cos(phases)
        syn_imag = -torch.sin(phases)

        frame_sig = (r.permute(0, 2, 1) @ syn_real - i.permute(0, 2, 1) @ syn_imag) * win

        T_out = (frames - 1) * hop + n_fft
        signal = torch.zeros(B_flat, T_out, device=z_real.device)
        norm = torch.zeros(B_flat, T_out, device=z_real.device)
        win2 = win * win

        for idx in range(frames):
            s = idx * hop
            signal[:, s:s + n_fft] = signal[:, s:s + n_fft] + frame_sig[:, idx, :]
            norm[:, s:s + n_fft] = norm[:, s:s + n_fft] + win2

        signal = signal / norm.clamp(min=1e-8)
        pad = n_fft // 2
        signal = signal[:, pad: pad + length]
        return signal.reshape(*leading, signal.shape[-1])

    def forward(self, mix):
        import demucs.htdemucs as ht
        import demucs.hdemucs as hd

        # Save originals
        orig_spec = ht.HTDemucs._spec
        orig_magnitude = ht.HTDemucs._magnitude
        orig_mask = ht.HTDemucs._mask
        orig_ispec = ht.HTDemucs._ispec

        # --- STFT/ISTFT patches (the built-in torch.stft is not ONNX-exportable) ---
        outer = self  # capture reference for closures

        def patched_spec(self_inner, x):
            hl = self_inner.hop_length
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            x = hd.pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode='reflect')
            z = outer._stft_real(x)
            z = z[:, :, :-1, :, :]
            z = z[:, :, :, 2: 2 + le, :]
            return z

        def patched_magnitude(self_inner, z):
            # z: (B, C, F, T, 2)  →  concatenate real & imag along channel dim
            real = z[..., 0]
            imag = z[..., 1]
            return torch.cat([real, imag], dim=1)

        def patched_mask(self_inner, z, m):
            # m: (B, S, C2, F, T) where C2 = 2*C  →  split back into complex
            C = m.shape[2] // 2
            real = m[:, :, :C, :, :]
            imag = m[:, :, C:, :, :]
            return torch.stack([real, imag], dim=-1)

        def patched_ispec(self_inner, z, length=None, scale=0):
            hl = self_inner.hop_length // (4 ** scale)
            z = F.pad(z, (0, 0, 0, 0, 0, 1))
            z = F.pad(z, (0, 0, 2, 2))
            pad = hl // 2 * 3
            le = hl * int(math.ceil(length / hl)) + 2 * pad
            x = outer._istft_real(z, length=le)
            x = x[..., pad: pad + length]
            return x

        ht.HTDemucs._spec = patched_spec
        ht.HTDemucs._magnitude = patched_magnitude
        ht.HTDemucs._mask = patched_mask
        ht.HTDemucs._ispec = patched_ispec

        try:
            out = self.inner(mix)
        finally:
            ht.HTDemucs._spec = orig_spec
            ht.HTDemucs._magnitude = orig_magnitude
            ht.HTDemucs._mask = orig_mask
            ht.HTDemucs._ispec = orig_ispec

        return out


def export_model():
    if not check_deps():
        sys.exit(1)

    print("--- Starting Demucs ONNX Export Process ---")
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / "demucs_6s.onnx"

    print("[1/5] Loading htdemucs_6s...")
    try:
        from demucs.pretrained import get_model
        bag = get_model("htdemucs_6s")
        inner = bag.models[0] if hasattr(bag, 'models') else bag
        inner.eval()
    except Exception as e:
        print(f"[!] Failed to load: {e}")
        return

    wrapped = OnnxSafeHTDemucs(inner)
    wrapped.eval()

    print("[2/5] Preparing dummy input (343980 samples)...")
    dummy = torch.randn(1, 2, 343980)

    print("[3/5] Tracing...")
    try:
        traced = torch.jit.trace(wrapped, dummy, check_trace=False, strict=False)
    except Exception as e:
        print(f"[!] Tracing failed: {e}")
        return

    print("[4/5] Exporting to ONNX (opset 17)...")
    try:
        torch.onnx.export(
            traced,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["mix"],
            output_names=["sources"],
            dynamic_axes={
                "mix": {2: "num_samples"},
                "sources": {3: "num_samples"},
            },
        )
    except Exception as e:
        print(f"[!] ONNX export failed: {e}")
        return

    # Verify
    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"[5/5] Export complete! -> {onnx_path} ({size_mb:.1f} MB)")
        print(f"  Input:  {[i.name for i in model_proto.graph.input]}")
        print(f"  Output: {[o.name for o in model_proto.graph.output]}")
    except Exception as e:
        print(f"[!] Verification failed: {e}")


if __name__ == "__main__":
    export_model()