import os
import sys
import torch
import math
from pathlib import Path

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


_WRAPPER = None


class OnnxSafeHTDemucs(torch.nn.Module):

    def __init__(self, inner):
        super().__init__()
        self.inner  = inner
        self.n_fft  = inner.nfft
        self.hop    = inner.hop_length
        self.n_freq = self.n_fft // 2 + 1
        print(f"[i] n_fft={self.n_fft}, hop={self.hop}, n_freq={self.n_freq}")

        br, bi = make_stft_basis(self.n_fft, device='cpu')
        self.register_buffer('basis_real', br)
        self.register_buffer('basis_imag', bi)

    def _stft_real(self, x):
        """
        x: (B, C, T)
        Returns REAL tensor (B, C, F, frames, 2) — last dim is [real, imag].
        NO view_as_complex anywhere.
        """
        B, C, T = x.shape
        pad    = self.n_fft // 2
        x_flat = x.reshape(B * C, 1, T)
        x_pad  = torch.nn.functional.pad(x_flat, (pad, pad), mode='reflect')
        real   = torch.nn.functional.conv1d(x_pad, self.basis_real, stride=self.hop)
        imag   = torch.nn.functional.conv1d(x_pad, self.basis_imag, stride=self.hop)
        frames = real.shape[-1]
        real   = real.reshape(B, C, self.n_freq, frames)
        imag   = imag.reshape(B, C, self.n_freq, frames)
        # Return as (B, C, F, T, 2) real float — no complex type
        return torch.stack([real, imag], dim=-1)

    def _istft_real(self, z_real, length):
        """
        z_real: (..., F, frames, 2) real float
        Returns: (..., length)
        """
        *leading, F, frames, _ = z_real.shape
        real = z_real[..., 0]   # (..., F, frames)
        imag = z_real[..., 1]

        n_fft = self.n_fft
        hop   = self.hop
        win   = torch.hann_window(n_fft, device=z_real.device)

        B_flat = 1
        for d in leading:
            B_flat *= d
        r = real.reshape(B_flat, F, frames)
        i = imag.reshape(B_flat, F, frames)

        t = torch.arange(n_fft, device=z_real.device, dtype=torch.float32)
        k = torch.arange(F,     device=z_real.device, dtype=torch.float32).unsqueeze(1)
        phases   = 2.0 * math.pi * k * t / n_fft
        syn_real = torch.cos(phases)
        syn_imag = -torch.sin(phases)

        frame_sig = (r.permute(0,2,1) @ syn_real - i.permute(0,2,1) @ syn_imag) * win

        T_out  = (frames - 1) * hop + n_fft
        signal = torch.zeros(B_flat, T_out, device=z_real.device)
        norm   = torch.zeros(B_flat, T_out, device=z_real.device)
        win2   = win * win

        for idx in range(frames):
            s = idx * hop
            signal[:, s:s + n_fft] = signal[:, s:s + n_fft] + frame_sig[:, idx, :]
            norm[:,   s:s + n_fft] = norm[:,   s:s + n_fft] + win2

        signal = signal / norm.clamp(min=1e-8)
        pad    = n_fft // 2
        signal = signal[:, pad: pad + length]
        return signal.reshape(*leading, signal.shape[-1])

    def forward(self, mix):
        global _WRAPPER
        _WRAPPER = self

        import demucs.htdemucs as ht
        import demucs.hdemucs  as hd

        orig_spec      = ht.HTDemucs._spec
        orig_magnitude = ht.HTDemucs._magnitude
        orig_mask      = ht.HTDemucs._mask
        orig_ispec     = ht.HTDemucs._ispec

        def patched_spec(self_inner, x):
            # Replicate _spec padding logic, return (B, C, F, T, 2) real
            hl  = self_inner.hop_length
            nfft = self_inner.nfft
            le  = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            x   = hd.pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
            # _stft_real returns (B, C, F, frames, 2)
            z = _WRAPPER._stft_real(x)
            z = z[:, :, :-1, :, :]      # drop last freq bin
            z = z[:, :, :, 2: 2 + le, :]  # trim frames
            return z  # (B, C, F, T, 2) real

        def patched_magnitude(self_inner, z):
            # z: (B, C, F, T, 2) real
            # cac mode: view_as_real then permute to (B, C*2, F, T)
            # We do it directly without any complex ops:
            B, C, F, T, _ = z.shape
            real = z[..., 0]  # (B, C, F, T)
            imag = z[..., 1]
            # Stack to (B, C*2, F, T): first C = real, next C = imag
            return torch.cat([real, imag], dim=1)

        def patched_mask(self_inner, z, m):
            # z: (B, C, F, T, 2) real  — ignored in cac mode
            # m: (B, S, C*2, F, T) — decoder output (real channels)
            # cac mode returns the mask directly as the "complex" output
            # but _ispec needs (B, S, C, F, T, 2)
            B, S, C2, F, T = m.shape
            C = C2 // 2
            real = m[:, :, :C, :, :]   # (B, S, C, F, T)
            imag = m[:, :, C:, :, :]
            return torch.stack([real, imag], dim=-1)  # (B, S, C, F, T, 2)

        def patched_ispec(self_inner, z, length=None, scale=0):
            # z: (B, S, C, F, T, 2) real
            hl  = self_inner.hop_length // (4 ** scale)
            # Replicate _ispec padding
            # F-pad: add one freq bin of zeros
            z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, 1))   # pad F dim
            # T-pad: add 2 on each side
            z = torch.nn.functional.pad(z, (0, 0, 2, 2))          # pad T dim
            pad = hl // 2 * 3
            le  = hl * int(math.ceil(length / hl)) + 2 * pad
            x   = _WRAPPER._istft_real(z, length=le)
            x   = x[..., pad: pad + length]
            return x

        ht.HTDemucs._spec      = patched_spec
        ht.HTDemucs._magnitude = patched_magnitude
        ht.HTDemucs._mask      = patched_mask
        ht.HTDemucs._ispec     = patched_ispec

        try:
            out = self.inner(mix)
        finally:
            ht.HTDemucs._spec      = orig_spec
            ht.HTDemucs._magnitude = orig_magnitude
            ht.HTDemucs._mask      = orig_mask
            ht.HTDemucs._ispec     = orig_ispec
            _WRAPPER = None

        return out


def export_model():
    if not check_deps():
        sys.exit(1)

    print("--- Starting Demucs ONNX Export Process ---")
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    onnx_path = models_dir / "demucs_6s.onnx"

    print("[1/5] Loading htdemucs_6s...")
    from demucs.pretrained import get_model
    try:
        bag   = get_model("htdemucs_6s")
        inner = bag.models[0] if hasattr(bag, 'models') else bag
        inner.eval()
    except Exception as e:
        print(f"[!] Failed to load: {e}")
        import traceback; traceback.print_exc()
        return

    wrapped = OnnxSafeHTDemucs(inner)
    wrapped.eval()

    print("[2/5] Preparing dummy input (343980 samples)...")
    dummy = torch.randn(1, 2, 343980)

    print("[3/5] Tracing...")
    with torch.no_grad():
        try:
            traced = torch.jit.trace(wrapped, dummy, check_trace=False, strict=False)
        except Exception as e:
            print(f"[!] Tracing failed: {e}")
            import traceback; traceback.print_exc()
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
            input_names=['input'],
            output_names=['output'],
            dynamo=False,
            dynamic_axes={
                'input':  {0: 'batch', 2: 'audio_length'},
                'output': {0: 'batch', 2: 'audio_length'},
            },
        )
    except Exception as e:
        print(f"\n[!] Export failed: {e}")
        import traceback; traceback.print_exc()
        return

    print("[5/5] Validating with onnxruntime...")
    import onnxruntime as ort
    try:
        ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        print("[V] Session OK.")
    except Exception as e:
        print(f"[!] Validation failed: {e}")
        import traceback; traceback.print_exc()
        return

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"\n[DONE] {onnx_path}  ({size_mb:.1f} MB)")
    print("[READY] Model exported and validated.")


if __name__ == "__main__":
    export_model()