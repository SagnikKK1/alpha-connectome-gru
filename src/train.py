"""Training loop: single-signal target, weighted-Pearson loss, windowed batches.

Evaluation uses the organisers' own scoring rules (eligibility, mask, clipping)
on a FIXED set of validation sequences, so numbers here are comparable across
runs and to the published baseline.
"""
from __future__ import annotations
import argparse, json, time, sys, os, random
# Deterministic cuBLAS needs a fixed workspace and this must be set before torch
# initialises CUDA -- setting it later silently does nothing. Prize verification
# requires that trained weights can be recreated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))
from utils import weighted_pearson, WARMUP, SEQUENCE_LENGTH
from data import SequenceCache
from metric import wp_loss, wp_loss_two_head, wp_loss_common
import famnorm
import lattice as _lat
from family import BASES
from losses import (RunningWP, SequenceStream, weighted_mse, weighted_dir, weighted_softrank, seq_pooled_wp_loss, seq_pooled_wp_loss_common,
                    clip_ste, hetero_nll, xlane_wp_loss, resnorm_terms,
                    seq_pooled_wp_loss_horizon, sample_grouped,
                    anchored_loss, zmse_loss, seq_target_stats)
from model import build, ExportWrapper


# ---- optimizer family beyond AdamW/SGD (offline/patches/add_optimizers.py) ----------------------------
def _ns5(G, steps=5, eps=1e-7):
    """Newton-Schulz orthogonalisation of the trailing two dims (batched over leading dims): Keller
    Jordan's quintic iteration, coefficients 3.4445 / -4.7750 / 2.0315, Frobenius pre-scaling so the
    spectral norm is <= 1, tall matrices transposed. fp32 (the reference uses bf16)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    tall = G.size(-2) > G.size(-1)
    X = G.mT if tall else G
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


class _MuonAdamW(torch.optim.Optimizer):
    """Muon (orthogonalised Nesterov momentum) for groups with use_muon=True and AdamW (torch.optim.AdamW
    maths: decoupled wd, bias-corrected moments, eps added after the bias correction) for the others, in
    ONE optimizer so the trainer's LambdaLR scales every group. Muon group keys: lr, weight_decay
    (p *= 1 - lr*wd, as the reference), momentum, nesterov, ns_steps, gate_blocks (orthogonalise the
    matrix as that many stacked row blocks: 3 for a GRU's r|z|n gates). Update scale max(1, r/c)**0.5."""
    def __init__(self, groups):
        super().__init__(groups, dict(lr=1e-3, weight_decay=0.0, use_muon=False, momentum=0.95, nesterov=True,
                                      ns_steps=5, gate_blocks=1, betas=(0.9, 0.98), eps=1e-8,
                                      scale_mode="kj", rownorm=0.0, cautious=0, mars=0.0, names=()))
        self.z8_verbose = 3            # add_z8.py: print the per-flag diagnostics line at the first few steps
        self.z8_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        _diag = []                                           # add_z8.py: first-steps diagnostics (stays empty when every flag is off)
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if g["use_muon"]:
                    if "buf" not in st: st["buf"] = torch.zeros_like(p)
                    buf = st["buf"]; mom = g["momentum"]; grad = p.grad
                    _nm = (dict(zip(map(id, g["params"]), g["names"])).get(id(p), "?")
                           if self.z8_steps < self.z8_verbose and (g["mars"] > 0 or g["rownorm"] > 0 or g["cautious"] > 0) else None)
                    if g["mars"] > 0:                        # MARS-approx variance reduction (add_z8.py)
                        if "pg" in st:
                            grad = p.grad + (g["mars"] * mom / (1.0 - mom)) * (p.grad - st["pg"])
                            grad = grad / grad.norm().clamp_min(1.0)
                            st["pg"].copy_(p.grad)
                        else:
                            st["pg"] = p.grad.clone()
                        if _nm: _diag.append(f"{_nm} mars |c|/|g| {(grad.norm() / p.grad.norm().clamp_min(1e-12)).item():.3f}")
                    buf.lerp_(grad, 1.0 - mom)
                    upd = grad.lerp(buf, mom) if g["nesterov"] else buf.clone()
                    nb = int(g["gate_blocks"])
                    if nb > 1: upd = upd.view(nb, p.size(0) // nb, *p.shape[1:])
                    upd = _ns5(upd, int(g["ns_steps"]))
                    if g["rownorm"] > 0:                     # NorMuon: per-row second-moment normalisation (add_z8.py)
                        if "vrow" not in st:
                            st["vrow"] = torch.zeros(*upd.shape[:-1], 1, dtype=p.dtype, device=p.device); st["rn_t"] = 0
                        st["rn_t"] += 1; _b2 = g["rownorm"]
                        _ms = upd.square().mean(-1, keepdim=True)
                        st["vrow"].mul_(_b2).add_(_ms, alpha=1.0 - _b2)
                        _n0 = upd.norm(dim=(-2, -1), keepdim=True)
                        _un = upd / ((st["vrow"] / (1.0 - _b2 ** st["rn_t"])).sqrt() + 1e-8)
                        upd = _un * (_n0 / (_un.norm(dim=(-2, -1), keepdim=True) + 1e-12))
                        if _nm: _diag.append(f"{_nm} row-rms max/min {(_ms.amax() / _ms.amin().clamp_min(1e-30)).sqrt().item():.2f}")
                    if g["scale_mode"] == "rms":             # Moonshot: per-element update RMS 0.2 (add_z8.py)
                        upd = upd * (0.2 * max(upd.size(-2), upd.size(-1)) ** 0.5)
                    else:
                        upd = upd * max(1.0, upd.size(-2) / upd.size(-1)) ** 0.5
                    if g["cautious"] > 0:                    # Cautious: keep sign-consistent entries, rescale (add_z8.py)
                        upd = upd.reshape(p.shape)
                        _mk = (upd * p.grad > 0).to(upd.dtype)
                        upd = upd * _mk * (_mk.numel() / (_mk.sum() + 1.0))
                        if _nm: _diag.append(f"{_nm} cautious keep {_mk.mean().item():.3f}")
                    p.mul_(1.0 - lr * wd)
                    p.add_(upd.reshape(p.shape), alpha=-lr)
                    if g.get("z27_adam", 0.0) > 0:           # add_z27.py: + the AdamW update of the same gradient (research Z27 H2)
                        if "z27_t" not in st:
                            st["z27_t"] = 0; st["z27_m"] = torch.zeros_like(p); st["z27_v"] = torch.zeros_like(p)
                        st["z27_t"] += 1; _t27 = st["z27_t"]; _b1, _b2 = g["z27_betas"]
                        st["z27_m"].lerp_(p.grad, 1.0 - _b1)
                        st["z27_v"].mul_(_b2).addcmul_(p.grad, p.grad, value=1.0 - _b2)
                        _den = (st["z27_v"].sqrt() / (1.0 - _b2 ** _t27) ** 0.5).add_(g["eps"])
                        p.addcdiv_(st["z27_m"], _den, value=-(lr * g["z27_adam"]) / (1.0 - _b1 ** _t27))
                else:
                    if "step" not in st:
                        st["step"] = 0; st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
                    st["step"] += 1; t = st["step"]; b1, b2 = g["betas"]
                    st["m"].lerp_(p.grad, 1.0 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1.0 - b2)
                    p.mul_(1.0 - lr * wd)
                    denom = (st["v"].sqrt() / (1.0 - b2 ** t) ** 0.5).add_(g["eps"])
                    if g["cautious"] >= 2:                   # Cautious masking on the AdamW groups as well (add_z8.py)
                        _u = st["m"] / denom
                        _mk = (_u * p.grad > 0).to(_u.dtype)
                        p.add_(_u * _mk * (_mk.numel() / (_mk.sum() + 1.0)), alpha=-lr / (1.0 - b1 ** t))
                    else:
                        p.addcdiv_(st["m"], denom, value=-lr / (1.0 - b1 ** t))
        self.z8_steps += 1
        if _diag: print(f"z8 step {self.z8_steps}: " + "; ".join(_diag), flush=True)


def _z18_parse_cols(drop, keep, blocks, n=112):
    """--drop-cols / --keep-cols spec (add_z18.py): comma-separated items, each an index `c`, a half-open range `lo:hi`
    (the --seq-affine-cols convention) or a block name of BLOCKS. Returns the sorted list of RAW column indices to drop."""
    if drop and keep: raise SystemExit("--drop-cols and --keep-cols are mutually exclusive")
    cols = set()
    for it in (drop or keep).split(","):
        it = it.strip()
        if not it: continue
        if it in blocks:
            lo, hi = blocks[it]; cols.update(range(lo, hi))
        elif ":" in it:
            lo, hi = it.split(":"); lo, hi = int(lo or 0), int(hi or n)
            if not (0 <= lo < hi <= n): raise SystemExit(f"--drop-cols/--keep-cols: bad range {it!r} (need 0 <= lo < hi <= {n})")
            cols.update(range(lo, hi))
        else:
            try: c = int(it)
            except ValueError: raise SystemExit(f"--drop-cols/--keep-cols: {it!r} is not an index, lo:hi range or block name {sorted(blocks)}")
            if not (0 <= c < n): raise SystemExit(f"--drop-cols/--keep-cols: column {c} out of range 0..{n - 1}")
            cols.add(c)
    if keep: cols = set(range(n)) - cols
    if not cols: raise SystemExit("--drop-cols/--keep-cols: nothing to drop")
    if len(cols) >= n: raise SystemExit("--drop-cols/--keep-cols: would drop every column")
    return sorted(cols)


def _z18_fmt(cols):
    """compact 'lo:hi,c' rendering of a sorted index list (add_z18.py)."""
    out, i = [], 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1: j += 1
        out.append(f"{cols[i]}:{cols[j] + 1}" if j > i else f"{cols[i]}"); i = j + 1
    return ",".join(out)


class _Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023), maths as lucidrains/lion-pytorch: p *= 1 - lr*wd;
    p -= lr * sign(b1*m + (1-b1)*g); m = b2*m + (1-b2)*g."""
    def __init__(self, groups, lr, betas=(0.9, 0.99)):
        super().__init__(groups, dict(lr=lr, betas=betas, weight_decay=0.0))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]; b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "m" not in st: st["m"] = torch.zeros_like(p)
                p.mul_(1.0 - lr * wd)
                p.add_(st["m"].mul(b1).add_(p.grad, alpha=1.0 - b1).sign_(), alpha=-lr)
                st["m"].mul_(b2).add_(p.grad, alpha=1.0 - b2)


class _SoapAdamW(torch.optim.Optimizer):
    """SOAP (Vyas et al. 2024: Shampoo's eigenbases + Adam inside them) for groups with use_soap=True and AdamW
    (torch.optim.AdamW maths: decoupled wd, bias-corrected moments, eps after the bias correction) for the others, in
    ONE optimizer so the trainer's LambdaLR scales every group. SOAP group keys: lr, lr_mult (<= 0: measured at the
    first update so the SOAP step RMS equals AdamW's on the same gradients, clamped to [0.5, 2]), weight_decay
    (p *= 1 - lr*lr_mult*wd), betas, eps, precond_freq (torch.linalg.eigh of L and R every k updates, fp32),
    shampoo_beta (EMA factor of L = GG^T and R = G^TG), gate_blocks (the matrix is that many stacked row blocks with
    their own L/R: 3 for a GRU's r|z|n), names (for the log). The first moment lives in the parameter basis (as the
    reference), the second moment in the eigenbasis and is carried across a basis change as (A o A) v (B o B)^T,
    A = Q_L'^T Q_L, B = Q_R'^T Q_R (exact for uncorrelated components, an exact permutation for reorders and sign
    flips). The first call for a matrix only initialises L, R and the basis (reference: never project a gradient onto
    a basis built from itself). offline/patches/add_soap.py documents every deviation from the reference."""
    def __init__(self, groups):
        super().__init__(groups, dict(lr=1e-3, weight_decay=0.0, use_soap=False, betas=(0.9, 0.98), eps=1e-8,
                                      precond_freq=10, shampoo_beta=0.95, gate_blocks=1, lr_mult=-1.0, names=()))
        self.eigh_fallbacks = 0; self.eigh_calls = 0; self.eigh_seconds = 0.0; self.n_refresh = 0
        self.calibration = None          # (rms_soap, rms_adamw, mult) once measured
        self.verbose_refreshes = 3       # print the direction RMS and eigh time at the first few refreshes

    def _eigvecs(self, M):
        """Eigenvectors of a batch of symmetric PSD matrices (fp32 eigh; fp64 retry if LAPACK/cuSOLVER fails)."""
        t = time.perf_counter(); self.eigh_calls += 1
        try:
            Q = torch.linalg.eigh(M)[1]
        except Exception:
            self.eigh_fallbacks += 1
            Q = torch.linalg.eigh(M.double())[1].to(M.dtype)
        self.eigh_seconds += time.perf_counter() - t
        return Q

    @torch.no_grad()
    def step(self, closure=None):
        pending, calib_soap, calib_adam = [], [], []
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if not g["use_soap"]:
                    if "step" not in st:
                        st["step"] = 0; st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
                    st["step"] += 1; t = st["step"]; b1, b2 = g["betas"]
                    st["m"].lerp_(p.grad, 1.0 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1.0 - b2)
                    p.mul_(1.0 - lr * wd)
                    denom = (st["v"].sqrt() / (1.0 - b2 ** t) ** 0.5).add_(g["eps"])
                    p.addcdiv_(st["m"], denom, value=-lr / (1.0 - b1 ** t))
                    continue
                grad = p.grad
                nb = int(g["gate_blocks"]); r, c = p.size(0) // nb, p.size(1)
                G = grad.reshape(nb, r, c); sb = g["shampoo_beta"]
                if "L" not in st:
                    st["step"] = 0; st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros(nb, r, c, dtype=p.dtype, device=p.device)
                    st["L"] = torch.zeros(nb, r, r, dtype=p.dtype, device=p.device)
                    st["R"] = torch.zeros(nb, c, c, dtype=p.dtype, device=p.device)
                    st["L"].lerp_(G @ G.mT, 1.0 - sb); st["R"].lerp_(G.mT @ G, 1.0 - sb)
                    st["QL"] = self._eigvecs(st["L"]); st["QR"] = self._eigvecs(st["R"])
                    continue                      # reference: the first call only builds the basis, no update
                st["step"] += 1; t = st["step"]; b1, b2 = g["betas"]
                QL, QR = st["QL"], st["QR"]
                Gp = QL.mT @ G @ QR                                      # gradient in the eigenbasis
                st["m"].lerp_(grad, 1.0 - b1)
                st["v"].mul_(b2).addcmul_(Gp, Gp, value=1.0 - b2)
                Mp = QL.mT @ st["m"].reshape(nb, r, c) @ QR              # first moment, rotated
                denom = (st["v"].sqrt() / (1.0 - b2 ** t) ** 0.5).add_(g["eps"])
                D = (Mp / denom) / (1.0 - b1 ** t)                        # Adam direction in the eigenbasis
                upd = (QL @ D @ QR.mT).reshape(p.shape)                  # back to the parameter basis
                pending.append((g, p, upd))
                if g["lr_mult"] <= 0 and t == 1:                         # calibrate against AdamW's first direction
                    calib_soap.append(upd); calib_adam.append(grad / (grad.abs() + g["eps"]))
                # Gram update after the step's direction is formed (reference order); refresh every precond_freq
                st["L"].lerp_(G @ G.mT, 1.0 - sb); st["R"].lerp_(G.mT @ G, 1.0 - sb)
                if t % int(g["precond_freq"]) == 0:
                    QLn, QRn = self._eigvecs(st["L"]), self._eigvecs(st["R"])
                    A, B = QLn.mT @ QL, QRn.mT @ QR
                    st["v"] = (A * A) @ st["v"] @ (B * B).mT
                    st["QL"], st["QR"] = QLn, QRn
                    st["refreshed"] = True
        if calib_soap:
            n = sum(u.numel() for u in calib_soap)
            rs = (sum(u.square().sum() for u in calib_soap) / n).sqrt().item()
            ra = (sum(u.square().sum() for u in calib_adam) / n).sqrt().item()
            mult = min(2.0, max(0.5, ra / max(rs, 1e-12)))
            self.calibration = (rs, ra, mult)
            for g in self.param_groups:
                if g["use_soap"]: g["lr_mult"] = mult
            per = ", ".join(f"{nm} {u.square().mean().sqrt().item():.3f}" for g, p, u in pending for nm in
                            [dict(zip(map(id, g["params"]), g["names"])).get(id(p), "?")] if g["use_soap"])
            print(f"soap: first update: direction RMS soap {rs:.4f} vs adamw {ra:.4f} on the same gradients -> "
                  f"lr mult {mult:.4f} (raw ratio {ra / max(rs, 1e-12):.4f}, clamped to [0.5, 2]); per matrix: {per}",
                  flush=True)
        refreshed = False
        for g, p, upd in pending:
            lr = g["lr"] * g["lr_mult"]
            p.mul_(1.0 - lr * g["weight_decay"])
            p.add_(upd, alpha=-lr)
            st = self.state[p]
            if st.pop("refreshed", False): refreshed = True
        if refreshed:
            self.n_refresh += 1
            if self.n_refresh <= self.verbose_refreshes:
                n = sum(u.numel() for _, _, u in pending)
                rms = (sum(u.square().sum() for _, _, u in pending) / n).sqrt().item()
                print(f"soap: refresh {self.n_refresh} (soap step {self.state[pending[0][1]]['step']}): direction RMS "
                      f"{rms:.4f}, eigh so far {self.eigh_calls} calls {1e3 * self.eigh_seconds:.1f} ms "
                      f"({self.eigh_fallbacks} fp64 fallbacks)", flush=True)


# ---------------------------------------------------------------------------------------------------------
# research Z14 (offline/patches/add_z14.py): factor parametrisation of the GRU matrices + learned input standardisation.
#   wn  W = g[:, None] * V / ||V||_row                 (weight normalisation; g per output unit)
#   sn  W_k = g_k * V_k / sigma_max(V_k) per gate block (spectral normalisation; one power iteration per step, u/v buffers)
# The cuDNN GRU keeps reading its plain weight tensor: _z14_materialise(grad=True) writes W <- f(V, g) before the forward and
# returns the differentiable copies; _z14_chain() pushes dL/dW into V, g after loss.backward() and drops W.grad (so neither
# optimizer steps or decays W); _z14_materialise(grad=False) rewrites W after the step so the EMA copy tracks the plain weight.
# The factors (z14_*) are dropped at save by _z14_export_state(): every checkpoint is a plain refgru state dict.
#   --z14-in-gain / --z14-in-shift: forward pre-hook x -> mu + (x - mu) * exp(logg) + shift * sd, so the model's own
# (x - mu)/sd + clamp sees (x - mu)/sd * exp(logg) + shift; folded into mu/sd at save (sd' = sd/exp(logg), mu' = mu - shift*sd').
def _z14_sigma(V, u, v, nb):
    """Per-gate-block Rayleigh quotient u_k^T V_k v_k -> (nb,); differentiable in V when V requires grad."""
    Vb = V.view(nb, V.shape[0] // nb, V.shape[1])
    return torch.einsum("kr,krc,kc->k", u, Vb, v)


def _z14_power(V, u, nb):
    """One power iteration on each gate block of V (nb, R, C): v <- norm(V^T u), u <- norm(V v). No grad."""
    Vb = V.view(nb, V.shape[0] // nb, V.shape[1])
    v = torch.nn.functional.normalize(torch.einsum("kr,krc->kc", u, Vb), dim=1)
    u = torch.nn.functional.normalize(torch.einsum("krc,kc->kr", Vb, v), dim=1)
    return u, v


def _z14_in_hook(mod, args):
    """Forward pre-hook on the core: learned per-column gain/offset BEFORE the model's own standardisation + clamp."""
    x = args[0]
    lg = getattr(mod, "z14_in_logg", None); sh = getattr(mod, "z14_in_shift", None)
    if lg is not None:
        x = mod.mu + (x - mod.mu) * torch.exp(lg)
    if sh is not None:
        x = x + sh * mod.sd
    return (x,) + tuple(args[1:])


def _z14_setup(core, a, dev):
    """Register the factors / gains on `core` (so the optimizer split, the EMA deepcopy and Lookahead see them). None = off."""
    mode = getattr(a, "z14_reparam", "off"); gain = bool(getattr(a, "z14_in_gain", False)); shift = bool(getattr(a, "z14_in_shift", False))
    if mode == "off" and not gain and not shift:
        return None
    st = {"mode": mode, "mats": []}
    if mode != "off":
        if a.wdrop > 0 or a.wnoise > 0 or a.rec_wd > 0 or a.sam > 0 or a.ncl > 0:
            raise SystemExit("--z14-reparam cannot combine with --wdrop/--wnoise/--rec-wd/--sam/--ncl (they act on the materialised matrix in place)")
        gru = getattr(core, "gru", None)
        if not isinstance(gru, (torch.nn.GRU, torch.nn.LSTM)):
            raise SystemExit("--z14-reparam needs a single nn.GRU/nn.LSTM core (refgru without --split-towers)")
        nb = 4 if isinstance(gru, torch.nn.LSTM) else 3
        want = {"hh": ("weight_hh_l",), "ih": ("weight_ih_l",), "hhih": ("weight_hh_l", "weight_ih_l")}[a.z14_reparam_set]
        core._z14_gate_blocks = {}
        for i, (n, W) in enumerate([(n, q) for n, q in gru.named_parameters(recurse=False) if n.startswith(want)]):
            V = torch.nn.Parameter(W.detach().clone())
            core.register_parameter(f"z14_v_{i}", V)
            core._z14_gate_blocks[id(V)] = nb
            if mode == "wn":
                g = torch.nn.Parameter(W.detach().norm(dim=1))
                uname = None
            else:
                R = W.shape[0] // nb
                u = torch.nn.functional.normalize(torch.ones(nb, R, device=W.device, dtype=W.dtype), dim=1)
                for _ in range(20):
                    u, v = _z14_power(W.detach(), u, nb)
                core.register_buffer(f"z14_pu_{i}", u); core.register_buffer(f"z14_pv_{i}", v)
                g = torch.nn.Parameter(_z14_sigma(W.detach(), u, v, nb))
                uname = f"z14_p{{}}_{i}"
            core.register_parameter(f"z14_g_{i}", g)
            st["mats"].append((f"gru.{n}", W, V, f"z14_g_{i}", uname, nb))
            print(f"z14: reparam {mode} on gru.{n} {tuple(W.shape)} -> factors z14_v_{i} (matrix, {nb} gate blocks) + z14_g_{i} "
                  f"({'per-row norms' if mode == 'wn' else 'per-gate-block spectral norms'}: "
                  + "/".join(f"{float(x):.3f}" for x in (g.detach().min(), g.detach().median(), g.detach().max())) + " min/med/max); "
                  f"W materialised before every forward and after every step, factors dropped at save", flush=True)
    if gain or shift:
        F = core.mu.shape[0]
        if gain:
            core.register_parameter("z14_in_logg", torch.nn.Parameter(torch.zeros(F, device=dev)))
        if shift:
            core.register_parameter("z14_in_shift", torch.nn.Parameter(torch.zeros(F, device=dev)))
        core.register_forward_pre_hook(_z14_in_hook)
        print(f"z14: learned input {'gain' if gain else ''}{'+' if gain and shift else ''}{'shift' if shift else ''} on {F} columns "
              f"before the +-{float(getattr(core, 'clamp', 0.0)):g} clamp (init identity, AdamW no decay); folded into mu/sd at save", flush=True)
    return st


def _z14_materialise(core, st, grad):
    """W <- f(V, g) for every factored matrix. grad=True (before the forward): the differentiable copies are returned for
    _z14_chain(); for `sn` one power iteration refreshes the (u, v) buffers first. grad=False (after the step): plain rewrite with
    the same (u, v), so the post-step and next pre-forward values are the same tensor ops on the same values."""
    pairs = []
    for wname, W, V, gname, uname, nb in st["mats"]:
        g = getattr(core, gname)
        if st["mode"] == "sn" and grad:
            with torch.no_grad():
                u, v = _z14_power(V.detach(), getattr(core, uname.format("u")), nb)
                getattr(core, uname.format("u")).copy_(u); getattr(core, uname.format("v")).copy_(v)
        with torch.set_grad_enabled(grad):
            if st["mode"] == "wn":
                Wm = V * (g / V.norm(dim=1).clamp_min(1e-12)).unsqueeze(1)
            else:
                s = _z14_sigma(V, getattr(core, uname.format("u")), getattr(core, uname.format("v")), nb)
                Wm = (V.view(nb, V.shape[0] // nb, V.shape[1]) * (g / s.clamp_min(1e-12)).view(nb, 1, 1)).reshape(V.shape)
        with torch.no_grad():
            W.copy_(Wm)
        if grad:
            pairs.append((W, Wm))
    return pairs


def _z14_chain(pairs):
    """dL/dV, dL/dg from dL/dW (vector-Jacobian product through the materialisation); W.grad dropped so no optimizer steps W."""
    for W, Wm in pairs:
        if W.grad is not None:
            torch.autograd.backward(Wm, W.grad)
            W.grad = None


def _z59_export_state(st):
    """add_z59.py: today's single-head export format from a Z59 training state. hlg_head.* (train-only) is dropped; a --mh
    readout mh_head.{k}.{0,3}.* is folded into head.0 / head.3: W1 = cat_k W1_k (rows), b1 = cat_k b1_k,
    W2 = cat_k W2_k / K (cols), b2 = mean_k b2_k -- the same function up to fp summation order. Key order is kept."""
    ks = sorted({int(k.split(".")[1]) for k in st if k.startswith("mh_head.")})
    if ks and ks != list(range(len(ks))): raise RuntimeError(f"z59 export: mh_head indices {ks} are not 0..K-1")
    K = len(ks); out = {}
    for k, v in st.items():
        if k.startswith("hlg_head."): continue
        if k.startswith("mh_head."):
            if "head.0.weight" not in out:
                out["head.0.weight"] = torch.cat([st[f"mh_head.{j}.0.weight"] for j in ks], 0)
                out["head.0.bias"] = torch.cat([st[f"mh_head.{j}.0.bias"] for j in ks], 0)
                out["head.3.weight"] = torch.cat([st[f"mh_head.{j}.3.weight"] for j in ks], 1) / K
                out["head.3.bias"] = torch.stack([st[f"mh_head.{j}.3.bias"] for j in ks], 0).mean(0)
            continue
        out[k] = v
    return out


def _z14_export_state(model):
    """Plain state dict for saving: z14_* factors/buffers dropped (W already holds the materialised weight), learned input
    gain/shift folded into the mu/sd buffers (sd' = sd/exp(logg), mu' = mu - shift*sd'). Identity when no z14 tensor exists."""
    sd = model.state_dict()
    out = {k: v.detach().cpu() for k, v in sd.items() if not k.startswith("z14_")}
    lg = sd.get("z14_in_logg"); sh = sd.get("z14_in_shift")
    if lg is not None or sh is not None:
        sd_eff = out["sd"].double() / (torch.exp(lg.detach().double().cpu()) if lg is not None else 1.0)
        mu_eff = out["mu"].double() - ((sh.detach().double().cpu() * sd_eff) if sh is not None else 0.0)
        out["sd"] = sd_eff.float(); out["mu"] = mu_eff.float()
    return out


def _z14_diag(core, st):
    parts = []
    for wname, W, V, gname, uname, nb in st["mats"]:
        g = getattr(core, gname).detach()
        parts.append(f"{wname} g[min/med/max] {float(g.min()):.3f}/{float(g.median()):.3f}/{float(g.max()):.3f} "
                     f"|W|_F {float(W.detach().norm()):.3f} |V|_F {float(V.detach().norm()):.3f}")
    lg = getattr(core, "z14_in_logg", None)
    if lg is not None:
        e = lg.detach().exp(); parts.append(f"in-gain exp(logg)[min/med/max] {float(e.min()):.3f}/{float(e.median()):.3f}/{float(e.max()):.3f}")
    sh = getattr(core, "z14_in_shift", None)
    if sh is not None:
        parts.append(f"in-shift[min/max] {float(sh.min()):.3f}/{float(sh.max()):.3f}")
    return "        z14: " + "; ".join(parts)
# ---------------------------------------------------------------------------------------------------------


def _muon_split(core, gate_split=True, scope="all"):
    """Muon set = hidden 2-D weight matrices. AdamW set = biases, norm gains, 1-D/3-D params, the head's
    output Linear (reference rule: classifier/embedding layers stay on Adam) and any matrix thinner than
    16. Returns (muon [(name, param, gate_blocks)], adamw_decay, adamw_nodecay) in named_parameters order."""
    blocks = {}
    for _, m in core.named_modules():
        nb = 3 if isinstance(m, torch.nn.GRU) else (4 if isinstance(m, torch.nn.LSTM) else 0)
        if nb:
            for q in m.parameters(recurse=False):
                if q.ndim == 2: blocks[id(q)] = nb
    blocks.update(getattr(core, "_z14_gate_blocks", {}))      # add_z14.py: factor matrices orthogonalised per gate block too
    head_out = None
    for q in getattr(core, "head", torch.nn.Identity()).parameters():
        if q.ndim == 2: head_out = q          # nn.Sequential order: the last 2-D weight is the output Linear
    muon, dec, nodec = [], [], []
    for n, q in core.named_parameters():
        mat = q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias")
        if mat and q.ndim == 2 and q is not head_out and min(q.shape) >= 16 and (scope == "all" or (scope == "gru" and id(q) in blocks) or (scope == "nohead" and not n.startswith("head.")) or (scope == "noinproj" and not n.startswith("in_proj.")) or (scope == "noinproj0" and not n.startswith("in_proj.0.")) or (scope == "noinproj3" and not n.startswith("in_proj.3."))):
            muon.append((n, q, blocks.get(id(q), 1) if gate_split else 1))
        elif mat: dec.append(q)
        else: nodec.append(q)
    if scope in ("noinproj0", "noinproj3") and not any(n.startswith(f"in_proj.{scope[-1]}.") for n, _ in core.named_parameters()):
        raise SystemExit(f"--muon-scope {scope}: this model has no in_proj.{scope[-1]} matrix (the scope would be a silent no-op)")
    return muon, dec, nodec
# ---------------------------------------------------------------------------------------------------------


_Z43_VIEWS = {}                                       # --extra-in (add_z43.py): id(cache) -> view with the K channels appended


def _z43_view(cache):
    """identity unless --extra-in registered a view for this cache object (add_z43.py)"""
    return _Z43_VIEWS.get(id(cache), cache)


class _Z43View:
    """--extra-in (add_z43.py): a cache whose X[i] is [X[i], fill(extra(i))] (112 + K columns); Y, M and len unchanged.
    Whole-sequence reads only (evaluate / calibrate_scale)."""

    def __init__(self, cache, extra, fill):
        self.c, self.e, self.fill = cache, extra, float(fill)
        self.Y, self.M = cache.Y, getattr(cache, "M", None)
        self.meta = getattr(cache, "meta", None)
        self.X = self

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            raise TypeError("--extra-in: windowed cache.X[i, a:b] reads are not supported (add_z43.py)")
        x = np.asarray(self.c.X[i], np.float32)
        e = np.asarray(self.e(int(i)), np.float32)
        return np.concatenate([x, np.where(np.isnan(e), np.float32(self.fill), e)], -1)


def _ems_parse(spec: str, steps: int):
    """add_ems.py (research EMS): --ema-extra SPEC -> [{name, kind, d0, d1, s0, r0, spec}] (s0/r0 = optimizer steps)."""
    out, seen = [], set()
    for item in [t.strip() for t in spec.split(",") if t.strip()]:
        f = item.split(":")
        name = f[0]
        if not name.isalnum() or name in seen:
            raise SystemExit(f"--ema-extra: bad or duplicate name {name!r} in {item!r} (letters/digits only)")
        seen.add(name)
        try:
            if len(f) >= 2 and f[1] == "swa":
                if len(f) != 3: raise ValueError
                kind, d0, d1, st = "swa", None, None, float(f[2]); rf = st
            else:
                if not 3 <= len(f) <= 5: raise ValueError
                kind, d0, st = "ema", float(f[1]), float(f[2])
                d1 = float(f[3]) if len(f) >= 4 else d0
                rf = float(f[4]) if len(f) >= 5 else st
        except ValueError:
            raise SystemExit(f"--ema-extra: cannot parse {item!r}: name:decay:start[:end_decay[:ramp_from]] or name:swa:start")
        if not (0.0 <= st < 1.0 and st <= rf < 1.0) or (kind == "ema" and not (0.0 < d0 < 1.0 and 0.0 < d1 < 1.0)):
            raise SystemExit(f"--ema-extra: out of range in {item!r}: 0 <= start <= ramp_from < 1 and 0 < decay < 1")
        out.append({"name": name, "kind": kind, "d0": d0, "d1": d1, "s0": int(round(st * steps)),
                    "r0": int(round(rf * steps)), "spec": item})
    return out


def _ems_decay(e, step: int, steps: int) -> float:
    """add_ems.py: decay used at optimizer step `step` (> e['s0']) by the extra copy e."""
    if e["kind"] == "swa":
        return 1.0 - 1.0 / (step - e["s0"] + 1)            # running mean of the weights after steps s0..step
    if e["d1"] == e["d0"] or step <= e["r0"]:
        return e["d0"]
    return e["d0"] + (e["d1"] - e["d0"]) * min(1.0, (step - e["r0"]) / max(1, steps - e["r0"]))


@torch.no_grad()
def calibrate_scale(core, cache: SequenceCache, n: int = 16, target_std: float = 1.0,
                    device="cpu", two_head=False, common_head=False, keep_cols=None,
                    xform=None, famv=None, fnorm=None, famv_norm=None, lat=None):
    """A single global output scale, fitted on TRAINING sequences.

    The metric clips predictions to [-2, 2], so output scale is NOT free: a model
    emitting std 9.8 is butchered by the clip. Correlation is otherwise
    scale-invariant, so this one scalar is pure calibration. Fitted on train, not
    on the evaluation set, so it leaks nothing.
    """
    cache = _z43_view(cache)                         # add_z43.py: the same object unless --extra-in registered a view
    core.eval(); stds = []
    for a in range(0, min(n, len(cache)), 8):
        b = min(a+8, min(n, len(cache)))
        X = np.stack([cache.X[i] for i in range(a, b)])
        if keep_cols is not None: X = X[..., keep_cols]
        if fnorm is not None: X = famnorm.apply(X, famv_norm[a:b], *fnorm)
        if xform is not None: X = xform(X)
        if famv is not None:
            oh = np.zeros(X.shape[:-1] + (4,), np.float32)
            for k in range(X.shape[0]): oh[k, :, famv[a+k]] = 1.0
            X = np.concatenate([X, oh], axis=-1)
        out, _ = core(torch.from_numpy(X).permute(1, 0, 2).to(device), core.init_state(b-a, device))
        o = out.permute(1, 0, 2).cpu().numpy()
        # for the (s, c) head the emitted prediction is s + c, so calibrate on that
        if lat is not None:
            K, qv = lat
            o = _lat.readout(torch.from_numpy(o), qv[a:b], K).numpy()
            ch = o
        else:
            ch = (o[..., 0] + o[..., 1]) if common_head else o[..., 0]
        stds += [float(ch[k][WARMUP:].std()) for k in range(b-a)]
        del X, out, o
    core.train()
    return float(target_std / max(np.mean(stds), 1e-9))


@torch.no_grad()
def evaluate(core, cache: SequenceCache, n: int | None = None, chunk: int = 8,
             device="cpu", two_head=False, scale: float = 1.0, common_head=False,
             keep_cols=None, xform=None, famv=None, fnorm=None, famv_norm=None, lat=None,
             reset_every: int = 0):
    """Full-sequence stateful inference, scored exactly as the platform would.

    reset_every > 0 zeroes the recurrent state every K rows, matching --state-reset. A model trained
    with resets MUST be evaluated on the same grid: scoring it with a state carried across all 20,000
    rows measures a model that was never trained to exist, and the reverse -- resets applied only at
    inference to a model trained without them -- is monotonically worse (-0.0003/-0.0016/-0.0029 at
    4000/2000/1000 on pooled-800). Grid mismatch in either direction makes the number meaningless.
    """
    cache = _z43_view(cache)                         # add_z43.py: the same object unless --extra-in registered a view
    core.eval()
    n = n or len(cache)
    scores, preds_std, skipped = [], [], 0
    pool = [np.zeros(6), np.zeros(6)]   # pooled sufficient statistics, per target
    steps = np.arange(SEQUENCE_LENGTH)
    for a in range(0, n, chunk):
        b = min(a+chunk, n)
        X = np.stack([cache.X[i] for i in range(a, b)])          # (c,20000,112)
        if keep_cols is not None: X = X[..., keep_cols]
        if fnorm is not None: X = famnorm.apply(X, famv_norm[a:b], *fnorm)
        if xform is not None: X = xform(X)
        if famv is not None:
            oh = np.zeros(X.shape[:-1] + (4,), np.float32)
            for k in range(X.shape[0]): oh[k, :, famv[a+k]] = 1.0
            X = np.concatenate([X, oh], axis=-1)
        Y = np.stack([cache.Y[i] for i in range(a, b)])
        M = np.stack([cache.M[i] for i in range(a, b)])
        xt = torch.from_numpy(X).permute(1, 0, 2).to(device)     # (T,c,F)
        if reset_every and reset_every < SEQUENCE_LENGTH:
            segs = []
            for t0 in range(0, SEQUENCE_LENGTH, reset_every):
                seg, _ = core(xt[t0:t0 + reset_every], core.init_state(b-a, device))
                segs.append(seg)
            out = torch.cat(segs, 0)
        else:
            out, _ = core(xt, core.init_state(b-a, device))
        out = out.permute(1, 0, 2).cpu().numpy()                 # (c,20000,n_out)
        if lat is not None:
            K, qv = lat
            out = _lat.readout(torch.from_numpy(out), qv[a:b], K).numpy()[..., None]
        for k in range(b-a):
            mask = M[k] & (steps >= WARMUP)
            y = Y[k]
            if not mask.any() or not all(np.any(y[mask, j] > 0) and np.any(y[mask, j] < 0) for j in (0, 1)):
                skipped += 1; continue
            if common_head:
                sg, c = out[k][:, 0]*scale, out[k][:, 1]*scale
                p0, p1 = sg + c, -sg + c
            elif two_head:
                p0, p1 = out[k][:, 0]*scale, out[k][:, 1]*scale
            else:
                sg = out[k][:, 0]*scale; p0, p1 = sg, -sg
            preds_std.append(float(p0[mask].std()))
            scores.append(0.5*(weighted_pearson(y[mask, 0], p0[mask]) +
                               weighted_pearson(y[mask, 1], p1[mask])))
            # ALSO accumulate the pooled statistics. `scores` averages a PER-SEQUENCE correlation,
            # which is the metric that was replaced on 15 Sep; the live metric pools every scored row
            # of every sequence into ONE correlation per target. The two disagree by 0.024-0.031, and
            # most for arms that change the objective, so a mid-run reading of the old one says very
            # little. Reported as wp_pooled; `wp` is untouched so checkpoint selection and every
            # historical history.json keep their meaning.
            for _k, _p in ((0, p0), (1, p1)):
                _y = np.clip(y[mask, _k].astype(np.float64), -2.0, 2.0)
                _q = np.clip(_p[mask].astype(np.float64), -2.0, 2.0)
                _w = np.abs(_y)
                pool[_k] += np.array([_w.sum(), (_w*_q).sum(), (_w*_y).sum(),
                                      (_w*_q*_q).sum(), (_w*_y*_y).sum(), (_w*_q*_y).sum()])
        del X, Y, M, xt, out
    core.train()
    s = np.array(scores)
    pooled = []
    for _k in (0, 1):
        n_, sq, sy, sqq, syy, sqy = pool[_k]
        if n_ <= 0: continue
        mq, my = sq/n_, sy/n_
        vq, vy = sqq/n_ - mq*mq, syy/n_ - my*my
        if vq <= 0 or vy <= 0: continue
        pooled.append((sqy/n_ - mq*my)/np.sqrt(vq*vy))
    return {"wp": float(s.mean()), "sd": float(s.std()), "se": float(s.std()/np.sqrt(len(s))),
            "n": len(s), "skipped": skipped, "pred_std": float(np.mean(preds_std)),
            "wp_pooled": float(np.mean(pooled)) if pooled else float("nan")}


@torch.no_grad()
def prefix_states(core, cache, seq_ids, rows, device, xprep, chunk=4096):
    """Hidden state at `rows[k]` of sequence `seq_ids[k]`, replayed from row 0.

    Every prefix starts at row 0, so all the batch's sequences replay TOGETHER in one
    batched pass, stopping at each requested row to capture whichever sequences asked
    for it. Replaying them one at a time at batch 1 ran 18x slower than the control
    (0.81 vs 14.6 steps/s) -- a GRU at batch 1 leaves the GPU idle.

    No gradient: this sets the state the body starts from, it does not extend credit
    assignment.
    """
    uniq = sorted({int(i) for i in seq_ids})
    pos_of = {i: k for k, i in enumerate(uniq)}
    hi = int(max(rows))
    X = np.stack([np.asarray(cache.X[i, :hi], np.float32) for i in uniq])   # (U, hi, F)
    X = xprep(X, np.array(uniq))
    xt = torch.from_numpy(np.ascontiguousarray(X)).permute(1, 0, 2).to(device)

    out = [None] * len(seq_ids)
    stops = {}
    for k, (i, r) in enumerate(zip(seq_ids, rows)):
        stops.setdefault(int(r), []).append((pos_of[int(i)], k))
    h = core.init_state(len(uniq), device)
    cur = 0
    for r in sorted(stops):
        while cur < r:
            e = min(cur + chunk, r)
            _, h = core(xt[cur:e], h)
            cur = e
        for u, k in stops[r]:
            out[k] = h[:, u:u+1, :].clone()
    return torch.cat(out, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="gru")
    ap.add_argument("--hidden", type=int); ap.add_argument("--layers", type=int)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--body", type=int, default=512)
    ap.add_argument("--burn", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--final-eval-n", type=int, default=300)
    ap.add_argument("--two-head", action="store_true")
    ap.add_argument("--family-input", action="store_true",
                    help="append a 4-dim one-hot of the sequence's tick FAMILY. The target "
                         "is a P&L in ticks whose base takes exactly 4 values, one per "
                         "sequence, and per-family WP is 0.359/0.659/0.699/0.728 -- the "
                         "large-tick family is 36-42%% of sequences and carries the whole "
                         "deficit. The family is 99.8%% recoverable from warm-up rows 0-98, "
                         "so it is available causally at inference.")
    ap.add_argument("--family-mix", default=None,
                    help="comma-separated target family shares for SEQUENCE sampling, e.g. "
                         "0.35,0.24,0.18,0.23 (the holdout mix). Train is 0.44/0.35/0.12/0.09, "
                         "so the smallest-tick family is 2.5x under-represented.")
    ap.add_argument("--ssl-aux", type=float, default=0.0,
                    help="weight on SELF-SUPERVISED auxiliary heads derived from features "
                         "only: (1) does an i0 price event occur at t+1, (2) the sign of "
                         "the next change vector, (3) log rows until that event. s changes "
                         "ONLY at those events and its sign is sign(P-mid), so these are the "
                         "target's own mechanics. The label is information-poor (median 22 "
                         "distinct s values per 20,000-row sequence); the features are not. "
                         "Heads are dropped at export, so inference cost is unchanged.")
    ap.add_argument("--aux-horizons", default=None,
                    help="comma-separated future offsets, e.g. 4,16,64. Extra heads predict "
                         "s at t+k during training and are DROPPED at export, so inference "
                         "cost is unchanged. The target is a decaying-kernel sum of the next "
                         "~10-20 i0 price moves, so s_{t+k} is a related, differently-noised label.")
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--rankgauss", default=None,
                    help="npz from src/rankgauss.py: APPEND rank-gauss versions of those "
                         "columns to the raw input. A monotone per-feature remap, which an "
                         "affine input projection cannot express. Replicated on EVAL at "
                         "+0.0061 in a linear model, concentrated in rg(i0_vA).")
    ap.add_argument("--drop-blocks", default=None,
                    help="comma-separated feature blocks to REMOVE from the input, e.g. "
                         "i1_v,a. Leave-one-block-out on the linear model found i1_v is "
                         "actively harmful (+0.0087 to drop) and a is mildly so (+0.0023). "
                         "Fewer inputs also means fewer layer-0 parameters.")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="also save weights every N steps, so weight averaging (SWA/EMA) "
                         "can be tested afterwards without retraining. best.pt and final.pt "
                         "are often the SAME weights, which makes 2-point averaging vacuous.")
    ap.add_argument("--row-weight", default=None,
                    help="npy (N,20000) of P(scored | causal features), learned on valid "
                         "where is_scored exists and applied to train, which has no such "
                         "column. Multiplies the metric's own |y| weight, so the training "
                         "objective matches the SCORED row distribution instead of all rows.")
    ap.add_argument("--row-weight-power", type=float, default=1.0,
                    help="w = p**power; 0 disables, 1 is plain importance weighting.")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to CONTINUE from (fresh optimizer and schedule). "
                         "With --common-head the n_out=1 head is widened to 2: the s row "
                         "is copied and the c row zeroed, so the arm starts bit-identical "
                         "to its control.")
    ap.add_argument("--common-head", action="store_true",
                    help="predict (s, c); emit (s+c, -s+c). c starts at zero, so the "
                         "arm begins exactly at the single-signal model.")
    ap.add_argument("--loss", default="wp", choices=["wp", "mse", "seqwp", "anchored", "zmse", "seqrun"])
    ap.add_argument("--lam", type=float, default=0.5, help="anchored: weight on the pooled-WP term")
    ap.add_argument("--groups", type=int, default=8, help="seqwp: distinct sequences per batch")
    ap.add_argument("--per-seq", type=int, default=4, help="seqwp: windows per sequence")
    ap.add_argument("--mask-weight", action="store_true", help="weight loss by is_scored (train cache has no mask; off by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--decay-lr-mult", type=float, default=1.0,
                    help="LR multiplier for recurrent DECAY parameters (a_logit etc). "
                         "The S4/S4D literature uses a much lower LR and no weight decay "
                         "for these; a single global LR is a known-wrong recipe for SSMs.")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--prefix-state", choices=("off", "fam0", "all"), default="off",
                    help="build the state reaching the loss rows by replaying the "
                         "window's TRUE prefix from row 0, instead of `burn` rows from "
                         "zero. Inference carries state across 20,000 rows; training "
                         "restarts every 768. 'fam0' applies it only to family 0. "
                         "Implies --detach-burn, so compare against a --detach-burn "
                         "control, not the stock recipe.")
    ap.add_argument("--detach-burn", action="store_true",
                    help="run the burn rows under no_grad. The stock recipe "
                         "backpropagates through them, so a prefix-state arm must be "
                         "compared against a detached control or history length is "
                         "confounded with the gradient path.")
    ap.add_argument("--lattice-wp", type=float, default=0.0, metavar="W",
                    help="train the lattice head on the metric THROUGH its own tilted "
                         "readout, with cross-entropy as an auxiliary at weight W: "
                         "loss = seqwp(readout) + W*CE. W=0 is pure cross-entropy, which "
                         "never sees the metric.")
    ap.add_argument("--state-pass", type=float, default=0.0, metavar="P",
                    help="initialise each window's state from ANOTHER window's detached "
                         "final state, with probability P of a zero init instead "
                         "(0 = off, 0.1 is the published setting). Two independent "
                         "literatures converge on this: training from a zero state both "
                         "leaves attainable states unexplored (Buitrago Ruiz & Gu 2025) "
                         "and actively teaches the model not to rely on its state at all "
                         "(R2D2 2019). Measured here: resetting the deployed state every "
                         "512 rows costs only 0.0013, i.e. the carried state is nearly "
                         "decorative -- while an oracle on the whole-sequence level scores "
                         "0.873. Validation always uses a zero state.")
    ap.add_argument("--gate-init", choices=("default", "ugi", "chrono"), default="default",
                    help="initialisation of the update (RETENTION) gate bias. PyTorch's z "
                         "is retention, so chrono needs a POSITIVE bias; getting the sign "
                         "backwards makes every unit forget as fast as possible.")
    ap.add_argument("--slow-units", type=int, default=0, metavar="K",
                    help="refgru SLOW LANE (research Z25 / Z23 I2): initialise the update-gate bias of the LAST K layer-0 GRU "
                         "units to log(t), t log-spaced over [--slow-tmin, --slow-tmax] rows (zero-input time constant ~t rows). "
                         "Bias values only: no new parameters, no RNG draw, export unchanged; initialised, not clamped. "
                         "Refused with --gate-init, --init-from, split towers or an LSTM cell. 0 = off (add_z25.py)")
    ap.add_argument("--slow-tmin", type=float, default=1000.0, metavar="T",
                    help="--slow-units: shortest time constant in rows (add_z25.py)")
    ap.add_argument("--slow-tmax", type=float, default=20000.0, metavar="T",
                    help="--slow-units: longest time constant in rows (add_z25.py)")
    ap.add_argument("--global-pool", action="store_true",
                    help="pool the WP statistics ACROSS sequences instead of scoring each "
                         "sequence separately -- the aggregation the live metric uses. "
                         "Per-sequence centring cancels level drift, which the global "
                         "metric scores.")
    ap.add_argument("--pool-decay", type=float, default=1.0, metavar="F",
                    help="with --global-pool, multiply the accumulated statistics by F "
                         "each step. F<1 gives a finite effective window; F=1 accumulates "
                         "for a whole epoch, which shrinks the current chunk's share of "
                         "the moments and can stall the gradient.")
    ap.add_argument("--dir-weight", type=float, default=0.0, metavar="W",
                    help="add W * |clip(y)|-weighted BCE on the sign of the spread "
                         "component. The global metric's deficit is direction, not "
                         "magnitude (sign oracle +0.175 vs magnitude +0.018), and neither "
                         "wMSE nor pooled Pearson gives a dense direction gradient.")
    ap.add_argument("--dir-all-rows", action="store_true",
                    help="apply --dir-weight to every row instead of resolved rows only. "
                         "Off by default: perfect direction on unresolved rows is worth "
                         "+0.0009 global against +0.0891 on resolved rows.")
    ap.add_argument("--rank-weight", type=float, default=0.0, metavar="W",
                    help="seqrun: add W * (1 - weighted soft-Spearman) to the loss. A RANK "
                         "objective -- untried here; every existing loss is magnitude or "
                         "sign. Not expected to beat the metric on its own (the metric is "
                         "weighted PEARSON, and ranking discards the magnitude its weights "
                         "care about); the point is a decorrelated ensemble member, the same "
                         "way --dir-weight and --row-weight earned their v42full slots. The "
                         "target's tick lattice means ties dominate, so the true side uses "
                         "tie-corrected midranks; see losses.weighted_softrank.")
    ap.add_argument("--rank-knots", type=int, default=64, metavar="K",
                    help="quantile knots for the soft rank. Cost is O(n*K) not O(n^2).")
    ap.add_argument("--rank-tau", type=float, default=0.25, metavar="T",
                    help="soft-rank temperature, in units of the batch spread's std.")
    ap.add_argument("--dir-tau", type=float, default=0.5,
                    help="logistic temperature for --dir-weight")
    ap.add_argument("--orth-init", action="store_true",
                    help="orthogonal init for each W_hh gate block (PyTorch defaults to "
                         "uniform). Initialise, never constrain -- constraints hurt.")
    ap.add_argument("--ns-reg", type=float, default=0.0, metavar="B",
                    help="norm-stabilizer on the RECURRENT STATE: B * mean((||h_t|| - "
                         "||h_t-1||)^2). Constrains the norm of the trajectory, not its "
                         "direction, so the state may still rotate to encode new information. "
                         "The only regulariser in the literature validated on 10,000-step "
                         "rollouts (Krueger & Memisevic, ICLR 2016).")
    ap.add_argument("--ar-reg", type=float, default=0.0, metavar="A",
                    help="activation regularisation on the STATE: A * mean(h_t^2). Distinct "
                         "from output scale, which the metric is invariant to (Merity 2017).")
    ap.add_argument("--tar-h", type=float, default=0.0, metavar="B",
                    help="temporal activation regularisation on the STATE: B * mean((h_t - "
                         "h_t-1)^2). This is Merity's TAR; --tar/--tar-seq act on predictions.")
    ap.add_argument("--zoneout", type=float, default=0.0, metavar="P",
                    help="zoneout on the recurrent state: with probability P a unit keeps "
                         "h_t-1 instead of updating. Identity at test time, so inference and "
                         "the exported graph are unchanged (Krueger et al., ICLR 2017).")
    ap.add_argument("--tar", type=float, default=0.0, metavar="B",
                    help="temporal activation regularisation: B * mean((h_t - h_t+1)^2), a "
                         "slowness prior on the hidden trajectory. AR is deliberately not "
                         "offered: weighted Pearson is scale-invariant, so penalising |h| "
                         "has no rationale here.")
    ap.add_argument("--betas", default=None, metavar="B1,B2",
                    help="AdamW betas (default 0.9,0.999). The reference implementation "
                         "for the whole-sequence loss uses 0.9,0.98 with weight decay "
                         "1e-2 -- 100x ours -- and a lower beta2 adapts faster to the "
                         "changing gradient scale a running-statistics loss produces.")
    ap.add_argument("--proj", type=int, default=0,
                    help="width of a bottleneck Linear between the GRU and the head "
                         "(0 = none). Not a recurrent projection -- the state stays "
                         "`hidden` wide, so the export path is unchanged.")
    ap.add_argument("--pearson-weight", type=float, default=0.9,
                    help="weight on the sequence Pearson in --loss seqrun; the remainder "
                         "goes to a |y|-weighted MSE whose job is to pin the output scale, "
                         "which a scale-invariant correlation objective cannot.")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="rows per step in --loss seqrun. Gradients span one chunk; the "
                         "OBJECTIVE spans the whole sequence via carried statistics.")
    ap.add_argument("--n-int", type=int, default=32,
                    help="width of the linear leaky-integrator bank for --preset intgru. "
                         "Family 0's deficit is the LEVEL of the target (oracle on the true "
                         "level alone scores 0.873 vs the model's ~0.48); a tanh GRU cannot "
                         "hold an unbounded running sum, a linear recurrence can. Timescales "
                         "are initialised geometrically from ~4 to ~20,000 rows.")
    ap.add_argument("--lattice-head", type=int, default=0, metavar="K",
                    help="categorical head over the integer tick lattice k = s/q, "
                         "k in [-K, K], trained with cross-entropy and read out with the "
                         "metric's own w-TILTED mean. On raw targets s/q is an integer to "
                         "5e-04 for every family, so this is a target reparameterisation, "
                         "not family conditioning. Motivated by family 0's bimodal lattice: "
                         "57%% of rows at k=+-1, 25%% at |k|>=4, and 0.008/0.000 at +-2/+-3, "
                         "so a scalar head interpolates through a region the data never "
                         "occupies. K=8 folds 0%% of family-0 rows.")
    ap.add_argument("--fam-norm", action="store_true",
                    help="normalise inputs with one (mu, sd) PER FAMILY instead of one "
                         "global pair. After the global transform family 0's price "
                         "columns reach the GRU at std 0.62 and family 3's at 1.84; "
                         "family 0 is also the one stuck at 60%% of its perfect-sign "
                         "ceiling while the rest reach 86%%. Family comes from the "
                         "warm-up classifier (99.8%%), so this deploys as 4x112 extra "
                         "constants and no extra parameters.")
    ap.add_argument("--no-norm", action="store_true",
                    help="skip per-column standardisation; the organisers already standardised globally")
    ap.add_argument("--train-cache", default="cache/train")
    ap.add_argument("--extra-cache", default=None,
                    help="second training cache (the RESERVED valid retrain split). "
                         "Rules permit training on valid; the point is distribution "
                         "coverage, since test is a third period.")
    ap.add_argument("--extra-weight", type=float, default=1.0)
    ap.add_argument("--stagger", action="store_true",
                    help="seqrun: start lane g at chunk g*cps//G of its first sequence so the lanes are not "
                         "in lock-step (research T / M item 10; add_stagger.py)")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="seqrun: train for this many passes over the (train + extra) "
                         "sequences WITHOUT replacement; overrides --steps")
    ap.add_argument("--train-row-factor", choices=("one", "pred", "actual"), default="one",
                    help="seqrun: row factor on the primary cache. 'pred' multiplies the "
                         "metric weight by --row-weight (a predicted P(is_scored)); "
                         "'actual' uses the cache's own mask")
    ap.add_argument("--extra-row-factor", choices=("one", "actual", "mixed"), default="one",
                    help="seqrun: row factor on --extra-cache; 'actual' uses its real mask; 'mixed' = 1 + "
                         "--mix-lambda x mask (every row trains, scored rows count more; research K #2)")
    ap.add_argument("--mix-lambda", type=float, default=3.0)
    ap.add_argument("--extra-fam-repeat", default="", metavar="R0,R1,R2,R3",
                    help="seqrun: every --extra-cache sequence of tick family f (SequenceStream.family) appears R_f times in "
                         "each epoch's draw pool (ints 1..16); step count unchanged, so repeats displace other visits "
                         "(research Z46, add_z46.py)")
    ap.add_argument("--seq-repeat", default="", metavar="FILE",
                    help="seqrun: .npy int array (n_primary,) in primary-cache order; primary sequence i appears FILE[i] "
                         "times in each epoch's draw pool (0..16); step count unchanged (research Z46, add_z46.py)")
    ap.add_argument("--dropout", type=float, default=None, help="refgru: dropout (default 0.1)")
    ap.add_argument("--head-hidden", type=int, default=0, help="refgru: MLP head width (default = hidden)")
    ap.add_argument("--bookconv", type=int, default=0, metavar="C",
                    help="refgru: C channels of 1-D convolution along the order-book DEPTH axis, "
                         "shared across levels and both instruments. A restriction on the input map, "
                         "not extra capacity -- the medicine for a model that overfits its sequences.")
    ap.add_argument("--bookconv-taps", type=int, default=3, metavar="K",
                    help="refgru --bookconv: odd kernel width along the ladder index (1 = same-index cross-block read only, "
                         "3 = default, 5 = +-2 columns); the fold into in_proj.0 is unchanged (research S1, add_s1.py)")
    ap.add_argument("--bookconv-unshare", action="store_true",
                    help="refgru --bookconv: a separate kernel for i1 (default: one kernel shared by both instruments) "
                         "(research S1, add_s1.py)")
    ap.add_argument("--bookconv-split", action="store_true",
                    help="refgru --bookconv: BLOCK-DIAGONAL kernel -- one (K taps x 2) kernel for the price pair (p_bid, p_ask) -> C/2 "
                         "channels and one for the volume pair (v_bid, v_ask) -> C/2 channels (no price-volume mixing at the index); "
                         "same 2*C*11 features and in_proj.0 shape, folded at export (research H7, add_h7_bcsplit.py)")
    ap.add_argument("--bookconv-side", action="store_true",
                    help="refgru --bookconv: SIDE-TIED kernel -- ONE (K taps x 2) kernel over (p_side, v_side) applied to the bid side "
                         "-> channels 0..C/2-1 and to the ask side -> C/2..C-1 (shared over 2 instruments x 2 sides x 11 indices); "
                         "same 2*C*11 features and in_proj.0 shape, folded at export (research H7, add_h7_bcsplit.py)")
    ap.add_argument("--tradeconv", type=int, default=0, metavar="C",
                    help="refgru: C channels of 3-tap convolution along the 4 trade slots, (dp, dv) as the two input channels, "
                         "shared across both instruments (2*C*4 features); linear in x, folded into in_proj.0 at export "
                         "(research S1, add_s1.py)")
    ap.add_argument("--instshare", type=int, default=0, metavar="E",
                    help="refgru: one dense E-wide linear encoder of an instrument's 52 columns, shared by i0 and i1 (2*E features, "
                         "the per-instrument readout stays free); linear in x, folded into in_proj.0 at export (research S1, add_s1.py)")
    ap.add_argument("--inproj-init-mult", type=float, default=1.0, metavar="M",
                    help="refgru: multiply in_proj.0.weight's default init by M (shape unchanged; sqrt(112/288) = 0.6236 reproduces "
                         "--bookconv 8's init std without the conv); 1 = off (research S1 control, add_s1b.py)")
    ap.add_argument("--densein", type=int, default=0, metavar="K",
                    help="refgru: K extra inputs = one dense learned linear map of the 88 book columns, no sharing or structure "
                         "(K = 176 matches --bookconv 8's in_proj.0 shape, init and Muon scaling); linear in x, folded at export "
                         "(research S1 control, add_s1b.py)")
    ap.add_argument("--xinst", type=int, default=0, metavar="C",
                    help="refgru: C channels of a CROSS-INSTRUMENT ladder read: at each ladder index the 8 values (p_bid, p_ask, "
                         "v_bid, v_ask of i0 and i1) over 3 taps -> C, one kernel shared over the 11 indices (C*11 features); "
                         "linear in x, folded into in_proj.0 at export (research ST, add_struct.py)")
    ap.add_argument("--bookconv-act", choices=("", "gelu"), default="", metavar="ACT",
                    help="refgru --bookconv: a position-wise GELU on the shared book read BEFORE in_proj.0 (equivariant nonlinear "
                         "local features). NOT foldable: exported as w['bcnl'] for a native-kernel extension; the ONNX/fused/native "
                         "builders refuse such a member until that exists (research ST, add_struct.py)")
    ap.add_argument("--hh-rank", type=int, default=0, metavar="R",
                    help="refgru: rank-R recurrence W_hh = hh_A (3H, R) @ hh_B (R, H) -- the three gates read the state through one "
                         "shared R-dim projection (LSTMP-style); same cuDNN kernel via torch._VF.gru; export materialises W_hh "
                         "(zero cost; a factored native path would save 3H*H - (3H+H)*R int8 bytes per member) (research ST, add_struct.py)")
    ap.add_argument("--ple-bins", type=int, default=0, metavar="B",
                    help="refgru: piecewise-linear embedding (Gorishniy et al. 2022) of the 44 price-code columns (0-21, 52-73) into B "
                         "quantile bins (edges from the --stats-n training sequences, fixed, stored in the checkpoint) followed by one "
                         "B -> --ple-ch linear map shared across the 44 positions; 44 * C extra inputs to in_proj.0. Causal (row t "
                         "only); NOT foldable at export (builders refuse until a kernel path exists). 0 = off (research H4, add_h4_ple.py)")
    ap.add_argument("--ple-ch", type=int, default=8, metavar="C",
                    help="--ple-bins: channels per price-code column from the shared B -> C map (default 8) (research H4, add_h4_ple.py)")
    ap.add_argument("--learn-h0", action="store_true",
                    help="refgru: learn the initial recurrent state instead of zeros (research X; add_learn_h0.py)")
    ap.add_argument("--mask-aux", type=float, default=0.0, metavar="W",
                    help="seqrun: train-only head predicting P(is_scored) from the GRU output on post-warm-up rows whose "
                         "source cache carries a mask (BCE, weight W); dropped at export (research Z6, add_signals.py)")
    ap.add_argument("--trim", type=float, default=0.0, metavar="Q",
                    help="seqrun: each step drop the top-Q fraction of supervised rows by max_k |p_k - clip(y_k)| from the "
                         "anchor term only; Pearson and distillation unchanged (research Z6, add_signals.py)")
    ap.add_argument("--curric", type=int, default=0, metavar="E",
                    help="seqrun: epochs 1..E give row factor 0 to rows whose teacher residual exceeds the family's 90th "
                         "percentile (thresholds from 200 sequences at startup; needs --distill) (research Z6, add_signals.py)")
    ap.add_argument("--ofi", action="store_true",
                    help="refgru: conditional order-flow imbalance per level (Cont), 24 channels. "
                         "The price-change conditioning is what separates queue replenishment from "
                         "the price level moving; a raw delta conflates them.")
    ap.add_argument("--state-reset", type=int, default=0, metavar="N",
                    help="seqrun: zero the recurrent state every N rows WITHIN a sequence (0 = never, "
                         "which is what we ship today). A competitor reports this as their largest "
                         "single lever (+0.0116 under the global metric; reset2k 0.6864 against "
                         "reset4k 0.6823 on their held-out), and the mechanism fits what we have "
                         "measured independently: the metric pools ALL rows into ONE correlation, so "
                         "per-sequence level and scale matter, and a GRU's state drifts over 20,000 "
                         "rows. Periodic resets stop that drift from poisoning the pooled statistic. "
                         "Corroborating evidence from the other box: resetting the DEPLOYED state "
                         "every 512 rows costs only 0.0013, i.e. the carried state is nearly "
                         "decorative, while an oracle on the whole-sequence level scores 0.873. "
                         "CRITICAL: applying resets at inference only, to a model trained without "
                         "them, is monotonically WORSE (-0.0003/-0.0016/-0.0029 at 4000/2000/1000 on "
                         "pooled-800). The reset has to be in TRAINING, and evaluation and the served "
                         "driver must then use the SAME grid or the number means nothing.")
    ap.add_argument("--levelaux", type=float, default=0.0, metavar="W",
                    help="HINT DISTILLATION on the per-sequence LEVEL, the one gap that is measured "
                         "and unclosed: an oracle on the per-sequence level alone scores 0.873 "
                         "against the model's ~0.48 on family 0, and an oracle affine is worth "
                         "+0.064. Five instruments failed to RECOVER it, but every one of them was a "
                         "post-hoc readout (ridge on aggregates, hidden-state summary, conditional "
                         "recalibration, pseudo-labels, position bands). This instead SUPERVISES the "
                         "representation to carry the level while training, which is a different "
                         "mechanism: it changes what is learned rather than reading out what is "
                         "already there. Four extra heads -- level and log-scale per instrument -- "
                         "each regressed at EVERY row onto that chunk's |y|-weighted statistics, so "
                         "the target is acausal with respect to row t and is therefore privileged "
                         "information in the LUPI sense. Dropped at export. Expect little: a ridge "
                         "on per-sequence feature aggregates reaches only R^2 0.12-0.20 on the level "
                         "error, which caps any method of this kind at a fraction of the +0.064.")
    ap.add_argument("--densaux", type=float, default=0.0, metavar="W",
                    help="dense auxiliary supervision: predict the FORWARD mid-price change of both "
                         "instruments at --dens-horizons, at EVERY row rather than only the ~9%% that "
                         "are scored. The labels are constructed from the price columns we already "
                         "have, so this multiplies gradient signal per sequence by ~10x without adding "
                         "parameters -- aimed squarely at the memorisation blocker. Heads are extra "
                         "output columns and are dropped at export, so inference cost is zero.")
    ap.add_argument("--dens-horizons", default="10,50,200",
                    help="row offsets for --densaux")
    ap.add_argument("--dens-mid", default="", metavar="DIR",
                    help="--densaux targets from the EXACT decoded mid paths DIR/<cache>.npy (int16 half-ticks, -32768 = "
                         "unknown; exact-mid dump builder, not included in this repo) instead of the p-column means, which track the "
                         "spread; windows with an unknown row are masked (research Z37, add_z37.py)")
    ap.add_argument("--dens-cross-chunk", action="store_true",
                    help="--dens-mid only: build the exact-mid densaux targets across chunk boundaries -- row t of a chunk "
                         "reads the full-sequence path at t+h even when t+h lies in the next chunk, so every row gets every "
                         "horizon; masked only where t+h >= 20000 or a row of [t, t+h] is unknown (research Z55, add_z55.py)")
    ap.add_argument("--dens-fam-mult", default="", metavar="M0,M1,M2,M3[/N0,N1,N2,N3]",
                    help="--dens-mid only: multiply each lane's exact i0 (i1) target by M_f (N_f) of the lane's family f "
                         "(SequenceStream.family); 0 masks that family/instrument; unknown family is masked. Sets the aux "
                         "loss's family mix, which in half-ticks sits on F2/F3 (research Z38, add_z38.py)")
    ap.add_argument("--paux", default="", metavar="DIR",
                    help="seqrun: K extra EXACT auxiliary targets per row from DIR/<cache>.npy (float16 (n, 20000, K), NaN = "
                         "masked; rule-target dump builder, not included in this repo): next mid-move direction, run extent, first passages, "
                         "timing. One extra head row each, appended after every other head, dropped at export; masked "
                         "per-batch-standardised MSE per channel, summed (research Z39, add_z39.py)")
    ap.add_argument("--paux-weight", type=float, default=0.0, metavar="W",
                    help="--paux: weight per channel term (the --densaux convention)")
    ap.add_argument("--paux-fam-mult", default="", metavar="M0,M1,M2,M3[/...]",
                    help="--paux only: multiply each lane's targets by M_f of its family f (SequenceStream.family) before the "
                         "standardisation; one group for all channels or one group per channel joined by '/'; 0 masks; "
                         "unknown family masked (research Z39, add_z39.py)")
    ap.add_argument("--extra-in", default="", metavar="DIR",
                    help="seqrun: append K exact DECODED input channels per row from DIR/<cache>.npy (float16 (n, 20000, K), "
                         "NaN = unknown -> the meta's fill; feature dump builder, not included in this repo, computed online by "
                         "its online twin); the input projection widens by K, the K channels are "
                         "standardised with stats from the training dump (research Z43, add_z43.py)")
    ap.add_argument("--adanorm-cols", default="", metavar="SPEC",
                    help="refgru: apply --adanorm only to these columns: vol (52 volume-like), clk (a0..a7), vol+clk or ranges "
                         "'22-43,48-51'; the rest stay plain standardised (research Z48, add_z48.py)")
    ap.add_argument("--dann", type=float, default=0.0, metavar="W",
                    help="seqrun/refgru: domain-adversarial weight; a train-only adversary reads the head input through a "
                         "gradient-reversal layer and predicts the lane's regime group (research Z48, add_z48.py)")
    ap.add_argument("--dann-groups", default="period", metavar="period|FILE",
                    help="--dann labels: 'period' = 0 primary / 1 extra cache, or .npy ints (n_primary,) or (n_total,), -1 = none")
    ap.add_argument("--dann-hidden", type=int, default=64, help="--dann adversary hidden width")
    ap.add_argument("--dann-lr", type=float, default=1e-3, help="--dann adversary AdamW learning rate")
    ap.add_argument("--dann-stride", type=int, default=4, help="--dann: adversary reads every N-th post-warm-up row")
    ap.add_argument("--adanorm", type=float, default=0.0, metavar="HALFLIFE",
                    help="refgru: divide each input by its own RUNNING scale (half-life in rows). "
                         "Different from --ewma, which adds a smoothed channel; this rescales.")
    ap.add_argument("--proj-layers", type=int, default=1,
                    help="refgru: number of Linear/GELU/LayerNorm blocks in the input projection. "
                         "Width on this path pays (+0.0010 for 64->192) and has flattened; depth is "
                         "the untested direction on the same axis.")
    ap.add_argument("--proj-gated", action="store_true",
                    help="refgru: gated input projection (a * SiLU(b)) instead of Linear/GELU/LayerNorm. "
                         "Forms products of input features, which the additive stack cannot do cheaply.")
    ap.add_argument("--clip-norm", type=float, default=1.0,
                    help="gradient-norm clip. Hard-coded at 1.0 until now and never swept, which\n                         is a strong constraint on a GRU trained with 1000-row truncated BPTT: it\n                         caps exactly the large updates that long-range credit assignment needs.")
    ap.add_argument("--clamp", type=float, default=10.0,
                    help="refgru: standardised inputs are clipped to +-CLAMP before the projection. "
                         "Standardisation amplifies some columns 33x, so this decides how much of "
                         "the tail the recurrence ever sees; it has never been moved under this recipe.")
    ap.add_argument("--cell", choices=("gru", "lstm"), default="gru", help="refgru: recurrent cell")
    ap.add_argument("--diff", action="store_true", help="refgru: also feed x_t - x_{t-1}")
    ap.add_argument("--posfeat", action="store_true", help="refgru: feed row index / 20000 (counter in state)")
    ap.add_argument("--split-towers", action="store_true",
                    help="refgru: two recurrent towers, one per instrument (i0 blocks + extras, i1 blocks + extras), fused before the head")
    ap.add_argument("--ewma", default="", help="refgru: comma list of EWMA half-lives (rows) fed as inputs")
    ap.add_argument("--nlfeat", default="", help="refgru: comma list of half-lives for NONLINEAR causal state (realized vol, vol-normalised momentum, EWMA imbalance, EWMA |trade volume|) plus 4 instantaneous features; --ewma is linear in the inputs, this is not")
    ap.add_argument("--fam-subset", default="", help="seqrun: train only on these tick families, e.g. 0 or 1,2,3")
    ap.add_argument("--big-emph", type=float, default=0.0,
                    help="seqrun: rows with |clip(y)| >= 1.5 get row factor x(1+W) -- emphasise large moves")
    ap.add_argument("--tar-seq", type=float, default=0.0,
                    help="seqrun: temporal smoothness prior, weight on mean (p_t - p_{t-1})^2 (targets are persistent)")
    ap.add_argument("--distill", default="", metavar="DIR",
                    help="seqrun: distil from teacher predictions DIR/train_full.npy [+ DIR/valid_retrain.npy] "
                         "(float16 (N,T,2) in calibrated target units, see src/teacher_dump.py)")
    ap.add_argument("--distill-wsrc", default="y", choices=("y", "t", "geo"),
                    help="seqrun: weight of the distillation term: |y| (default), |teacher| or sqrt(|y||t|).")
    ap.add_argument("--noise-w", type=float, default=0.0, metavar="TAU",
                    help="seqrun: multiply the distillation weight by 1/(1 + r^2/TAU), r = y - teacher "
                         "(row noise estimate). 0 = off.")
    ap.add_argument("--aux-teacher", action="store_true",
                    help="seqrun: --aux-horizons heads target the TEACHER's value at t+k, not y.")
    ap.add_argument("--corr-target", default="y", choices=("y", "t"),
                    help="seqrun: the pooled-Pearson term uses the teacher (clipped) instead of y.")
    ap.add_argument("--distill2", default="", metavar="DIR",
                    help="seqrun: a SECOND distillation teacher (same file layout as --distill), matched by "
                         "|y|-weighted MSE at --distill2-weight. Lets one student learn from the in-sample "
                         "causal teacher AND the lookahead teacher at once (add_distill2.py).")
    ap.add_argument("--distill2-weight", type=float, default=0.0)
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="seqrun: weight of the |y|-weighted MSE between the prediction and the teacher")
    ap.add_argument("--distill-mode", choices=("mse", "corr"), default="mse",
                    help="seqrun: how the student is pulled toward the teacher. 'mse' is the "
                         "|y|-weighted squared error (the original, worth +0.0022 resolved on "
                         "pooled-800). 'corr' instead maximises the |y|-weighted PEARSON between "
                         "student and teacher, per target. The rationale is that the live metric "
                         "is scale-invariant, so MSE spends capacity matching the teacher's "
                         "MAGNITUDE as well as its shape, and only the shape is scored.")
    ap.add_argument("--ncl", type=float, default=0.0, metavar="LAMBDA",
                    help="negative-correlation learning: train TWO members jointly and ADD "
                         "LAMBDA * their |y|-weighted prediction CORRELATION to the loss, which pushes "
                         "their predictions apart. The ensemble ceiling is set by member correlation "
                         "(0.987 -> +0.004 at any M), so the gate is corr <= 0.94, not the loss. "
                         "Writes runs/TAG_m0 and runs/TAG_m1 as ordinary checkpoints.")
    ap.add_argument("--fastw", type=int, default=0, metavar="D",
                    help="refgru: width of the online associative memory that estimates this "
                         "sequence's own transition operator (0 = off; 32 is about 2.5%% more "
                         "work per row). See section 37: the per-sequence latent is what the "
                         "plateau is made of, and a fixed-weight GRU can only average over it.")
    ap.add_argument("--fastw-gru", action="store_true",
                    help="refgru: feed the fast-weight readout into the GRU input instead of "
                         "the head, so the recurrence conditions on the sequence's own dynamics")
    ap.add_argument("--fastw-groups", type=int, default=4,
                    help="refgru: number of decay groups in the fast-weight memory")
    ap.add_argument("--fastw-hl", default="32,4096", metavar="MIN,MAX",
                    help="refgru: half-life range (rows) the fast-weight decay groups span at init")
    ap.add_argument("--quad", action="store_true",
                    help="refgru: feed the GRU z * (z - z_prev) alongside z -- level x change in the learned "
                         "projection basis. The input path (one narrow GELU layer) approximates products "
                         "badly, and every feature test so far has been additive, while the order-flow "
                         "quantity the LOB literature points at is multiplicative.")
    ap.add_argument("--ncl-warm", type=float, default=0.0, metavar="F",
                    help="with --ncl, apply no diversity pressure for the first F of training, then ramp it "
                         "in linearly over the next F. Applied from step 0 the penalty stops the second "
                         "member ever learning (it ended at 0.58 against 0.65); warming up lets both reach "
                         "the plateau first and only then pushes them apart.")
    ap.add_argument("--ncl-target", type=float, default=1.0, metavar="T",
                    help="with --ncl, penalise only the EXCESS correlation above T: lambda * relu(corr - T). "
                         "A constant push is paid for by one member degrading (at lambda 0.05 the members "
                         "reached corr 0.883 but split 0.662/0.615, so the pair lost); a hinge releases the "
                         "pressure once the members are far enough apart and lets each return to its own optimum.")
    ap.add_argument("--block-drop", type=float, default=0.0, metavar="P",
                    help="with probability P per chunk, zero ONE of the nine feature blocks (i0/i1 price, "
                         "volume, dp, dv, and the clock block). Section 30 showed the models memorise "
                         "sequence-specific structure; forcing redundancy across blocks attacks that where "
                         "plain input noise (which hurt) does not.")
    ap.add_argument("--rec-wd", type=float, default=0.0, metavar="W",
                    help="extra weight decay applied ONLY to the recurrent W_hh matrices. The memorised "
                         "component lives in the recurrent state (a last-layer refit could not touch it, "
                         "section 33c), so decay it specifically rather than the whole network.")
    ap.add_argument("--cosine-loss", action="store_true",
                    help="replace the correlation term with the pooled |y|-weighted COSINE between the "
                         "centred prediction and the centred target: cov / sd(p). The metric divides by "
                         "sd(y) too, but that is a constant of the data, so the gradient differs only in "
                         "how batch-to-batch target variance is handled -- a lower-variance estimator of "
                         "the same objective.")
    ap.add_argument("--smooth-anchor", type=int, default=0, metavar="K",
                    help="denoise the REGRESSION ANCHOR's target with a centred moving average of "
                         "+-K rows (the correlation term keeps the exact metric target). The targets "
                         "are strongly autocorrelated (0.86 at lag 10), so the average is a lower-noise "
                         "estimate of the same conditional mean; labels are training-time data, so this "
                         "is denoising, not leakage.")
    ap.add_argument("--smooth-both", action="store_true",
                    help="with --smooth-anchor, also use the smoothed target in the correlation term")
    ap.add_argument("--anchor", choices=("mse", "huber", "l1"), default="mse",
                    help="seqrun: form of the (1 - pearson-weight) regression anchor")
    ap.add_argument("--hetero", type=float, default=0.0, metavar="W",
                    help="seqrun/refgru: ADD W x |clip(y)|-weighted Gaussian NLL 0.5*(exp(-s)(p-y)^2 + s) with a train-only "
                         "per-target log-variance head s (clamped to [-6, 4]); the plain anchor stays at (1 - pearson-weight). "
                         "State-dict keys hetero_head.*, never exported (research Z5; add_losses2.py)")
    ap.add_argument("--resnorm-w", type=float, default=0.0, metavar="W",
                    help="seqrun/refgru: W x row-weighted Gaussian NLL log(sigma) + 0.5 ((y - p)/sigma)^2 of a train-only per-target "
                         "log-sigma head on the head's hidden layer (keys resnorm_head.*, never exported; research H12, add_h12_resnorm.py)")
    ap.add_argument("--resnorm-beta", type=float, default=1.0, metavar="B",
                    help="--resnorm-w: beta-NLL exponent; each row's term x sg(sigma^(2B)). 1 = p's gradient is exactly the MSE "
                         "anchor's (sigma = auxiliary calibrated-CI task); 0 = plain heteroscedastic NLL (p's row weight / sigma^2)")
    ap.add_argument("--resnorm-mode", choices=("add", "replace"), default="add",
                    help="--resnorm-w: add = on top of the full loss; replace = the NLL takes the MSE anchor's slot and weight")
    ap.add_argument("--resnorm-famnorm", action="store_true",
                    help="--resnorm-w: x the row-weighted harmonic mean of sigma^2 over the lane's family (chunk), so the family "
                         "shares of the term stay the anchor's and only the within-family row allocation moves (B < 1)")
    ap.add_argument("--resnorm-target", choices=("y", "t"), default="y",
                    help="--resnorm-w: residual against the clipped label (y) or the lookahead KD teacher's prediction (t)")
    ap.add_argument("--resnorm-weighting", choices=("metric", "none"), default="metric",
                    help="--resnorm-*: row weights |clip y| x row factor (metric, the anchor's) or the row factor alone (none)")
    ap.add_argument("--resnorm-ed", type=float, default=0.0, metavar="W2",
                    help="seqrun/refgru: W2 x energy distance between the row-weighted standardised residuals z and N(0,1) (closed-form "
                         "cross terms, one sort; 0 iff z ~ N(0,1) in distribution); p detached inside z unless --resnorm-ed-p")
    ap.add_argument("--resnorm-ed-fam", action="store_true", help="--resnorm-ed: one energy distance per (family, target), mass-weighted mean")
    ap.add_argument("--resnorm-ed-p", action="store_true", help="--resnorm-ed: let the energy-distance gradient reach the prediction p")
    ap.add_argument("--resnorm-sd0", type=float, default=1.2, metavar="S", help="--resnorm-*: initial sigma (head bias = log S)")
    ap.add_argument("--xlane-pw", type=float, default=0.0, metavar="W",
                    help="seqrun: ADD W x (1 - r), r = |clip(y)|-weighted Pearson pooled over ALL lanes x rows of the chunk "
                         "(one centre/scale across lanes, per target): the per-step surrogate of the globally pooled metric; "
                         "distinct from the per-lane running Pearson and from --global-pool (research Z5; add_losses2.py)")
    ap.add_argument("--anchor-clip", action="store_true",
                    help="seqrun: compute the MSE anchor on clip(p, +-2) with a straight-through gradient inside |p| < 2.2 "
                         "(zero outside), so over-shoots the metric clips away are not penalised (research Z5; add_losses2.py)")
    ap.add_argument("--huber-delta", type=float, default=0.5, help="seqrun: Huber transition point in target units")
    ap.add_argument("--in-noise", type=float, default=0.0, metavar="SIGMA",
                    help="seqrun: Gaussian input noise of SIGMA train-sd per feature (augmentation against sequence memorisation)")
    ap.add_argument("--stats-n", type=int, default=64,
                    help="sequences used for the standardisation mu/sd (research M #3: 64 under-estimates ten "
                         "price columns' sd by 1.4-2x; the +-10 clamp then binds on 4.7%% of scored rows)")
    ap.add_argument("--stats-cache", default="", metavar="CACHE",
                    help="take the per-column normalisation (mu, sd) from this cache instead of the first 64 train sequences")
    ap.add_argument("--zmse-weight", type=float, default=0.0,
                    help="seqrun: extra weighted MSE to the RUNNING per-sequence z-score of the clipped target")
    ap.add_argument("--focal", type=float, default=0.0,
                    help="seqrun: focal exponent on the per-sequence (1 - corr) loss (0 = off)")
    ap.add_argument("--soft-clip", action="store_true",
                    help="seqrun: pass predictions through 2*tanh(p/2) inside the loss, so the "
                         "model trains through the metric's +-2 clip instead of drifting past it")
    ap.add_argument("--sign-balance", default="", help="seqrun: families whose rows get 0.5/P_seq(sign) weight so both directions weigh equally")
    ap.add_argument("--fam0-mix", type=float, default=None,
                    help="seqrun: family-0 sequences get row factor MIX*mask + (1-MIX)*1 "
                         "(0 = all-row supervision on family 0, 1 = the arm's mask factor)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="seqrun: track an EMA of the weights at this decay and evaluate "
                         "it beside the raw weights (0 = off)")
    ap.add_argument("--ema-extra", default="", metavar="SPEC",
                    help="seqrun (add_ems.py, research EMS): extra weight averages beside the raw weights and the --ema "
                         "copy, NEVER used for training, calibrated at the end and saved as runs/<tag>/ema_<name>_final.pt "
                         "(ema_final.pt format). SPEC = comma list of name:decay:start_frac[:end_decay[:ramp_from_frac]] "
                         "(EMA; start 0 = from the initial weights like --ema, else reset to the current weights after "
                         "step round(start*steps); decay ramped linearly to end_decay from step round(ramp_from*steps) to "
                         "the last step) or name:swa:start_frac (uniform running mean from step round(start*steps)). "
                         "Empty = off")
    ap.add_argument("--ref-opt", action="store_true",
                    help="reference optimiser: AdamW with weight decay only on matrices, "
                         "5%% linear warm-up then cosine to 2%% of peak (instead of OneCycle)")
    ap.add_argument("--fam0-repeat", type=int, default=1,
                    help="draw family-0 sequences this many times per epoch. Family 0 (largest "
                         "tick) is ~42%% of sequences but 49%% of the |y| weight mass, and scores "
                         "0.4263 there vs 0.72-0.80 on families 1-3.")
    ap.add_argument("--seq-subset", default=None,
                    help=".npy of sequence indices to sample from (pseudo-period experiment)")
    ap.add_argument("--seq-subset-all", action="store_true",
                    help="seqrun: --seq-subset indexes the UNION of primary and extra caches (primary "
                         "first), so extra-cache sequences can be held out too -- cross-fitting folds.")
    ap.add_argument("--wdrop", type=float, default=0.0, metavar="P",
                    help="seqrun: DropConnect on gru.weight_hh_l0 -- one Bernoulli(1-P) mask per chunk, "
                         "scaled by 1/(1-P), restored before the optimiser step with the gradient masked "
                         "(AWD-LSTM weight-drop). The recurrence otherwise has NO stochastic regulariser "
                         "in a one-layer GRU. Inference uses the raw weights.")
    ap.add_argument("--in-noise-hl", type=float, default=0.0, metavar="H",
                    help="seqrun: make --in-noise temporally correlated, AR(1) with half-life H rows "
                         "(0 = i.i.d. per row). A GRU averages i.i.d. row noise away; persistent noise "
                         "is what perturbs the state.")
    ap.add_argument("--dualmode", type=int, default=0, metavar="W",
                    help="refgru: dual-mode student -- train-only backward GRU on W-row windows + an aux "
                         "non-causal head trained on y; the causal head is distilled in place from it at "
                         "--distill-weight. Export unchanged (add_dualmode.py).")
    ap.add_argument("--dualmode-weight", type=float, default=1.0)
    ap.add_argument("--dualmode-init", default="", metavar="CKPT",
                    help="warm-start aux_bwd from a --bidir teacher checkpoint's *_reverse weights")
    ap.add_argument("--grid-head", type=int, default=0, metavar="K",
                    help="refgru: K-bin value grid over [-2,2] with a softmax readout instead of a scalar head "
                         "(research P: the targets are lattice-valued); loss unchanged unless --grid-ce")
    ap.add_argument("--grid-ce", type=float, default=0.0, metavar="W",
                    help="with --grid-head: add W x |clip(y)|-weighted two-hot cross-entropy on the grid logits")
    ap.add_argument("--hlg", type=float, default=0.0, metavar="W",
                    help="seqrun/refgru two-head: ADD W x |clip(y)|-weighted HL-Gauss cross-entropy from a train-only head on the "
                         "main head's hidden layer (K bins over [-2.2, 2.2], Gaussian target of clip(y)); hlg_head.* is dropped "
                         "from the exported state (research Z59, add_z59.py)")
    ap.add_argument("--hlg-k", type=int, default=41, metavar="K", help="--hlg: number of bins (default 41, width 0.107)")
    ap.add_argument("--hlg-sigma", type=float, default=0.75, metavar="S", help="--hlg: Gaussian sigma in bin widths (default 0.75)")
    ap.add_argument("--mh", type=int, default=0, metavar="K",
                    help="seqrun/refgru two-head: TabM readout -- K sub-heads of width hh/K, output = their mean; ADD --mh-w x "
                         "the mean of the per-head (pearson + anchor) losses; folded into the single head at save time "
                         "(research Z59, add_z59.py)")
    ap.add_argument("--mh-w", type=float, default=1.0, metavar="W", help="--mh: weight of the mean per-head loss (default 1.0)")
    ap.add_argument("--wp-rescale", action="store_true",
                    help="seqrun: rescale each chunk's running-Pearson loss by (accumulated mass / chunk mass), "
                         "detached, so the Pearson gradient does not decay ~1/k along the sequence (research M #1)")
    ap.add_argument("--in-drop", type=float, default=0.0, help="refgru: variational dropout on the GRU input (train-only)")
    ap.add_argument("--out-drop", type=float, default=0.0, help="refgru: variational dropout on the GRU output (train-only)")
    ap.add_argument("--sched", choices=("cos", "d2z", "wsd"), default="cos",
                    help="--ref-opt LR shape after warm-up: cosine to 2%% (default), linear decay to zero, or "
                         "warmup-stable-decay (constant, linear to zero over the last 20%%)")
    ap.add_argument("--aux-next", type=float, default=0.0, metavar="W",
                    help="self-supervised aux head: predict the next row's standardised features from the GRU "
                         "output (Huber), weight W; train-only, dropped at export (add_aux_next.py)")
    ap.add_argument("--z14-reparam", choices=("off", "wn", "sn"), default="off",
                    help="factor parametrisation of the GRU matrices (research Z14, add_z14.py): wn = weight normalisation "
                         "W = g * V/||V||_row (direction V updated like a matrix -- Muon under --opt muon -- one norm g per "
                         "output unit on AdamW, no decay); sn = spectral normalisation per gate block W_k = g_k V_k/sigma_max(V_k) "
                         "(one power iteration per step). The cuDNN GRU keeps reading a PLAIN weight: W is materialised from the "
                         "factors before every forward and after every step, dL/dW is chained into the factors, and every saved "
                         "checkpoint holds the plain weight (factors dropped). off = unchanged")
    ap.add_argument("--z14-reparam-set", choices=("hh", "ih", "hhih"), default="hh",
                    help="--z14-reparam: which GRU matrices -- recurrent weight_hh (default), input weight_ih, or both (add_z14.py)")
    ap.add_argument("--z14-in-gain", action="store_true",
                    help="learned per-column input scale exp(logg) on the standardised input BEFORE the +-clamp (init 1, AdamW no "
                         "decay), i.e. each column's clamp point is learned; folded into the saved sd (sd' = sd/exp(logg)) so "
                         "checkpoints and exports stay plain (add_z14.py)")
    ap.add_argument("--z14-in-shift", action="store_true",
                    help="learned per-column input offset (sd units) before the clamp; folded into the saved mu (add_z14.py)")
    ap.add_argument("--opt", choices=("adamw", "sgd", "muon", "lion", "soap"), default="adamw",
                    help="optimizer under --ref-opt: adamw (default), sgd with Nesterov momentum (--momentum), "
                         "muon (orthogonalised momentum on hidden matrices + AdamW elsewhere, --muon-*) or "
                         "lion (sign momentum, --lr 3-10x lower and --wd 3-10x higher than AdamW's; add_optimizers.py) or "
                         "soap (Shampoo eigenbases + Adam on hidden matrices, AdamW elsewhere, --soap-*; add_soap.py)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--muon-lr", type=float, default=0.02, help="--opt muon: lr of the Muon groups (reference 0.02)")
    ap.add_argument("--muon-momentum", type=float, default=0.95, help="--opt muon: Nesterov momentum (reference 0.95)")
    ap.add_argument("--muon-ns", type=int, default=5, help="--opt muon: Newton-Schulz iterations (reference 5)")
    ap.add_argument("--muon-wd", type=float, default=-1.0,
                    help="--opt muon: decoupled weight decay of the Muon groups, applied as p *= 1 - muon_lr*wd "
                         "(default -1 = --wd)")
    ap.add_argument("--muon-scope", choices=("all", "gru", "nohead", "noinproj", "noinproj0", "noinproj3"), default="all",
                    help="--opt muon/soap: all (default) = every hidden 2-D matrix incl. in_proj/head.0; gru = only the GRU/LSTM "
                         "weight matrices are orthogonalised, proj/head matrices stay on AdamW (research Z23 I1); nohead / noinproj = all but "
                         "head.0 / in_proj.* (add_muonscope2.py); noinproj0 / noinproj3 = all but in_proj.0 / in_proj.3 "
                         "(research Z27, add_z27.py)")
    ap.add_argument("--muon-adam", type=float, default=0.0, metavar="X",
                    help="--opt muon: the in_proj Muon matrices ALSO take X times the AdamW update of the same gradient "
                         "every step (own moments, lr X*--lr on the same schedule, --betas, no extra decay); 0 = off "
                         "(research Z27 H2, add_z27.py)")
    ap.add_argument("--muon-adam-on", choices=("inproj", "inproj0", "inproj3"), default="inproj",
                    help="--muon-adam: which Muon matrices get the extra AdamW update: both input-projection Linears "
                         "(default), in_proj.0 only, or in_proj.3 only (add_z27.py)")
    ap.add_argument("--muon-gate-split", type=int, default=1, choices=(0, 1),
                    help="--opt muon: orthogonalise GRU/LSTM gate stacks per gate block (1, default) or whole (0)")
    ap.add_argument("--muon-rownorm", type=float, default=0.0, metavar="B2",
                    help="--opt muon: NorMuon per-row second-moment normalisation of the orthogonalised update, EMA factor B2 "
                         "(paper 0.95), Frobenius norm restored per gate block; 0 = off (add_z8.py)")
    ap.add_argument("--muon-cautious", type=int, default=0, choices=(0, 1, 2),
                    help="--opt muon: Cautious masking (zero update entries whose sign disagrees with the gradient's, rescale by "
                         "numel/(kept+1)): 0 off, 1 the Muon matrices, 2 every group incl. the AdamW ones (add_z8.py)")
    ap.add_argument("--muon-mars", type=float, default=0.0, metavar="GAMMA",
                    help="--opt muon: MARS-approx variance reduction c = g + GAMMA*mom/(1-mom)*(g - g_prev), ||c|| <= 1 per "
                         "matrix, fed to the momentum buffer (paper 0.025); 0 = off (add_z8.py)")
    ap.add_argument("--muon-scale", choices=("kj", "rms"), default="kj",
                    help="--opt muon: update scale: kj = max(1, rows/cols)**0.5 (reference, default); rms = 0.2*sqrt(max(rows, "
                         "cols)) (Moonshot, per-element RMS 0.2 like AdamW; movement-matched to kj 0.005 at --muon-lr ~1.6e-3) (add_z8.py)")
    ap.add_argument("--muon-nesterov", type=int, default=1, choices=(0, 1),
                    help="--opt muon: 1 = Nesterov form g.lerp(buf, mom) (reference, default), 0 = heavy-ball buffer (add_z8.py)")
    ap.add_argument("--lion-betas", default="0.9,0.99", help="--opt lion: betas (paper default 0.9,0.99)")
    ap.add_argument("--log-every", type=int, default=100, help="seqrun: print the step line every N steps")
    ap.add_argument("--soap-lr", type=float, default=-1.0,
                    help="--opt soap: lr of the SOAP groups (default -1 = --lr), times --soap-lr-mult (add_soap.py)")
    ap.add_argument("--soap-lr-mult", type=float, default=-1.0,
                    help="--opt soap: multiplier on the SOAP lr; -1 (default) = measured at the first update so the SOAP "
                         "step RMS matches AdamW's on the same gradients (clamped to [0.5, 2], printed); > 0 pins it")
    ap.add_argument("--soap-precond-freq", type=int, default=10,
                    help="--opt soap: recompute the eigenbases of L and R (torch.linalg.eigh, fp32) every N updates (reference 10)")
    ap.add_argument("--soap-beta2", type=float, default=0.95,
                    help="--opt soap: EMA factor of the Shampoo Gram factors L = GG^T and R = G^TG (reference 0.95)")
    ap.add_argument("--soap-eps", type=float, default=1e-8,
                    help="--opt soap: Adam eps in the eigenbasis (added after the bias correction, torch.optim.AdamW convention)")
    ap.add_argument("--soap-betas", default="",
                    help="--opt soap: Adam betas in the eigenbasis (default empty = --betas)")
    ap.add_argument("--soap-wd", type=float, default=-1.0,
                    help="--opt soap: decoupled weight decay of the SOAP groups, p *= 1 - soap_lr*mult*wd (default -1 = --wd)")
    ap.add_argument("--distill-ramp", choices=("none", "up", "down"), default="none",
                    help="scale the distillation weight linearly over training (up: 0 -> 2x, down: 2x -> 0); "
                         "the mean dose stays --distill-weight (research G3)")
    ap.add_argument("--anchor-target", choices=("y", "t"), default="y",
                    help="seqrun anchor target: y (default) or the clipped teacher output (research G5)")
    ap.add_argument("--y-weight", type=float, default=1.0,
                    help="seqrun: multiply the y-based loss (WP + anchor [+ densaux]) by this before the "
                         "distillation/hint terms are added; 0 trains on the teacher alone")
    ap.add_argument("--hint", default="", metavar="CKPT",
                    help="hint distillation: a --bidir teacher checkpoint run online; the student's GRU "
                         "output (through a train-only linear adapter) is pulled to the teacher's forward "
                         "GRU output by row-masked MSE at --hint-weight (add_hint.py)")
    ap.add_argument("--hint-weight", type=float, default=0.0)
    ap.add_argument("--self-distill", type=float, default=0.0, metavar="W",
                    help="seqrun: mean-teacher self-distillation -- the --ema copy runs on the same chunks (no grad, eval "
                         "mode) with its own carried state, and the student is pulled to its predictions by |clip(y)| x "
                         "row-weighted MSE at weight W (Tarvainen & Valpola 2017; ELR, Liu et al. 2020); 0 = off (add_z9.py)")
    ap.add_argument("--self-distill-start", type=float, default=0.0, metavar="F",
                    help="--self-distill: apply the term from step F*steps on (the EMA state is carried from step 1); "
                         "0 = from the first step (add_z9.py)")
    ap.add_argument("--lookahead-k", type=int, default=0, metavar="K",
                    help="Lookahead optimizer wrapper (Zhang et al. 2019) around the inner optimizer: after every K-th step "
                         "the slow weights move alpha of the way to the fast weights and the fast weights are reset to them; "
                         "0 = off (add_z9.py)")
    ap.add_argument("--lookahead-alpha", type=float, default=0.5, metavar="A",
                    help="--lookahead-k: slow-weight step (paper default 0.5) (add_z9.py)")
    ap.add_argument("--elr", type=float, default=0.0, metavar="W",
                    help="seqrun: early-learning regularisation (Liu et al. 2020) -- every row keeps a float16 temporal "
                         "ensemble of the model's OWN earlier-pass predictions; the student is pulled to it where it exists by "
                         "|clip(y)| x row-weighted MSE at weight W (the first pass over a tape is made by a model that never "
                         "trained on it, so the target carries no cross-epoch memorisation); 0 = off (add_z12.py)")
    ap.add_argument("--elr-beta", type=float, default=0.7, metavar="B",
                    help="--elr: temporal-ensemble momentum S <- B S + (1-B) pred per visit (paper 0.7/0.9); 0 = previous pass only (add_z12.py)")
    ap.add_argument("--elr-warm", type=float, default=0.0, metavar="F",
                    help="--elr: predictions made at steps <= F*steps are not recorded (those rows get their first target on "
                         "the next visit); 0 = record from step 1 (add_z12.py)")
    ap.add_argument("--train-reset-p", type=float, default=0.0, metavar="P",
                    help="seqrun, TRAIN ONLY: at every chunk boundary inside a sequence each lane's carried recurrent state is "
                         "zeroed with probability P (own rng, data order unchanged; the running-WP statistics are kept; "
                         "evaluation and export carry the state over all 20,000 rows as today). 0 = off (add_z15.py)")
    ap.add_argument("--chunk-jitter", action="store_true",
                    help="seqrun: the first chunk of every sequence has a random length 1..chunk-1 (own rng), so the TBPTT "
                         "truncation points move every epoch instead of sitting at the same multiples of --chunk; 41 chunks "
                         "per sequence instead of 40 (--epochs accounts for it). Lanes must be in lock-step (no --stagger) (add_z15.py)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TensorFloat-32 matmuls (cuBLAS + cuDNN) instead of the pinned full fp32; deterministic kernels "
                         "are kept, so a run stays reproducible on the same GPU model, but it is not bit-identical to an fp32 "
                         "run and the cross-box rebuild guarantee is weaker; no effect on CPU (add_z15.py)")
    ap.add_argument("--pw-schedule", choices=("const", "ramp", "step"), default="const",
                    help="seqrun: schedule of the running-Pearson weight. const = --pearson-weight from step 1 (today); "
                         "step = 0 until --pw-start of training, then --pearson-weight; ramp = linear 0 -> --pearson-weight "
                         "over the first --pw-start of training, then constant. The anchor gets 1 - pw(step); distillation "
                         "unchanged; the running-WP prefix is accumulated at every step regardless (add_z18.py)")
    ap.add_argument("--pw-start", type=float, default=0.0, metavar="F",
                    help="--pw-schedule step/ramp: fraction of training before the Pearson term is active (step) or fully "
                         "ramped (ramp) (add_z18.py)")
    ap.add_argument("--anchor-end", type=float, default=0.0, metavar="G",
                    help="seqrun: the anchor weight is 0 after fraction G of training (pure Pearson [+ distillation] "
                         "fine-tune); under --pw-schedule step G must not precede --pw-start; 0 = off (add_z18.py)")
    ap.add_argument("--drop-cols", default=None, metavar="SPEC",
                    help="raw input columns to REMOVE without changing the graph: their columns of in_proj.0.weight are "
                         "zeroed at init and gradient-masked, so train, eval and the exported 112-wide weights all see them "
                         "as absent. SPEC = comma list of index c, half-open range lo:hi, or block name (i0_p,i0_v,i0_dp,"
                         "i0_dv,i1_p,i1_v,i1_dp,i1_dv,a) (add_z18.py)")
    ap.add_argument("--keep-cols", default=None, metavar="SPEC",
                    help="the complement of --drop-cols: keep only these raw columns (same SPEC syntax) (add_z18.py)")
    ap.add_argument("--distill-look", choices=("none", "hi", "lo"), default="none",
                    help="seqrun KD: multiply the distillation row weight by a function of the row's position k = row %% 100 inside "
                         "the teacher's 100-row dump window (the teacher saw 99-k future rows there). hi = ((100-k)/100)^P favours "
                         "the future-heavy rows, lo = ((1+k)/100)^P the near-causal rows; P = --distill-look-pow; the KD dose is "
                         "re-normalised. Needs --chunk a multiple of 100 dividing 20,000; refused with --chunk-jitter/--stagger/"
                         "--row-drop/--splice (add_z20.py)")
    ap.add_argument("--distill-look-pow", type=float, default=1.0, metavar="P",
                    help="--distill-look: exponent of the position weight (add_z20.py)")
    ap.add_argument("--distill-dis", type=float, default=0.0, metavar="B",
                    help="seqrun KD: bounded disagreement boost of the distillation row weight, x (1 + B*r2/(TAU+r2)) with "
                         "r2 = mean_k (clip(y)-clip(t))^2 (the mirror of --noise-w); 0 = off (add_z20.py)")
    ap.add_argument("--distill-dis-tau", type=float, default=0.2, metavar="TAU",
                    help="--distill-dis: r2 scale; 0.2 = the median r2 of the window-100 teacher on train_full (add_z20.py)")
    ap.add_argument("--distill-step", type=float, default=0.0, metavar="F",
                    help="seqrun KD: the distillation weight is --distill-weight for steps 1..F*steps and --distill-w2 after; "
                         "0 = off; not with --distill-ramp (add_z20.py)")
    ap.add_argument("--distill-w2", type=float, default=0.0, metavar="W2",
                    help="--distill-step: the distillation weight after the step (default 0 = y-only fine-tune) (add_z20.py)")
    ap.add_argument("--distill-clip", action="store_true",
                    help="seqrun KD: distil to clip(t, +-2) instead of the raw teacher value (add_z20.py)")
    ap.add_argument("--bidir", action="store_true",
                    help="refgru: BIDIRECTIONAL (non-causal) GRU -- a distillation TEACHER only, never a "
                         "member and not exportable. The backward direction restarts every chunk in "
                         "training; evaluate()/teacher_dump see the whole sequence.")
    ap.add_argument("--sam", type=float, default=0.0, metavar="RHO",
                    help="seqrun: sharpness-aware minimisation, stale-direction variant -- perturb the "
                         "weights by RHO * g_prev/||g_prev|| before the forward, take this step's gradient "
                         "there, restore before the optimiser step. No second forward pass.")
    ap.add_argument("--pretrain-steps", type=int, default=0, metavar="N",
                    help="seqrun: self-supervised pretraining of the core for N chunks before the main "
                         "loop (next-row prediction, or masked-row reconstruction with --pretrain-mask); "
                         "the pretraining head is discarded, export unchanged (add_pretrain.py)")
    ap.add_argument("--pretrain-lr", type=float, default=0.0, help="pretraining lr (0 = --lr)")
    ap.add_argument("--pretrain-mask", type=float, default=0.0, metavar="P",
                    help="0: next-row prediction; P>0: mask input rows with prob P (column mean) and "
                         "reconstruct the current standardised row there")
    ap.add_argument("--pretrain-horizon", type=int, default=1, metavar="K", help="next-row horizon")
    ap.add_argument("--row-drop", type=float, default=0.0, metavar="P",
                    help="seqrun: drop each row of a chunk with prob P from input, targets, weights and "
                         "teacher targets alike (random time warp; state carried). Train-only "
                         "(add_rowdrop.py)")
    ap.add_argument("--row-drop-warp", action="store_true",
                    help="draw the drop rate per chunk from U(0, 2P) instead of the constant P")
    ap.add_argument("--splice", type=float, default=0.0, metavar="P",
                    help="seqrun: with prob P a lane keeps its recurrent state when it starts a new "
                         "sequence (two sequences spliced, state carried); loss statistics still "
                         "restart. Train-only (add_splice.py)")
    ap.add_argument("--seq-affine", type=float, default=0.0, metavar="S",
                    help="seqrun: per-sequence random gain exp(S*N(0,1)) on --seq-affine-cols, drawn per "
                         "lane at sequence start and held for the sequence. Train-only (add_seqaffine.py)")
    ap.add_argument("--seq-affine-shift", type=float, default=0.0, metavar="T",
                    help="per-sequence random shift T*N(0,1) in train-sd units on the same columns")
    ap.add_argument("--seq-affine-cols", default="0:22,52:74",
                    help="comma list of lo:hi column ranges (default: both instruments' price levels)")
    ap.add_argument("--clock-aug", type=float, default=0.0, metavar="DELTA",
                    help="seqrun: per-sequence wall-clock DILATION of the log-time columns a0,a2,a3,a5,a6 (raw cols "
                         "104,106,107,109,110): one u ~ U(-DELTA, DELTA) per lane per sequence (a0 sd units), a_j += u*r_j; "
                         "values at the -5.19934 clip stay put, results clipped to +-5.19934; labels, teacher, eval and "
                         "export untouched; 0 = off (research Z36, add_z36clk.py)")
    ap.add_argument("--clock-ratios", default="1.0,2.0,2.7,0.4,0.3", metavar="R0,R2,R3,R5,R6",
                    help="--clock-aug: shift of a0,a2,a3,a5,a6 per unit u, i.e. sigma_a0/sigma_aj (Z36 Step 0 fits)")
    ap.add_argument("--vaug", type=float, default=0.0, metavar="T",
                    help="seqrun: label-exact VOLUME-LEVEL augmentation -- per lane (= per sequence, redrawn at each sequence "
                         "start) and per column b ~ N(0, T) train-sd units added to the raw --vaug-cols (default i0_v + i1_v); "
                         "labels, weights, teacher and densaux targets, eval and export untouched; 0 = off (research G1, add_g1.py)")
    ap.add_argument("--vaug-gain", type=float, default=0.0, metavar="S",
                    help="--vaug: also a per-lane per-column gain exp(S*N(0,1)) around the column mean: x' = mu + g*(x-mu) + b*sd "
                         "(0 = gain 1; may be used without --vaug T)")
    ap.add_argument("--vaug-mode", choices=("percol", "common"), default="percol",
                    help="--vaug: percol = independent draw per column (default); common = one draw per lane for every column")
    ap.add_argument("--vaug-fams", default="", metavar="F,F,..",
                    help="--vaug: augment only lanes whose sequence is in these tick families (e.g. 1,2,3); default all")
    ap.add_argument("--vaug-cols", default="22:44,74:96", metavar="lo:hi,..",
                    help="--vaug: raw columns, inside 22:52 or 74:104 only (default both instruments' 22 volume columns)")
    ap.add_argument("--pcgrad", choices=("off", "fam", "term"), default="off",
                    help="seqrun: gradient surgery (PCGrad, Yu et al. 2020) between task gradients that share the network -- fam = the "
                         "four tick families' lanes, term = the y-based terms vs the distillation term. Each task gradient is taken "
                         "through the head output o (exact: the parts sum to the plain gradient, self-checked at step 1 and every 1000 "
                         "steps); conflicting pairs are projected onto each other's normal plane in a seeded random order, then summed. "
                         "off = plain backward (research B3, add_b3.py)")
    ap.add_argument("--anchor-floor", type=float, default=0.0, metavar="K",
                    help="seqrun: realisability floor on the MSE anchor -- a row's anchor term becomes max(0, (p - clip y)^2 - K (t - clip y)^2) "
                         "with t the lookahead teacher: once the causal student fits a trained-on row better than K x the teacher that "
                         "saw its future, the anchor stops pulling it toward the label (a memorisation stop); the running Pearson and "
                         "KD terms are untouched; needs --distill and --anchor mse; 0 = off (research B3, add_b3.py)")
    ap.add_argument("--sched-cycles", type=int, default=1, metavar="N",
                    help="--ref-opt: N equal cosine cycles after the warm-up (warm restarts, SGDR); each cycle decays to the schedule's "
                         "floor and restarts at the peak; 1 = the single cycle used today (research B3, add_b3.py)")
    ap.add_argument("--distill-fam-mult", default="", metavar="M0,M1,M2,M3",
                    help="seqrun KD: multiply each lane's distillation row weight by its tick family's factor (both KD modes; unknown "
                         "family x1); '' = off (research B3, add_b3.py)")
    ap.add_argument("--wnoise", type=float, default=0.0, metavar="S",
                    help="seqrun: Gaussian noise on gru.weight_hh_l0 per chunk, std = S x mean|W|; the "
                         "additive twin of --wdrop.")
    ap.add_argument("--seq-bias", default="none", choices=["none","quiet","active"],
                    help="bias sequence sampling by book activity; our edge over the baseline is "
                         "5x smaller on quiet sequences, which the metric weights equally")
    ap.add_argument("--skip-grad", type=float, default=1.0, metavar="A",
                    help="refgru/seqrun, train-only: gradient scale of the in_proj output z on its SKIP path into the head "
                         "(post_ln(cat[y, z])). The head still reads z; only A x its gradient reaches in_proj through the skip "
                         "(0 = stop-gradient: in_proj is shaped by the recurrence alone). Train-only pre-hook on post_ln; "
                         "forward, export and inference unchanged; 1 = off (research SX, add_sx.py)")
    ap.add_argument("--gru-grad", type=float, default=1.0, metavar="A",
                    help="refgru/seqrun, train-only: gradient scale of z on its path INTO the GRU (0 = the GRU trains on a "
                         "detached z, in_proj is shaped by the head's skip alone). Train-only pre-hook on gru; forward "
                         "unchanged; 1 = off (research SX, add_sx.py)")
    ap.add_argument("--branch-epoch", type=float, default=0.0, metavar="E",
                    help="seqrun + --ref-opt + --init-from <parent>/ck_epE.pt: continue the parent's LR schedule from its "
                         "epoch-E boundary (multiplier at step+off of the parent's T steps) and run only the remaining "
                         "steps; data order from --seed. Branches of one parent are weight-averaged into one member "
                         "(research R2, add_r2_branch.py). 0 = off")
    ap.add_argument("--branch-rewarm", type=int, default=0, metavar="N",
                    help="--branch-epoch: scale the continued schedule by min(1, (step+1)/N) for the first N branch steps "
                         "(fresh optimizer moments); 0 = off (add_r2_branch.py)")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="allow cuDNN autotuning: faster, but the weights cannot be "
                         "recreated, so such a run must never be shipped")
    a = ap.parse_args()
    if a.ema_extra and a.loss != "seqrun":
        raise SystemExit("--ema-extra is implemented for --loss seqrun only (add_ems.py)")

    torch.set_num_threads(a.threads)
    # Every source of randomness is seeded and kernels are pinned to deterministic
    # algorithms, so a run can be rebuilt weight-for-weight. --nondeterministic opts out
    # for throughput experiments whose weights will never be shipped.
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    random.seed(a.seed); np.random.seed(a.seed % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    dev = torch.device(a.device)
    if a.nondeterministic:
        torch.backends.cudnn.benchmark = True
        print("WARNING: --nondeterministic -- weights from this run CANNOT be recreated; "
              "do not ship them", flush=True)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # TF32 changes matmul numerics with the kernel chosen; pin full fp32 so two runs
        # (and a rebuild on another box) see the same arithmetic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if a.tf32:                                     # add_z15.py: TF32 on, deterministic kernels kept
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("tf32: TensorFloat-32 matmuls ENABLED (cuBLAS + cuDNN; deterministic algorithms still pinned; "
                  "reproducible on the same GPU model only; no effect on CPU)", flush=True)
        # warn_only: a handful of ops have no deterministic CUDA kernel. We want that in the
        # log rather than a crash mid-run; cudnn.deterministic already covers the RNN path.
        torch.use_deterministic_algorithms(True, warn_only=True)
    if dev.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"seed {a.seed} | deterministic={not a.nondeterministic} | "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
    train = SequenceCache(ROOT/a.train_cache); ev = SequenceCache(ROOT/"cache/eval")
    if a.extra_cache:
        from data import ConcatCache, open_extra
        extra = open_extra(a.extra_cache, ROOT)
        train = ConcatCache(train, extra, a.extra_weight)
        print(f"train cache: {len(train.a)} + {len(train.b)} extra sequences from "
              f"{a.extra_cache} (extra sampled {a.extra_weight:.1f}x)")
    print(f"train {len(train)} seqs | eval {len(ev)} seqs (fixed)")
    if a.epochs > 0:
        if a.loss != "seqrun":
            raise SystemExit("--epochs is defined for --loss seqrun only")
        import math as _m
        _cps = _m.ceil(SEQUENCE_LENGTH / a.chunk)
        if a.chunk_jitter: _cps += 1                   # add_z15.py: a jittered sequence takes one extra (partial) chunk
        _n_seq = len(train)
        if a.fam_subset:
            from family import base_of as _bo
            _fs = {int(v) for v in a.fam_subset.split(",")}
            _n_seq = sum(1 for i in range(len(train)) if _bo(np.asarray(train.Y[i])) in _fs)
            print(f"family subset {sorted(_fs)}: {_n_seq} sequences", flush=True)
        _rounds = _m.ceil(a.epochs * _n_seq / a.groups)
        a.steps = _rounds * _cps
        print(f"epochs {a.epochs:g} x {_n_seq} seqs / {a.groups} lanes = {_rounds} "
              f"sequence rounds x {_cps} chunks -> {a.steps} optimizer steps", flush=True)
    if a.branch_epoch > 0 or a.branch_rewarm > 0:       # add_r2_branch.py (research R2): branch of a parent run
        if a.branch_epoch <= 0: raise SystemExit("--branch-rewarm needs --branch-epoch")
        if not (a.loss == "seqrun" and a.epochs > a.branch_epoch and a.ref_opt and a.init_from):
            raise SystemExit("--branch-epoch E: --loss seqrun, --epochs > E, --ref-opt and --init-from <parent>/ck_epE.pt")
        _r2_bad = [n for n, v in (("--distill-ramp", a.distill_ramp != "none"), ("--distill-step", a.distill_step > 0),
                                  ("--pw-schedule", a.pw_schedule != "const"), ("--anchor-end", a.anchor_end > 0),
                                  ("--self-distill", a.self_distill > 0), ("--elr", a.elr > 0),
                                  ("--sched-cycles", a.sched_cycles > 1), ("--pretrain-steps", a.pretrain_steps > 0),
                                  ("--curric", a.curric > 0)) if v]
        if _r2_bad: raise SystemExit(f"--branch-epoch: step-fraction schedules would change meaning: {_r2_bad}")
        a._r2_T = a.steps
        a._r2_off = int(round(a.steps * a.branch_epoch / a.epochs / _cps)) * _cps
        a.steps = a._r2_T - a._r2_off
        print(f"r2-branch: from {a.init_from} at epoch {a.branch_epoch:g} of {a.epochs:g}: schedule continued at step "
              f"{a._r2_off} of {a._r2_T}, running {a.steps} steps; data order from --seed {a.seed}; re-warm "
              f"{a.branch_rewarm} steps (research R2, add_r2_branch.py)", flush=True)

    kw = {k: v for k, v in (("hidden", a.hidden), ("layers", a.layers)) if v}
    BLOCKS = {"i0_p": (0, 22), "i0_v": (22, 44), "i0_dp": (44, 48), "i0_dv": (48, 52),
              "i1_p": (52, 74), "i1_v": (74, 96), "i1_dp": (96, 100), "i1_dv": (100, 104),
              "a": (104, 112)}
    rg = None
    if a.rankgauss:
        _z = np.load(ROOT/a.rankgauss)
        rg = (_z["cols"], _z["knots"], _z["z"])
        print(f"rank-gauss: appending {len(rg[0])} transformed columns "
              f"-> {112 + len(rg[0])} input features", flush=True)

    def _rg(x):
        """x: (..., 112) -> (..., 112+len(cols)) with the transformed columns appended."""
        if rg is None: return x
        cols, knots, zt = rg
        extra = np.empty(x.shape[:-1] + (len(cols),), np.float32)
        for j, c in enumerate(cols):
            extra[..., j] = np.interp(x[..., c], knots[j], zt)
        return np.concatenate([x, extra], axis=-1)

    _z18_cols = None                                   # add_z18.py: raw columns zeroed in the first input projection
    if a.drop_cols or a.keep_cols:
        if a.drop_blocks or a.rankgauss or a.family_input:
            raise SystemExit("--drop-cols/--keep-cols index the raw 112 columns; not with --drop-blocks, --rankgauss or --family-input")
        _z18_cols = _z18_parse_cols(a.drop_cols, a.keep_cols, BLOCKS)
    keep_cols = None
    if a.drop_blocks:
        drop = set()
        for nm in a.drop_blocks.split(","):
            lo, hi = BLOCKS[nm.strip()]; drop.update(range(lo, hi))
        keep_cols = np.array([c for c in range(112) if c not in drop])
        print(f"dropping blocks {a.drop_blocks}: {112} -> {len(keep_cols)} input features", flush=True)
    fam = fam_ev = None
    if a.family_input or a.family_mix or a.fam_norm or a.prefix_state == "fam0" \
            or a.lattice_head:
        fam = np.load(ROOT/"runs/family_train.npy")
        if len(fam) != len(train):
            # family_train.npy is indexed by position in ONE cache; silently reusing it
            # for a different cache would mislabel every sequence. But a ConcatCache
            # puts the PRIMARY cache first and appends the extra block, so when the
            # primary length still matches we can just label the extras the same way
            # family.py did: the warm-up classifier on rows 0..98, no targets.
            n_prim = len(getattr(train, "a", train))
            if len(fam) != n_prim:
                raise SystemExit(f"runs/family_train.npy has {len(fam)} labels but "
                                 f"{a.train_cache} has {n_prim} sequences -- rebuild it "
                                 f"for this cache (python src/family.py)")
            from family import predict as _fp0
            _z0 = np.load(ROOT/"runs/family_clf.npz")
            fam = np.concatenate([fam, np.array(
                [_fp0(train.X[i], _z0["W"], _z0["mu"], _z0["sd"])
                 for i in range(n_prim, len(train))], np.int64)])
            print(f"family labels: extended by {len(train)-n_prim} for the extra cache "
                  f"via the warm-up classifier", flush=True)
        print(f"family labels: {np.bincount(fam, minlength=4)}", flush=True)
    if a.family_input or a.fam_norm or a.lattice_head:
        # the eval cache gets its family from the SAME warm-up classifier that would
        # run at inference -- never from its targets
        from family import fit as _ffit, predict as _fpred
        _z = np.load(ROOT/"runs/family_clf.npz")
        fam_ev = np.array([_fpred(ev.X[i], _z["W"], _z["mu"], _z["sd"])
                           for i in range(len(ev))], np.int64)
        print(f"eval family mix: {np.bincount(fam_ev, minlength=4)/len(fam_ev)}", flush=True)

    if a.prefix_state != "off":
        a.detach_burn = True

    def xprep(x, idx):
        """Every input transform, in the order the training loop applies them."""
        if keep_cols is not None: x = x[..., keep_cols]
        if fnorm is not None: x = famnorm.apply(x, fam[idx], *fnorm)
        if rg is not None: x = _rg(x)
        return _fam(x, idx)

    _BASES = np.asarray(BASES, dtype=np.float64)
    _BASES_T = None
    lat_tr = lat_ev = None
    if a.lattice_head:
        lat_tr = (a.lattice_head, BASES[fam])
        lat_ev = (a.lattice_head, BASES[fam_ev])

    fnorm = None
    if a.fam_norm:
        _gmu, _gsd = train.feature_stats(64)
        fnorm = famnorm.stats(train, fam, f"train{len(train)}", _gmu, _gsd)

    def _fam(x, idx):
        """Append the 4-dim family one-hot, constant along the window."""
        if fam is None or not a.family_input: return x
        oh = np.zeros(x.shape[:-1] + (4,), np.float32)
        for k, si in enumerate(idx):
            oh[k, :, fam[si]] = 1.0
        return np.concatenate([x, oh], axis=-1)

    if a.family_mix:
        tgt = np.array([float(v) for v in a.family_mix.split(",")])
        cnt = np.bincount(fam, minlength=4).astype(np.float64)
        per = tgt / np.maximum(cnt, 1)
        seq_p_fam = per[fam]; seq_p_fam /= seq_p_fam.sum()
        seq_fam_shares = tgt
        print(f"family-mix sampling -> target shares {tgt}", flush=True)
    else:
        seq_p_fam = None
        seq_fam_shares = None
    rw = None
    if a.row_weight:
        rw = np.load(ROOT/a.row_weight, mmap_mode="r")
        print(f"row weights: {a.row_weight} shape {rw.shape}, power {a.row_weight_power}", flush=True)
    if keep_cols is not None:
        kw["n_features"] = len(keep_cols)
    if rg is not None:
        kw["n_features"] = 112 + len(rg[0])
    if a.family_input:
        kw["n_features"] = kw.get("n_features", 112) + 4
    _z43 = None                                       # --extra-in (add_z43.py, research Z43)
    if a.extra_in:
        if a.loss != "seqrun": raise SystemExit("--extra-in is implemented in the seqrun loop only")
        if a.row_drop > 0: raise SystemExit("--extra-in with --row-drop: dropped rows would misalign the decoded channels; not implemented")
        if getattr(a, "swap_frac", 0) > 0:
            raise SystemExit("--extra-in with --swap-frac (add_z35.py): swapped lanes would need the i0/i1 channels swapped; not implemented")
        if a.extra_cache and ("," in a.extra_cache or ":" in a.extra_cache):
            raise SystemExit("--extra-in supports one plain --extra-cache (no list, no slice)")
        _z4bad = [n for n, v in (("--drop-blocks", keep_cols is not None), ("--drop-cols/--keep-cols", _z18_cols is not None),
                                 ("--rankgauss", rg is not None), ("--family-input", a.family_input), ("--fam-norm", a.fam_norm),
                                 ("whiten preset", a.preset.startswith("whiten")), ("--no-norm", getattr(a, "no_norm", False)),
                                 ("--stats-cache", bool(getattr(a, "stats_cache", ""))), ("--seq-affine", getattr(a, "seq_affine", 0) > 0),
                                 ("--prefix-state", getattr(a, "prefix_state", "off") != "off"), ("--hint", bool(getattr(a, "hint", ""))),
                                 ("--split-towers", bool(getattr(a, "split_towers", False))), ("--diff", bool(getattr(a, "diff", False))),
                                 ("--ewma", bool(getattr(a, "ewma", ""))), ("--nlfeat", bool(getattr(a, "nlfeat", ""))),
                                 ("--ofi", bool(getattr(a, "ofi", False))), ("--bookconv", bool(getattr(a, "bookconv", 0))),
                                 ("--tradeconv", bool(getattr(a, "tradeconv", 0))), ("--instshare", bool(getattr(a, "instshare", 0))),   # add_s1.py
                                 ("--densein", bool(getattr(a, "densein", 0))),   # add_s1b.py
                                 ("--xinst", bool(getattr(a, "xinst", 0))), ("--bookconv-act", bool(getattr(a, "bookconv_act", ""))),   # add_struct.py
                                 ("--ple-bins", bool(getattr(a, "ple_bins", 0))),   # add_h4_ple.py
                                 ("--adanorm", getattr(a, "adanorm", 0) > 0)) if v]
        if _z4bad:
            raise SystemExit(f"--extra-in with {', '.join(_z4bad)}: those index, rescale or expand the raw 112 columns; not implemented")
        _z4d = Path(a.extra_in) if Path(a.extra_in).is_absolute() else ROOT / a.extra_in
        _z4has = hasattr(train, "a") and hasattr(train, "b")
        _z4parts = [(Path(a.train_cache).name, train.a if _z4has else train), ("eval", ev)]
        if _z4has:
            _z4parts.insert(1, (Path(a.extra_cache).name, train.b))
        _z4arr, _z4chs, _z4fill = {}, [], None
        for _z4nm, _z4c in _z4parts:
            _z4f, _z4mf = _z4d / f"{_z4nm}.npy", _z4d / f"{_z4nm}_meta.json"
            if not (_z4f.exists() and _z4mf.exists()):
                raise SystemExit(f"--extra-in: {_z4f} or {_z4mf.name} is missing (build: feature dump builder, not included in this repo)")
            _z4m = json.load(open(_z4mf)); _z4k = int(_z4m.get("K", 0))
            _z4aa = np.load(_z4f, mmap_mode="r")
            if _z4k < 1 or _z4aa.dtype != np.float16 or _z4aa.shape != (len(_z4c), SEQUENCE_LENGTH, _z4k):
                raise SystemExit(f"--extra-in: {_z4f} is {_z4aa.dtype} {_z4aa.shape} (meta K = {_z4k}), expected float16 "
                                 f"({len(_z4c)}, {SEQUENCE_LENGTH}, K)")
            if _z4m.get("seq_ix") != list(_z4c.meta["seq_ix"]):
                raise SystemExit(f"--extra-in: {_z4mf} seq_ix differs from the {_z4nm} cache manifest (wrong cache or order)")
            _z4chs.append([c.get("name") for c in _z4m.get("channels", [])])
            _z4fl = float(_z4m.get("fill", {}).get("nan", 0.0))
            if _z4fill is not None and _z4fl != _z4fill: raise SystemExit("--extra-in: the dumps declare different fill values")
            _z4fill = _z4fl
            _z4arr[_z4nm] = _z4aa
        if any(c != _z4chs[0] for c in _z4chs) or len(_z4chs[0]) != _z4aa.shape[-1]:
            raise SystemExit(f"--extra-in: channel lists differ between the dumps or from K: {_z4chs}")
        _z4K = len(_z4chs[0])
        _z4p = _z4arr[Path(a.train_cache).name]; _z4e = _z4arr[Path(a.extra_cache).name] if _z4has else None
        _z4np = len(_z4p)
        _z43 = (_z4p, _z4e, _z4K, _z4fill)
        _Z43_VIEWS[id(train)] = _Z43View(train, (lambda i, _p=_z4p, _e=_z4e, _n=_z4np: _p[i] if i < _n else _e[i - _n]), _z4fill)
        _Z43_VIEWS[id(ev)] = _Z43View(ev, (lambda i, _ae=_z4arr["eval"]: _ae[i]), _z4fill)
        kw["n_features"] = 112 + _z4K
        print(f"extra-in: {_z4K} decoded input channels {_z4chs[0]} from {_z4d} ({', '.join(n for n, _ in _z4parts)}); NaN -> "
              f"{_z4fill}; input 112 -> {112 + _z4K} columns (research Z43)", flush=True)
    aux_off = [int(x) for x in a.aux_horizons.split(",")] if a.aux_horizons else []
    ssl_w = None
    if a.ssl_aux > 0:
        from ssl_aux import aux_targets as _auxt
        ssl_w = np.load(ROOT/"runs/aux_direction.npy")
        print(f"self-supervised aux heads at weight {a.ssl_aux} "
              f"(event / sign / wait), dropped at export", flush=True)
    n_out = 2 if (a.two_head or a.common_head) else 1
    dens_h = [int(v) for v in a.dens_horizons.split(',') if v] if a.densaux > 0 else []
    if a.dens_mid:                                    # add_z37.py
        if not dens_h: raise SystemExit("--dens-mid replaces the --densaux targets; it needs --densaux > 0 (a silent no-op otherwise)")
        if a.loss != "seqrun": raise SystemExit("--dens-mid is implemented in the seqrun loop only (where --densaux lives)")
        if a.row_drop > 0: raise SystemExit("--dens-mid with --row-drop: a dropped unknown row would hide a decode gap; not implemented")
        if a.extra_cache and ("," in a.extra_cache or ":" in a.extra_cache):
            raise SystemExit("--dens-mid supports one plain --extra-cache (no list, no slice)")
        if getattr(a, "swap_frac", 0) > 0:
            raise SystemExit("--dens-mid with --swap-frac (add_z35.py): swapped lanes would need the mid channels swapped; not implemented")
    if a.hlg != 0 or a.mh != 0:                       # add_z59.py
        if a.hlg < 0 or a.mh < 0 or a.mh == 1: raise SystemExit("--hlg must be > 0 and --mh >= 2 (0 = off)")
        if a.hlg > 0 and a.mh > 0: raise SystemExit("--hlg with --mh is not implemented")
        if not a.preset.startswith("refgru") or a.loss != "seqrun" or not a.two_head:
            raise SystemExit("--hlg / --mh: --preset refgru --loss seqrun --two-head only (where their loss lives)")
        if a.grid_head > 0 or a.lattice_head or a.common_head or a.ncl > 0 or a.split_towers or a.bidir or a.dualmode:
            raise SystemExit("--hlg / --mh: not with --grid-head / --lattice-head / --common-head / --ncl / split towers / bidir / dualmode")
        if a.hlg > 0 and (a.hlg_k < 2 or not (a.hlg_sigma > 0)): raise SystemExit("--hlg: --hlg-k >= 2 and --hlg-sigma > 0")
        if a.mh > 0 and (a.init_from or a.anchor != "mse" or a.cosine_loss or a.muon_scope == "nohead" or not (a.mh_w >= 0)):
            raise SystemExit("--mh: not with --init-from / --anchor huber|l1 / --cosine-loss / --muon-scope nohead; --mh-w >= 0")
        print(f"z59: " + (f"HL-Gauss aux head, {a.hlg_k} bins over [-2.2, 2.2] (width {4.4 / a.hlg_k:.4f}), sigma {a.hlg_sigma} "
                          f"bin widths, weight {a.hlg}; hlg_head dropped at export" if a.hlg > 0 else
                          f"TabM readout, {a.mh} sub-heads, output = mean, + {a.mh_w} x mean per-head (pearson + anchor) loss; "
                          f"folded into the single head at save time") + " (research Z59)", flush=True)
    elif a.hlg_k != 41 or a.hlg_sigma != 0.75 or a.mh_w != 1.0:
        raise SystemExit("--hlg-k / --hlg-sigma need --hlg W > 0, --mh-w needs --mh K (a silent no-op otherwise)")
    if getattr(a, "dens_cross_chunk", False) and not a.dens_mid:   # add_z55.py
        raise SystemExit("--dens-cross-chunk extends the EXACT --dens-mid targets past the chunk; it needs --dens-mid (a silent no-op otherwise)")
    if any(h <= 0 for h in dens_h):                    # add_sx.py (research SX): backward exact-mid targets
        if any(h == 0 for h in dens_h): raise SystemExit("--dens-horizons: 0 is not a horizon")
        if not a.dens_mid: raise SystemExit("negative --dens-horizons (backward exact-mid targets mid[t] - mid[t-|h|]) need --dens-mid")
        if getattr(a, "dens_cross_chunk", False): raise SystemExit("negative --dens-horizons with --dens-cross-chunk: not implemented")
        print(f"sx: backward exact-mid targets at {[h for h in dens_h if h < 0]} (row t predicts mid[t] - mid[t-|h|]; same head-row "
              f"layout, dropped at export) beside the forward horizons {[h for h in dens_h if h > 0]} (research SX, add_sx.py)", flush=True)
    if a.skip_grad != 1.0 or a.gru_grad != 1.0:        # add_sx.py (research SX): train-only gradient gates on z
        if not (0.0 <= a.skip_grad <= 1.0 and 0.0 <= a.gru_grad <= 1.0): raise SystemExit("--skip-grad / --gru-grad must be in [0, 1]")
        if (not a.preset.startswith("refgru") or a.loss != "seqrun" or a.split_towers or a.bidir or a.dualmode or a.fastw
                or a.cell != "gru" or a.ncl > 0):
            raise SystemExit("--skip-grad / --gru-grad: plain single-tower causal refgru GRU under --loss seqrun only (no split towers, "
                             "bidir, dualmode, fastw, lstm or ncl)")
    if a.dens_fam_mult:                               # add_z38.py
        if not a.dens_mid:
            raise SystemExit("--dens-fam-mult rescales the EXACT --dens-mid targets per family; it needs --dens-mid (a silent no-op otherwise)")
        _z38g = [g.split(",") for g in a.dens_fam_mult.split("/")]
        if len(_z38g) not in (1, 2) or any(len(g) != 4 for g in _z38g):
            raise SystemExit("--dens-fam-mult: give M0,M1,M2,M3 or M0,M1,M2,M3/N0,N1,N2,N3 (i0 / i1 multipliers for families 0-3)")
        try:
            _z38g = [[float(v) for v in g] for g in _z38g]
        except ValueError:
            raise SystemExit(f"--dens-fam-mult: non-numeric multiplier in {a.dens_fam_mult!r}")
        if any(not (v >= 0.0) for g in _z38g for v in g) or any(max(g) <= 0.0 for g in _z38g):
            raise SystemExit("--dens-fam-mult: multipliers must be >= 0, with at least one > 0 per instrument")
    if a.paux:                                        # add_z39.py
        if not (a.paux_weight > 0.0):
            raise SystemExit("--paux needs --paux-weight W > 0 (a silent no-op otherwise)")
        if a.loss != "seqrun": raise SystemExit("--paux is implemented in the seqrun loop only (where --densaux lives)")
        if a.row_drop > 0: raise SystemExit("--paux with --row-drop: dropped rows would misalign the per-row targets; not implemented")
        if a.extra_cache and ("," in a.extra_cache or ":" in a.extra_cache):
            raise SystemExit("--paux supports one plain --extra-cache (no list, no slice)")
        if getattr(a, "swap_frac", 0) > 0:
            raise SystemExit("--paux with --swap-frac (add_z35.py): swapped lanes would need i1-based targets; not implemented")
        if a.lattice_head or a.grid_head > 0:
            raise SystemExit("--paux with --lattice-head / --grid-head: those heads read every output channel; not implemented")
        if a.paux_fam_mult:
            try:
                _z9g0 = [[float(v) for v in g.split(",")] for g in a.paux_fam_mult.split("/")]
            except ValueError:
                raise SystemExit(f"--paux-fam-mult: non-numeric multiplier in {a.paux_fam_mult!r}")
            if any(len(g) != 4 for g in _z9g0) or any(not (v >= 0.0) for g in _z9g0 for v in g) or max(max(g) for g in _z9g0) <= 0.0:
                raise SystemExit("--paux-fam-mult: give M0,M1,M2,M3 (all channels) or one such group per channel joined by '/'; "
                                 "multipliers >= 0 and at least one > 0 (an all-zero group switches that channel off)")
    elif a.paux_weight != 0.0 or a.paux_fam_mult:
        raise SystemExit("--paux-weight / --paux-fam-mult need --paux DIR (a silent no-op otherwise)")
    if dens_h:
        print(f'dense aux: forward mid change at {dens_h} for both instruments, weight {a.densaux}', flush=True)

    if a.lattice_head:
        n_out = 2*a.lattice_head + 1
        print(f"lattice head: {n_out} classes over k in "
              f"[{-a.lattice_head}, {a.lattice_head}], w-tilted readout", flush=True)
    if ssl_w is not None: n_out += 3
    n_out += 2 * len(dens_h)
    if a.levelaux > 0:
        n_out += 4                       # level and log-scale, per instrument
        print(f"level aux (hint distillation on the per-sequence latent) at weight {a.levelaux}, "
              f"4 heads, dropped at export", flush=True)
    _auxh_base, _auxh_c = None, 0
    if aux_off and a.loss == "seqrun":
        # seqrun port: APPEND one channel per target per horizon after every other head, so the
        # deployed channels (:2) and the densaux/levelaux layout are unchanged and the heads are
        # dropped at export like --densaux. See offline/patches/add_auxh_seqrun.py.
        _auxh_base, _auxh_c = n_out, (2 if a.two_head else 1)
        n_out += _auxh_c * len(aux_off)
        print(f"auxiliary horizons {aux_off} at weight {a.aux_weight} (seqrun, RunningWP per "
              f"horizon) -> channels {_auxh_base}.. of {n_out}, dropped at export", flush=True)
    elif aux_off:
        n_out = 1 + len(aux_off)
        print(f"auxiliary horizons {aux_off} at weight {a.aux_weight} "
              f"-> {n_out} training heads, 1 deployed", flush=True)
    _paux_base, _paux_k = None, 0                     # --paux (add_z39.py, research Z39)
    if a.paux:
        _z9d0 = Path(a.paux) if Path(a.paux).is_absolute() else ROOT / a.paux
        _z9m0 = _z9d0 / f"{Path(a.train_cache).name}_meta.json"
        if not _z9m0.exists():
            raise SystemExit(f"--paux: {_z9m0} is missing (build: rule-target dump builder, not included in this repo)")
        _paux_k = int(json.load(open(_z9m0))["K"])
        if _paux_k < 1: raise SystemExit(f"--paux: {_z9m0} declares K = {_paux_k}")
        if a.paux_fam_mult and len(a.paux_fam_mult.split("/")) not in (1, _paux_k):
            raise SystemExit(f"--paux-fam-mult: give 1 group (all channels) or {_paux_k} groups (one per channel)")
        _paux_base = n_out
        n_out += _paux_k
        print(f"paux: {_paux_k} exact auxiliary targets from {_z9d0} -> channels {_paux_base}..{n_out - 1} of {n_out} (appended "
              f"after every other head, dropped at export); masked per-batch-standardised MSE, weight {a.paux_weight} per "
              f"channel (research Z39)", flush=True)
    if a.preset.startswith("intgru"): kw["n_int"] = a.n_int
    if a.proj: kw["proj"] = a.proj
    if a.preset.startswith("refgru"):
        kw["cell"] = a.cell; kw["diff"] = a.diff; kw["posfeat"] = a.posfeat; kw["split"] = a.split_towers
        if a.bidir: kw["bidir"] = True
        if a.dualmode: kw["dualmode"] = int(a.dualmode)
        if a.hint:
            kw["hint"] = int(torch.load(ROOT / a.hint, map_location="cpu", weights_only=False)["cfg"]["hidden"])
        if a.aux_next > 0: kw["aux_next"] = 1
        if a.mask_aux > 0: kw["mask_aux"] = 1
        if a.hetero > 0: kw["hetero"] = 1          # train-only log-variance head (add_losses2.py)
        if a.resnorm_w > 0 or a.resnorm_ed > 0:     # train-only log-sigma head (research H12, add_h12_resnorm.py)
            kw["resnorm"] = 1; kw["resnorm_sd0"] = float(a.resnorm_sd0)
        if a.in_drop > 0: kw["in_drop"] = a.in_drop
        if a.grid_head > 0: kw["grid"] = int(a.grid_head)
        if a.hlg > 0: kw["hlg_bins"] = int(a.hlg_k)      # add_z59.py (train-only head)
        if a.mh > 0: kw["mh_heads"] = int(a.mh)          # add_z59.py (folded at save time)
        if a.out_drop > 0: kw["out_drop"] = a.out_drop
        kw["quad"] = bool(a.quad)
        kw["ewma"] = tuple(float(v) for v in a.ewma.split(",") if v)
        if a.fastw:
            kw["fastw"] = int(a.fastw); kw["fw_groups"] = int(a.fastw_groups)
            kw["fw_to_gru"] = bool(a.fastw_gru)
            _hl = [float(v) for v in a.fastw_hl.split(",")]
            kw["fw_hl_min"], kw["fw_hl_max"] = _hl[0], _hl[-1]
        kw["nlfeat"] = tuple(float(v) for v in a.nlfeat.split(",") if v)
        if a.dropout is not None: kw["dropout"] = a.dropout
        if a.head_hidden: kw["head_hidden"] = a.head_hidden
        if a.clamp != 10.0: kw["clamp"] = float(a.clamp)
        if a.bookconv: kw["bookconv"] = int(a.bookconv)
        if a.bookconv and a.bookconv_taps != 3: kw["bookconv_taps"] = int(a.bookconv_taps)      # add_s1.py (research S1)
        if a.bookconv and a.bookconv_unshare: kw["bookconv_unshare"] = True
        if a.bookconv and a.bookconv_split: kw["bookconv_split"] = True                          # add_h7_bcsplit.py (research H7)
        if a.bookconv and a.bookconv_side: kw["bookconv_side"] = True
        if a.tradeconv: kw["tradeconv"] = int(a.tradeconv)
        if a.instshare: kw["instshare"] = int(a.instshare)
        if a.inproj_init_mult != 1.0: kw["inproj_init_mult"] = float(a.inproj_init_mult)   # add_s1b.py (research S1 controls)
        if a.densein: kw["densein"] = int(a.densein)
        if a.xinst: kw["xinst"] = int(a.xinst)                                              # add_struct.py (research ST)
        if a.bookconv_act: kw["bookconv_act"] = str(a.bookconv_act)
        if a.hh_rank: kw["hh_rank"] = int(a.hh_rank)
        if a.ple_bins: kw["ple_bins"] = int(a.ple_bins); kw["ple_ch"] = int(a.ple_ch)      # add_h4_ple.py (research H4)
        if a.ofi: kw["ofi"] = True
        if a.learn_h0: kw["learn_h0"] = True
        if a.adanorm > 0: kw["adanorm"] = float(a.adanorm)
        if a.adanorm_cols:                                  # add_z48.py (research Z48): selective causal normalisation
            if a.adanorm <= 0: raise SystemExit("--adanorm-cols needs --adanorm HALFLIFE > 0")
            kw["adanorm_cols"] = str(a.adanorm_cols)
        if a.proj_layers != 1: kw["proj_layers"] = int(a.proj_layers)
        if a.proj_gated: kw["proj_gated"] = True
    core = build(a.preset, n_out=n_out, **kw).to(dev)
    if a.skip_grad != 1.0 or a.gru_grad != 1.0:        # add_sx.py (research SX): train-only gradient gates on z (forward unchanged)
        _sxH = int(core.hidden)
        if a.skip_grad != 1.0:
            def _sx_skip_hook(mod, args, _A=float(a.skip_grad), _H=_sxH):
                if not mod.training: return None
                t = args[0]; zs = t[..., _H:]
                return (torch.cat([t[..., :_H], zs.detach() if _A == 0.0 else _A * zs + (1.0 - _A) * zs.detach()], -1),) + tuple(args[1:])
            core.post_ln.register_forward_pre_hook(_sx_skip_hook)
        if a.gru_grad != 1.0:
            def _sx_gru_hook(mod, args, _A=float(a.gru_grad)):
                if not mod.training: return None
                z = args[0]
                return (z.detach() if _A == 0.0 else _A * z + (1.0 - _A) * z.detach(),) + tuple(args[1:])
            core.gru.register_forward_pre_hook(_sx_gru_hook)
        print(f"sx: gradient gates on the in_proj output z: skip (z -> head) x {a.skip_grad:g}, GRU input x {a.gru_grad:g} "
              f"(train-only pre-hooks on post_ln / gru; forward, export and inference unchanged; research SX, add_sx.py)", flush=True)
    if (a.gate_init != "default" or a.orth_init) and hasattr(core, "gru"):
        import gateinit as _gi
        if a.gate_init != "default":
            print(_gi.apply_gate_init(core.gru, a.gate_init, t_max=SEQUENCE_LENGTH), flush=True)
        if a.orth_init:
            print(_gi.apply_orth_init(core.gru), flush=True)
    if a.slow_units > 0:
        # add_z25.py (research Z25, Z23's I2): a SLOW LANE inside the existing GRU. The last K units of layer 0 get their update
        # (retention) gate bias set to log(t), t log-spaced over [--slow-tmin, --slow-tmax] rows, so at zero input they retain
        # sigmoid(log t) = t/(1+t) of their state per row (time constant ~t rows). The shipping model has no unit with an
        # effective time constant above ~900 rows (z25_gates.py); chrono on EVERY unit was -0.006/-0.010 (Z13 SS1.3). Bias
        # values only: no new parameters, no RNG draw (every other weight keeps the control's init), graph/export unchanged.
        # Initialised, not clamped: b_z sits in the no-decay group under --ref-opt and the optimiser may move it.
        _g = getattr(core, "gru", None)
        if not isinstance(_g, torch.nn.GRU):
            raise SystemExit("--slow-units: the recurrence must be a plain nn.GRU (refgru without split towers / LSTM cell)")
        if a.gate_init != "default":
            raise SystemExit("--slow-units and --gate-init both set the update-gate bias; use one")
        if a.init_from:
            raise SystemExit("--slow-units with --init-from: the loaded state would overwrite the slow-lane init")
        _H = _g.hidden_size
        if not (0 < a.slow_units <= _H):
            raise SystemExit(f"--slow-units must be in 1..{_H} (the hidden width)")
        if not (1.0 < a.slow_tmin <= a.slow_tmax):
            raise SystemExit("--slow-tmin/--slow-tmax must satisfy 1 < tmin <= tmax (rows)")
        _tj = torch.as_tensor(np.geomspace(a.slow_tmin, a.slow_tmax, a.slow_units), dtype=torch.float32)
        _sl = slice(2 * _H - a.slow_units, 2 * _H)          # PyTorch packs GRU biases as [r, z, n]; z is the middle third
        with torch.no_grad():
            _g.bias_hh_l0[_sl].copy_(torch.log(_tj).to(_g.bias_hh_l0))
            _g.bias_ih_l0[_sl].zero_()                        # the two biases add; use one (gateinit.py's convention)
        _ret = torch.sigmoid(torch.log(_tj))
        print(f"z25-slow: {a.slow_units} of {_H} layer-0 GRU units (indices {_H - a.slow_units}..{_H - 1}) start as a slow lane: "
              f"update-gate bias log(t), t log-spaced {a.slow_tmin:g}..{a.slow_tmax:g} rows -> zero-input retention "
              f"{float(_ret.min()):.5f}..{float(_ret.max()):.6f}; b_iz zeroed; initialised, not clamped; graph/export unchanged", flush=True)
    if a.common_head:
        core.common_head = True
    if a.init_from:
        _ck = torch.load(ROOT/a.init_from, map_location="cpu", weights_only=False)
        _sd = dict(_ck["state"])
        # refgru's head is an MLP (head.0.*, head.3.*), NOT a bare Linear, so "head.weight"
        # does not exist and every --init-from run against a refgru died with KeyError before
        # training a single step. Find the LAST linear generically, and only touch it when
        # --common-head actually asks for the 1 -> 2 widening.
        _hk = sorted(k for k in _sd if k.startswith("head") and k.endswith(".weight"))
        if not _hk:
            raise SystemExit(f"--init-from {a.init_from}: no head.*weight in the checkpoint")
        _wk = _hk[-1]; _bk = _wk[: -len("weight")] + "bias"
        _w = _sd[_wk]
        if a.common_head and _w.shape[0] == 1:      # widen 1 -> 2, c row zeroed
            _b = _sd[_bk]
            nw = torch.zeros(2, _w.shape[1], dtype=_w.dtype); nw[0] = _w[0]
            nb = torch.zeros(2, dtype=_b.dtype);               nb[0] = _b[0]
            _sd[_wk], _sd[_bk] = nw, nb
        core.load_state_dict(_sd)
        print(f"continuing from {a.init_from} (head {_wk} {tuple(_sd[_wk].shape)})", flush=True)
    elif a.common_head:
        with torch.no_grad():                       # fresh run: zero-init c
            _last = core.head[-1] if isinstance(core.head, torch.nn.Sequential) else core.head
            _last.weight[1].zero_(); _last.bias[1].zero_()
    _z14 = _z14_setup(core, a, dev)                 # add_z14.py: factor parametrisation / learned input gain (None when off)
    if a.preset.startswith("whiten"):
        from whiten import fit_whitener
        wpath = ROOT / a.train_cache / "whitener.npz"
        if wpath.exists():
            z = np.load(wpath); W, bb = z["W"], z["b"]
        else:
            print("fitting PCA whitener (streamed covariance)...", flush=True)
            W, bb = fit_whitener(train)
            np.savez(wpath, W=W, b=bb)
        core.set_whitener(W, bb)
        print(f"input: PCA-WHITENED (gain {np.abs(W).max():.1f}x on the smallest direction)")
    elif a.fam_norm:
        core.set_norm(np.zeros(core.mu.shape[0], np.float32),
                      np.ones(core.sd.shape[0], np.float32))
        print("normalisation: PER FAMILY (global buffers set to identity)")
    elif a.no_norm:
        print("normalisation: OFF (raw features; organisers already standardised globally)")
    else:
        if a.stats_cache:
            # normalisation statistics from another cache (e.g. valid_retrain: the test period
            # resembles valid; holdout means shift up to 0.7 train-sd on i1 volume columns)
            from data import SequenceCache as _SC
            mu, sd = _SC(ROOT / a.stats_cache).feature_stats(a.stats_n)
            print(f"normalisation statistics from {a.stats_cache}", flush=True)
        else:
            mu, sd = train.feature_stats(a.stats_n)
        if keep_cols is not None: mu, sd = mu[keep_cols], sd[keep_cols]
        if rg is not None:
            cols = rg[0]
            mu = np.concatenate([mu, np.zeros(len(cols), mu.dtype)])
            sd = np.concatenate([sd, np.ones(len(cols), sd.dtype)])
        if a.family_input:
            mu = np.concatenate([mu, np.zeros(4, mu.dtype)])
            sd = np.concatenate([sd, np.ones(4, sd.dtype)])
        if _z43 is not None:
            # --extra-in (add_z43.py): the K channels' mean / std over the same first --stats-n PRIMARY training sequences as
            # data.SequenceCache.feature_stats (float64 sums, NaN filled first, same formula), appended after the 112
            _z4s1 = np.zeros(_z43[2], np.float64); _z4s2 = np.zeros(_z43[2], np.float64); _z4c = 0
            for _z4i in range(min(a.stats_n, len(_z43[0]))):
                _z4x = np.asarray(_z43[0][_z4i], np.float64)
                _z4x = np.where(np.isnan(_z4x), _z43[3], _z4x)
                _z4s1 += _z4x.sum(0); _z4s2 += (_z4x * _z4x).sum(0); _z4c += len(_z4x)
            _z4mu = _z4s1 / _z4c; _z4sd = np.sqrt(np.maximum(_z4s2 / _z4c - _z4mu * _z4mu, 1e-12))
            mu = np.concatenate([mu, _z4mu.astype(np.float32)]); sd = np.concatenate([sd, _z4sd.astype(np.float32)])
            print(f"extra-in: channel mean {np.round(_z4mu, 3).tolist()} sd {np.round(_z4sd, 3).tolist()} "
                  f"(first {min(a.stats_n, len(_z43[0]))} primary sequences)", flush=True)
        core.set_norm(mu, sd)
        print(f"normalisation: per-column from train, max amplification {float(1/sd.min()):.1f}x")
    if getattr(a, "ple_bins", 0) or getattr(a, "ple_ch", 8) != 8:      # add_h4_ple.py (research H4): piecewise-linear embedding of the p codes
        if not a.ple_bins: raise SystemExit("--ple-ch needs --ple-bins B > 0 (a silent no-op otherwise)")
        if a.ple_bins < 2: raise SystemExit("--ple-bins B must be >= 2 (0 = off)")
        if a.ple_ch < 1: raise SystemExit("--ple-ch C must be >= 1")
        if not a.preset.startswith("refgru"): raise SystemExit("--ple-bins is a refgru input structure")
        _h4bad = [n for n, v in (("--extra-in", bool(getattr(a, "extra_in", ""))), ("--drop-blocks", bool(getattr(a, "drop_blocks", None))),
                                 ("--drop-cols/--keep-cols", bool(getattr(a, "drop_cols", None)) or bool(getattr(a, "keep_cols", None))),
                                 ("--rankgauss", bool(getattr(a, "rankgauss", None))), ("--family-input", bool(getattr(a, "family_input", False))),
                                 ("--fam-norm", bool(getattr(a, "fam_norm", False))), ("whiten preset", a.preset.startswith("whiten")),
                                 ("--swap-frac", getattr(a, "swap_frac", 0) > 0), ("--adanorm", getattr(a, "adanorm", 0) > 0),
                                 ("--split-towers", bool(getattr(a, "split_towers", False))), ("--seq-affine", getattr(a, "seq_affine", 0) > 0)) if v]
        if _h4bad:
            raise SystemExit(f"--ple-bins with {', '.join(_h4bad)}: those re-index, rescale or renormalise the raw columns the bin edges "
                             "are computed on; not implemented")
        # Bin edges: quantiles of the standardised, clamped column over the same first --stats-n PRIMARY sequences that give mu / sd
        # (data.SequenceCache.feature_stats; --stats-cache if given), NaN -> the column mean as the model's forward and the native
        # kernel treat it. Fixed for the run and saved in the checkpoint (buffer ple_edges), so the export reads them like mu / sd.
        _h4src = getattr(train, "a", train)
        if getattr(a, "stats_cache", ""):
            from data import SequenceCache as _H4SC
            _h4src = _H4SC(ROOT / a.stats_cache)
        _h4n = min(int(a.stats_n), len(_h4src))
        _h4cols = list(range(0, 22)) + list(range(52, 74))
        _h4mu = core.mu.detach().cpu().numpy().astype(np.float32)[_h4cols]
        _h4sd = core.sd.detach().cpu().numpy().astype(np.float32)[_h4cols]
        _h4z = []
        for _h4i in range(_h4n):
            _h4x = np.asarray(_h4src.X[_h4i], np.float32)[:, _h4cols]
            _h4x = np.where(np.isnan(_h4x), _h4mu, _h4x)
            _h4z.append(np.clip((_h4x - _h4mu) / _h4sd, -float(core.clamp), float(core.clamp)))
        _h4z = np.concatenate(_h4z, 0).astype(np.float64)                      # (rows, 44): exactly what _inputs sees
        _h4q = np.quantile(_h4z, np.linspace(0.0, 1.0, a.ple_bins + 1), axis=0).T   # (44, B+1): b_0 = min ... b_B = max
        _h4dup, _h4maxrun, _h4ncol = 0, 1, 0
        for _h4j in range(44):
            _h4e = _h4q[_h4j]; _h4tol = 1e-6 * max(float(_h4e[-1] - _h4e[0]), 1e-6); _h4i = 0; _h4hit = False
            while _h4i < len(_h4e):
                _h4k = _h4i
                while _h4k + 1 < len(_h4e) and _h4e[_h4k + 1] <= _h4e[_h4i] + _h4tol: _h4k += 1    # run of equal quantiles = an atom
                if _h4k > _h4i:
                    _h4run = _h4k - _h4i + 1; _h4dup += _h4k - _h4i; _h4maxrun = max(_h4maxrun, _h4run); _h4hit = True
                    if _h4k + 1 < len(_h4e): _h4gap = (_h4e[_h4k + 1] - _h4e[_h4i]) / _h4run        # spread up to the next distinct edge
                    else: _h4gap = max((_h4e[_h4i] - (_h4e[_h4i - 1] if _h4i > 0 else _h4e[_h4i] - 1e-3 * _h4run)) / _h4run, 1e-6)
                    for _h4m in range(1, _h4run): _h4e[_h4i + _h4m] = _h4e[_h4i] + _h4gap * _h4m
                _h4i = _h4k + 1
            _h4ncol += int(_h4hit)
        core.set_ple_edges(_h4q)
        print(f"ple: {a.ple_bins}-bin piecewise-linear embedding of the 44 price-code columns (0-21, 52-73) -> {a.ple_ch} channels each "
              f"through one shared {a.ple_bins}x{a.ple_ch} map = {44 * a.ple_ch} extra inputs to in_proj.0 (n_in {core.n_in}); edges = "
              f"quantiles of the standardised clamped columns over {_h4n} sequences of "
              f"{a.stats_cache or Path(a.train_cache).name} ({_h4z.shape[0]:,} rows): {_h4ncol}/44 columns had equal quantiles "
              f"({_h4dup} edges spread, longest run {_h4maxrun}); edge range [{_h4q.min():+.3f}, {_h4q.max():+.3f}]; causal (row t "
              f"only); NOT foldable at export (research H4, add_h4_ple.py)", flush=True)
    print(f"model {a.preset}: {core.n_params():,} params, est {core.est_us_per_row():.1f} us/row "
          f"({core.est_us_per_row()/96.1:.0%} of budget)")

    # Decay/timescale parameters live on a different scale from weight matrices:
    # a_logit sets a per-channel forget rate, so the same step size that suits a
    # matmul weight moves it across orders of magnitude of effective memory.
    DECAY = ("a_logit", "log_decay", "lam_logit", "dt_log")
    _betas = {}
    if a.betas:
        _betas["betas"] = tuple(float(v) for v in a.betas.split(","))
        print(f"AdamW betas {_betas['betas']}, weight decay {a.wd}", flush=True)
    decay_p = [q for n, q in core.named_parameters() if any(k in n for k in DECAY)]
    other_p = [q for n, q in core.named_parameters() if not any(k in n for k in DECAY)]
    if decay_p and a.decay_lr_mult != 1.0:
        groups = [{"params": other_p, "lr": a.lr, "weight_decay": a.wd},
                  {"params": decay_p, "lr": a.lr * a.decay_lr_mult, "weight_decay": 0.0}]
        print(f"param groups: {len(other_p)} tensors at lr {a.lr:.1e}, "
              f"{len(decay_p)} DECAY tensors at lr {a.lr*a.decay_lr_mult:.1e} (wd 0)")
        opt = torch.optim.AdamW(groups, **_betas)
    else:
        opt = torch.optim.AdamW(core.parameters(), lr=a.lr, weight_decay=a.wd, **_betas)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in opt.param_groups],
                                                total_steps=a.steps,
                                                pct_start=0.15, final_div_factor=30)
    if a.opt in ("muon", "lion", "soap") and not a.ref_opt:
        raise SystemExit(f"--opt {a.opt} is implemented under --ref-opt only (it would silently run AdamW)")
    if a.muon_adam > 0 and a.opt != "muon":
        raise SystemExit(f"--muon-adam is implemented for --opt muon only (it would be a silent no-op under --opt {a.opt})")
    if a.vaug != 0 or a.vaug_gain != 0 or a.vaug_mode != "percol" or a.vaug_fams or a.vaug_cols != "22:44,74:96":   # add_g1.py
        if a.vaug < 0 or a.vaug_gain < 0: raise SystemExit("--vaug T and --vaug-gain S must be >= 0")
        if a.vaug <= 0 and a.vaug_gain <= 0:
            raise SystemExit("--vaug-mode / --vaug-fams / --vaug-cols need --vaug T > 0 or --vaug-gain S > 0 (a silent no-op otherwise)")
        if a.loss != "seqrun": raise SystemExit("--vaug is implemented in the seqrun loop only (a silent no-op otherwise)")
        if a.bidir: raise SystemExit("--vaug is a student-side augmentation; not with --bidir")
        _g1bad = [n for n, v in (("--extra-in", bool(getattr(a, "extra_in", ""))), ("--rankgauss", bool(a.rankgauss)),
                                 ("--family-input", bool(a.family_input)), ("--fam-norm", bool(a.fam_norm)),
                                 ("whiten preset", a.preset.startswith("whiten")), ("--no-norm", bool(getattr(a, "no_norm", False))),
                                 ("--swap-frac", getattr(a, "swap_frac", 0) > 0)) if v]
        if _g1bad:
            raise SystemExit(f"--vaug with {', '.join(_g1bad)}: those re-index, re-normalise or swap the raw columns; not implemented")
        try:
            _g1c = np.concatenate([np.arange(int(u), int(v)) for u, v in (c.split(":") for c in a.vaug_cols.split(",") if c)])
        except Exception:
            raise SystemExit(f"--vaug-cols: expected lo:hi[,lo:hi..], got {a.vaug_cols!r}")
        if len(_g1c) == 0 or len(np.unique(_g1c)) != len(_g1c) or not all((22 <= c < 52) or (74 <= c < 104) for c in _g1c.tolist()):
            raise SystemExit(f"--vaug-cols {a.vaug_cols!r}: columns must be distinct and inside 22:52 or 74:104 (volume / trade blocks); "
                             "the price-code blocks 0:22 / 52:74 carry the label's path and the clock block has --clock-aug")
        if (a.seq_affine > 0 or a.seq_affine_shift > 0) and set(_g1c.tolist()) & set(np.concatenate(
                [np.arange(int(u), int(v)) for u, v in (c.split(":") for c in a.seq_affine_cols.split(",") if c)]).tolist()):
            raise SystemExit("--vaug and --seq-affine on overlapping columns: pick one")
        try:
            _g1f = [int(v) for v in a.vaug_fams.split(",") if v != ""]
        except ValueError:
            raise SystemExit(f"--vaug-fams: expected a comma list of 0..3, got {a.vaug_fams!r}")
        if any(f not in (0, 1, 2, 3) for f in _g1f): raise SystemExit(f"--vaug-fams: families are 0..3, got {a.vaug_fams!r}")
    if a.pcgrad != "off" or a.anchor_floor != 0 or a.sched_cycles != 1 or a.distill_fam_mult:   # add_b3.py
        if a.loss != "seqrun" and (a.pcgrad != "off" or a.anchor_floor != 0 or a.distill_fam_mult):
            raise SystemExit("--pcgrad / --anchor-floor / --distill-fam-mult are implemented in the seqrun loop only (a silent no-op otherwise)")
        if a.sched_cycles < 1: raise SystemExit("--sched-cycles N must be >= 1")
        if a.sched_cycles > 1 and not a.ref_opt: raise SystemExit("--sched-cycles needs --ref-opt (the LambdaLR schedule)")
        if a.anchor_floor < 0: raise SystemExit("--anchor-floor K must be >= 0")
        if a.anchor_floor > 0:
            if not a.distill: raise SystemExit("--anchor-floor needs --distill (the teacher's residual is the floor)")
            if a.anchor != "mse" or a.anchor_target != "y" or a.smooth_anchor > 0 or a.cosine_loss:
                raise SystemExit("--anchor-floor is implemented for the plain MSE anchor on y only")
        if a.distill_fam_mult:
            if not a.distill: raise SystemExit("--distill-fam-mult needs --distill")
            try:
                _b3m = [float(v) for v in a.distill_fam_mult.split(",")]
            except ValueError:
                raise SystemExit(f"--distill-fam-mult: expected M0,M1,M2,M3, got {a.distill_fam_mult!r}")
            if len(_b3m) != 4 or any(v < 0 for v in _b3m): raise SystemExit("--distill-fam-mult: exactly four factors >= 0")
        if a.pcgrad != "off":
            if a.pcgrad == "term" and not (a.distill and a.distill_weight > 0):
                raise SystemExit("--pcgrad term needs --distill with --distill-weight > 0 (the second task)")
            _b3bad = [n for n, v in (("--hetero", a.hetero > 0), ("--ns-reg", a.ns_reg > 0), ("--ar-reg", a.ar_reg > 0),
                                     ("--tar-h", a.tar_h > 0), ("--dann", a.dann > 0), ("--hint", bool(a.hint)),
                                     ("--mh", getattr(a, "mh", 0) > 0), ("--hlg", getattr(a, "hlg", 0) > 0), ("--paux", bool(a.paux)),
                                     ("--dualmode", a.dualmode > 0), ("--lattice-head", a.lattice_head > 0),
                                     ("--z14-reparam", a.z14_reparam != "off"), ("--sam", a.sam > 0), ("--ncl", a.ncl > 0),
                                     ("--wdrop", a.wdrop > 0), ("--wnoise", a.wnoise > 0), ("--learn-h0", bool(a.learn_h0)),
                                     ("--mask-aux", a.mask_aux > 0), ("--aux-next", a.aux_next > 0), ("--levelaux", a.levelaux > 0),
                                     ("--aux-horizons", bool(a.aux_horizons)), ("--grid-head", a.grid_head > 0),
                                     ("--ssl-aux", a.ssl_aux > 0)) if v]
            if _b3bad:
                raise SystemExit(f"--pcgrad with {', '.join(_b3bad)}: those add gradient paths that bypass the head output or perturb "
                                 "the weights around backward; not implemented")
    if a.bookconv_taps != 3 or a.bookconv_unshare or a.tradeconv or a.instshare:   # add_s1.py (research S1)
        if (a.bookconv_taps != 3 or a.bookconv_unshare) and not a.bookconv:
            raise SystemExit("--bookconv-taps / --bookconv-unshare need --bookconv C > 0 (a silent no-op otherwise)")
        if a.bookconv_taps < 1 or a.bookconv_taps % 2 == 0: raise SystemExit("--bookconv-taps K must be odd and >= 1")
        if a.tradeconv < 0 or a.instshare < 0: raise SystemExit("--tradeconv / --instshare must be >= 0")
        if not a.preset.startswith("refgru"): raise SystemExit("--bookconv-taps/--bookconv-unshare/--tradeconv/--instshare are refgru inputs")
    if a.bookconv_split or a.bookconv_side:             # add_h7_bcsplit.py (research H7: restricted bookconv kernels)
        if not a.preset.startswith("refgru"): raise SystemExit("--bookconv-split / --bookconv-side are refgru inputs")
        if not a.bookconv: raise SystemExit("--bookconv-split / --bookconv-side need --bookconv C > 0 (a silent no-op otherwise)")
        if a.bookconv_split and a.bookconv_side: raise SystemExit("--bookconv-split and --bookconv-side are exclusive (one kernel form)")
        if a.bookconv % 2: raise SystemExit("--bookconv-split / --bookconv-side need an even --bookconv C (C/2 channels per part)")
        if a.bookconv_unshare: raise SystemExit("--bookconv-split / --bookconv-side with --bookconv-unshare: not implemented")
        if a.bookconv_act: raise SystemExit("--bookconv-split / --bookconv-side with --bookconv-act: export not implemented")
        if getattr(a, "extra_in", ""): raise SystemExit("--bookconv-split / --bookconv-side with --extra-in: not implemented")
        _h7n = a.bookconv_taps * 2 * (a.bookconv // 2) * (2 if a.bookconv_split else 1)
        print(f"h7: bookconv kernel {'SPLIT (price pair -> C/2 channels, volume pair -> C/2, block-diagonal)' if a.bookconv_split else 'SIDE-TIED (one (p_side, v_side) kernel shared by the bid and ask sides)'}: "
              f"C {a.bookconv}, {a.bookconv_taps} taps, {_h7n} kernel weights instead of {a.bookconv_taps * 4 * a.bookconv}; same 2*C*11 "
              f"features and in_proj.0 shape as --bookconv; folded at export (research H7, add_h7_bcsplit.py)", flush=True)
    if a.xinst or a.bookconv_act or a.hh_rank:          # add_struct.py (research ST: model / input structures)
        if not a.preset.startswith("refgru"): raise SystemExit("--xinst / --bookconv-act / --hh-rank are refgru structures")
        if a.xinst < 0: raise SystemExit("--xinst C must be >= 0")
        if a.bookconv_act and not a.bookconv: raise SystemExit("--bookconv-act needs --bookconv C > 0 (a silent no-op otherwise)")
        if a.bookconv_act and a.bookconv_unshare: raise SystemExit("--bookconv-act with --bookconv-unshare: export not implemented")
        if a.hh_rank:
            if not (16 <= a.hh_rank < a.hidden): raise SystemExit("--hh-rank R must satisfy 16 <= R < --hidden")
            if a.layers != 1 or a.cell != "gru" or a.split_towers or a.bidir or a.dualmode or a.fastw or a.ncl > 0:
                raise SystemExit("--hh-rank: plain single-layer single-tower causal refgru GRU only (no lstm/split/bidir/dualmode/fastw/ncl)")
            if (a.wdrop > 0 or a.wnoise > 0 or a.rec_wd > 0 or a.z14_reparam != "off" or a.orth_init or a.gate_init != "default"
                    or a.init_from or a.gru_grad != 1.0):
                raise SystemExit("--hh-rank: --wdrop/--wnoise/--rec-wd/--z14-reparam/--orth-init/--gate-init/--init-from/--gru-grad act on "
                                 "gru.weight_hh_l0 or hook core.gru; not implemented for the factored recurrence (would be silent no-ops)")
        if getattr(a, "extra_in", ""): raise SystemExit("--xinst/--bookconv-act/--hh-rank with --extra-in: not implemented")
        print(f"struct: xinst {a.xinst} (cross-instrument ladder read, folded at export), bookconv-act '{a.bookconv_act}' "
              f"(position-wise nonlinearity on the shared book read, NOT foldable), hh-rank {a.hh_rank} (rank-R recurrence, "
              f"materialised at export) (research ST, add_struct.py)", flush=True)
    if a.inproj_init_mult != 1.0 or a.densein:         # add_s1b.py (research S1 controls)
        if a.inproj_init_mult <= 0: raise SystemExit("--inproj-init-mult M must be > 0")
        if a.densein < 0: raise SystemExit("--densein K must be >= 0")
        if not a.preset.startswith("refgru"): raise SystemExit("--inproj-init-mult / --densein are refgru inputs")
        if a.inproj_init_mult != 1.0 and a.proj_gated: raise SystemExit("--inproj-init-mult is implemented for the plain projection only")
    if a.clock_aug != 0:                               # add_z36clk.py
        if a.clock_aug < 0: raise SystemExit("--clock-aug DELTA must be >= 0")
        if a.loss != "seqrun": raise SystemExit("--clock-aug is implemented in the seqrun loop only (a silent no-op otherwise)")
        if a.bidir: raise SystemExit("--clock-aug is a student-side augmentation; not with --bidir")
        try:
            _ckr = [float(v) for v in a.clock_ratios.split(",")]
        except ValueError:
            raise SystemExit(f"--clock-ratios: expected 5 floats R0,R2,R3,R5,R6, got {a.clock_ratios!r}")
        if len(_ckr) != 5: raise SystemExit(f"--clock-ratios: expected 5 floats R0,R2,R3,R5,R6, got {a.clock_ratios!r}")
    if not (a.resnorm_w > 0 or a.resnorm_ed > 0) and (a.resnorm_mode != "add" or a.resnorm_famnorm or a.resnorm_beta != 1.0 or a.resnorm_target != "y"
                                                      or a.resnorm_weighting != "metric" or a.resnorm_ed_fam or a.resnorm_ed_p or a.resnorm_sd0 != 1.2):
        raise SystemExit("--resnorm-* sub-flags need --resnorm-w > 0 or --resnorm-ed > 0 (research H12: they would be a silent no-op)")
    if a.resnorm_w > 0 or a.resnorm_ed > 0:              # research H12 (add_h12_resnorm.py): refuse silent no-ops and clashes
        if a.loss != "seqrun" or not a.preset.startswith("refgru") or not a.two_head:
            raise SystemExit("--resnorm-w/--resnorm-ed: --loss seqrun with --preset refgru --two-head only")
        if a.resnorm_mode == "replace" and not (a.resnorm_w > 0 and a.anchor == "mse" and a.anchor_floor == 0 and not a.cosine_loss):
            raise SystemExit("--resnorm-mode replace: needs --resnorm-w > 0 and the plain MSE anchor (--anchor mse, no --anchor-floor, no --cosine-loss)")
        if a.resnorm_target == "t" and not (a.distill and a.distill_weight > 0):
            raise SystemExit("--resnorm-target t needs --distill DIR and --distill-weight > 0")
        if not (0.0 <= a.resnorm_beta <= 1.0): raise SystemExit("--resnorm-beta must be in [0, 1]")
        if a.resnorm_famnorm and a.resnorm_beta >= 1.0: raise SystemExit("--resnorm-famnorm is a no-op at --resnorm-beta 1 (p's weights are already the anchor's)")
        if a.resnorm_w <= 0 and (a.resnorm_mode == "replace" or a.resnorm_famnorm or a.resnorm_beta != 1.0):
            raise SystemExit("--resnorm-mode/--resnorm-famnorm/--resnorm-beta need --resnorm-w > 0")
        if a.hetero > 0 or a.hlg > 0 or a.mh > 0 or a.grid_head > 0 or a.lattice_head or a.split_towers or a.bidir or a.dualmode:
            raise SystemExit("--resnorm-*: not with --hetero / --hlg / --mh / --grid-head / --lattice-head / --split-towers / --bidir / --dualmode")
        if a.resnorm_sd0 <= 0: raise SystemExit("--resnorm-sd0 must be > 0")
        print(f"resnorm (research H12): nll W {a.resnorm_w} beta {a.resnorm_beta} mode {a.resnorm_mode} target {a.resnorm_target} "
              f"weighting {a.resnorm_weighting} famnorm {a.resnorm_famnorm} | energy W2 {a.resnorm_ed} per-fam {a.resnorm_ed_fam} "
              f"grad-p {a.resnorm_ed_p} | sigma0 {a.resnorm_sd0}; train-only head resnorm_head (never exported)", flush=True)
    if a.muon_scope in ("noinproj0", "noinproj3") and a.opt not in ("muon", "soap"):
        raise SystemExit(f"--muon-scope {a.muon_scope} needs --opt muon or soap (a silent no-op under --opt {a.opt})")
    if a.ref_opt:
        # The public reference recipe: decay only weight matrices (no biases, no norm
        # gains), betas (0.9, 0.98) unless overridden, 5% linear warm-up, cosine to 2%.
        import math as _m
        _dec = [q for n, q in core.named_parameters()
                if q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias")]
        _nodec = [q for n, q in core.named_parameters()
                  if not (q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias"))]
        _b = _betas.get("betas", (0.9, 0.98))
        if a.opt == "sgd":
            # research I 3.5: SGD + Nesterov momentum was never swept here (DRW 2025 1st: Adam-family
            # optimizers underperformed on noisy financial targets); same decay split and schedule
            opt = torch.optim.SGD([{"params": _dec, "weight_decay": a.wd},
                                   {"params": _nodec, "weight_decay": 0.0}], lr=a.lr, momentum=a.momentum, nesterov=True)
            print(f"optimizer: SGD lr {a.lr} momentum {a.momentum} nesterov", flush=True)
        elif a.opt == "muon":
            # Muon on hidden matrices (per gate for the GRU unless --muon-gate-split 0), AdamW on the rest
            _mu, _mdec, _mnodec = _muon_split(core, bool(a.muon_gate_split), a.muon_scope)
            _mwd = a.wd if a.muon_wd < 0 else a.muon_wd
            _mg = {}
            for _n, _q, _nb in _mu: _mg.setdefault(_nb, []).append((_n, _q))
            _groups = [{"params": [_q for _, _q in _nq], "names": [_n for _n, _ in _nq], "use_muon": True, "lr": a.muon_lr,
                        "weight_decay": _mwd, "momentum": a.muon_momentum, "ns_steps": a.muon_ns, "gate_blocks": _nb,
                        "nesterov": bool(a.muon_nesterov), "scale_mode": a.muon_scale, "rownorm": a.muon_rownorm,
                        "cautious": a.muon_cautious, "mars": a.muon_mars}
                       for _nb, _nq in sorted(_mg.items())]
            _groups += [{"params": _qs, "lr": a.lr, "weight_decay": _w, "betas": _b, "cautious": a.muon_cautious}
                        for _qs, _w in ((_mdec, a.wd), (_mnodec, 0.0)) if _qs]
            if not _mu: raise SystemExit("--opt muon: no hidden 2-D weight matrices found")
            if a.muon_adam > 0:
                # research Z27 H2 (add_z27.py): the chosen in_proj Muon matrices ALSO take the AdamW update of the same
                # gradient (own moments, lr muon_adam x --lr on the same schedule, the recipe's betas, Muon's decay only).
                # Only the Muon groups holding a target are split; every other group is passed through unchanged.
                _z27p = {"inproj": "in_proj.", "inproj0": "in_proj.0.", "inproj3": "in_proj.3."}[a.muon_adam_on]
                _z27n = [_n for _n, _q, _nb in _mu if _n.startswith(_z27p)]
                if not _z27n:
                    raise SystemExit(f"--muon-adam: no Muon matrix named {_z27p}* under --muon-scope {a.muon_scope} "
                                     f"(the hybrid would be a silent no-op)")
                _z27g = []
                for _g in _groups:
                    if not _g.get("use_muon"):
                        _z27g.append(_g); continue
                    _ia = [i for i, _n in enumerate(_g["names"]) if _n.startswith(_z27p)]
                    _ib = [i for i in range(len(_g["names"])) if i not in _ia]
                    if _ib: _z27g.append(dict(_g, params=[_g["params"][i] for i in _ib], names=[_g["names"][i] for i in _ib]))
                    if _ia: _z27g.append(dict(_g, params=[_g["params"][i] for i in _ia], names=[_g["names"][i] for i in _ia],
                                              z27_adam=a.lr * a.muon_adam / a.muon_lr, z27_betas=_b))
                _groups = _z27g
                print(f"z27: Muon + AdamW hybrid on {', '.join(_z27n)}: every step adds {a.muon_adam} x the AdamW update of the "
                      f"same gradient (lr {a.lr * a.muon_adam:g} on the schedule, betas {_b}, own moments, no extra decay); "
                      f"{len(_groups)} param groups", flush=True)
            opt = _MuonAdamW(_groups)
            _z8 = [_s for _s, _on in (("nesterov OFF (heavy-ball)", not a.muon_nesterov),
                                      ("scale rms 0.2*sqrt(max dim)", a.muon_scale == "rms"),
                                      (f"NorMuon row-norm b2 {a.muon_rownorm}", a.muon_rownorm > 0),
                                      (f"cautious {a.muon_cautious} " + ("(all groups)" if a.muon_cautious >= 2 else "(Muon matrices)"),
                                       a.muon_cautious > 0),
                                      (f"MARS gamma {a.muon_mars}", a.muon_mars > 0)) if _on]
            print(f"optimizer: MUON lr {a.muon_lr} momentum {a.muon_momentum} {'nesterov' if a.muon_nesterov else 'heavy-ball'}, "
                  f"NS {a.muon_ns} (fp32), wd {_mwd} (p *= 1 - lr*wd) on {len(_mu)} matrices: "
                  + ", ".join(f"{_n}{tuple(_q.shape)}" + (f"/{_nb}blk" if _nb > 1 else "") for _n, _q, _nb in _mu)
                  + f" | AdamW lr {a.lr} betas {_b}: {len(_mdec)} matrices (wd {a.wd}) + {len(_mnodec)} vectors"
                  + (f" | add_z8: {', '.join(_z8)}" if _z8 else ""),
                  flush=True)
        elif a.opt == "soap":
            # SOAP (Shampoo eigenbases + Adam) on the hidden matrices, per gate for the GRU unless --muon-gate-split 0;
            # AdamW on the rest (add_soap.py)
            _mu, _mdec, _mnodec = _muon_split(core, bool(a.muon_gate_split), a.muon_scope)
            if not _mu: raise SystemExit("--opt soap: no hidden 2-D weight matrices found")
            _slr = a.lr if a.soap_lr < 0 else a.soap_lr
            _swd = a.wd if a.soap_wd < 0 else a.soap_wd
            _sbt = tuple(float(v) for v in a.soap_betas.split(",")) if a.soap_betas else _b
            _mg = {}
            for _n, _q, _nb in _mu: _mg.setdefault(_nb, []).append((_n, _q))
            _groups = [{"params": [_q for _, _q in _nq], "names": [_n for _n, _ in _nq], "use_soap": True,
                        "lr": _slr, "lr_mult": a.soap_lr_mult, "weight_decay": _swd, "betas": _sbt, "eps": a.soap_eps,
                        "precond_freq": a.soap_precond_freq, "shampoo_beta": a.soap_beta2, "gate_blocks": _nb}
                       for _nb, _nq in sorted(_mg.items())]
            _groups += [{"params": _qs, "lr": a.lr, "weight_decay": _w, "betas": _b}
                        for _qs, _w in ((_mdec, a.wd), (_mnodec, 0.0)) if _qs]
            opt = _SoapAdamW(_groups)
            print(f"optimizer: SOAP lr {_slr} x mult "
                  + ("auto (matched to AdamW's step RMS at the first update)" if a.soap_lr_mult <= 0 else f"{a.soap_lr_mult}")
                  + f", betas {_sbt} eps {a.soap_eps}, L/R beta {a.soap_beta2}, eigh every {a.soap_precond_freq} updates (fp32), "
                  f"wd {_swd} (p *= 1 - lr*wd) on {len(_mu)} matrices: "
                  + ", ".join(f"{_n}{tuple(_q.shape)}" + (f"/{_nb}blk" if _nb > 1 else "") for _n, _q, _nb in _mu)
                  + f" | AdamW lr {a.lr} betas {_b}: {len(_mdec)} matrices (wd {a.wd}) + {len(_mnodec)} vectors",
                  flush=True)
        elif a.opt == "lion":
            _lb = tuple(float(v) for v in a.lion_betas.split(","))
            opt = _Lion([{"params": _dec, "weight_decay": a.wd},
                         {"params": _nodec, "weight_decay": 0.0}], lr=a.lr, betas=_lb)
            print(f"optimizer: LION lr {a.lr} betas {_lb} wd {a.wd} on {len(_dec)} matrices "
                  f"(0 on {len(_nodec)} vectors; --betas ignored)", flush=True)
        else:
            opt = torch.optim.AdamW([{"params": _dec, "weight_decay": a.wd},
                                     {"params": _nodec, "weight_decay": 0.0}], lr=a.lr, betas=_b)
        _warm = max(1, int(0.05 * getattr(a, "_r2_T", a.steps)))       # add_r2_branch.py: the parent's T (== a.steps when off)
        def _lr_mult(step, _w=_warm, _T=getattr(a, "_r2_T", a.steps)):
            if step < _w:
                return (step + 1) / _w
            pr = min((step - _w) / max(1, _T - _w), 1.0)
            if a.sched_cycles > 1 and pr < 1.0: pr = (pr * a.sched_cycles) % 1.0     # add_b3.py: warm restarts (SGDR), N equal cycles
            if a.sched == "d2z": return max(1.0 - pr, 0.0)
            if a.sched == "wsd": return 1.0 if pr < 0.8 else max((1.0 - pr) / 0.2, 0.0)
            return 0.02 + 0.98 * 0.5 * (1 + _m.cos(_m.pi * pr))
        if getattr(a, "_r2_off", 0) > 0:                 # add_r2_branch.py: continue the parent's schedule (+ optional re-warm)
            _r2_f0 = _lr_mult
            def _lr_mult(step, _f=_r2_f0, _o=a._r2_off, _rw=a.branch_rewarm):
                return _f(step + _o) * (min(1.0, (step + 1) / _rw) if _rw > 0 else 1.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_mult)
        print(f"ref-opt: AdamW betas {_b}, wd {a.wd} on {len(_dec)} matrices "
              f"(0 on {len(_nodec)} vectors), warm-up {_warm} steps then cosine to 2%",
              flush=True)
    outdir = ROOT/"runs"/a.tag; outdir.mkdir(parents=True, exist_ok=True)
    hist, best, t0 = [], -1e9, time.perf_counter()
    # Provenance for reproducibility: the exact code that produced the weights.
    import hashlib as _hl, subprocess as _sp
    try:
        _git = _sp.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        _git = ""
    if not _git and (ROOT/"runs/CODE_VERSION").exists():      # the box has no checkout: the sync writes it
        _git = (ROOT/"runs/CODE_VERSION").read_text().strip()
    _srcs = ["src/train.py", "src/losses.py", "src/data.py", "src/model.py", "src/arch/refgru.py"]
    a.provenance = {"git": _git, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"),
                    "deterministic": not a.nondeterministic,
                    "src_md5": {f: _hl.md5((ROOT/f).read_bytes()).hexdigest()[:12] for f in _srcs if (ROOT/f).exists()}}
    print(f"provenance: git {_git[:10]} torch {torch.__version__} src {a.provenance['src_md5']}", flush=True)

    seq_p = None
    # seq_p is consumed ONLY by sample_grouped() in the windowed loop. The seqrun
    # loop draws its sequences from SequenceStream, which never sees it, so both
    # flags would be silent no-ops there -- and a silent no-op is how a full-data
    # run gets written up as a subset result. Refuse instead.
    if a.loss == "seqrun" and a.seq_bias != "none":
        raise SystemExit("--seq-bias is not implemented for --loss seqrun "
                         "(SequenceStream ignores seq_p); it would silently do nothing")
    if a.seq_subset:
        sub = np.load(ROOT/a.seq_subset)
        seq_p = np.zeros(len(train)); seq_p[sub] = 1.0 / len(sub)
        print(f"seq-subset: sampling restricted to {len(sub)} of {len(train)} sequences "
              f"({Path(a.seq_subset).stem})")
    if a.seq_bias != "none":
        # book activity per training sequence, cached: mean |diff| of the i0 price block
        apath = ROOT / a.train_cache / "seq_activity.npy"
        if apath.exists():
            act = np.load(apath)
        else:
            print("computing per-sequence book activity (one pass)...", flush=True)
            act = np.array([float(np.abs(np.diff(np.asarray(train.X[i][WARMUP::17, :22]), axis=0)).mean())
                            for i in range(len(train))], dtype=np.float64)
            np.save(apath, act)
        r = act.argsort().argsort() / max(len(act)-1, 1)          # uniform rank in [0,1]
        w = (1.0 - r) if a.seq_bias == "quiet" else r
        seq_p = (w + 0.25); seq_p = seq_p / seq_p.sum()           # 5:1 tilt, never zero
        print(f"seq-bias {a.seq_bias}: sampling weight {seq_p.min()*len(seq_p):.2f}x..{seq_p.max()*len(seq_p):.2f}x")

    seq_mu = seq_sd = None
    if a.loss in ("anchored", "zmse"):
        # per-sequence |y|-weighted mean/std of the clipped signal; cached, it is a
        # single pass over Y and never changes for a given cache
        cpath = ROOT / a.train_cache / "seq_stats.npz"
        if cpath.exists():
            z = np.load(cpath); seq_mu, seq_sd = z["mu"], z["sd"]
        else:
            print("computing per-sequence target stats (one pass over Y)...", flush=True)
            seq_mu, seq_sd = seq_target_stats(train)
            np.savez(cpath, mu=seq_mu, sd=seq_sd)
        print(f"seq stats: mu {seq_mu.mean():+.4f}  sd {seq_sd.mean():.4f}")

    if a.loss in ("seqwp", "anchored", "zmse"):
        a.batch = a.groups * a.per_seq
        print(f"{a.loss}: {a.groups} sequences x {a.per_seq} windows = batch {a.batch}; "
              f"WP statistics pooled per sequence over {a.per_seq*a.body} rows")

    state_bank = None

    if a.loss == "seqrun":
        # Whole-sequence objective. Each lane walks ONE sequence from row 0 to the end in
        # chunks, carrying the recurrent state AND the weighted-Pearson sufficient
        # statistics. Gradients still span one chunk, so BPTT depth is unchanged -- what
        # becomes whole-sequence is the objective, which is where level drift across the
        # other ~90% of the sequence finally gets penalised.
        import copy as _copy
        if a.lattice_head:
            _BASES_T = torch.as_tensor(_BASES, device=dev, dtype=torch.float32)
        stream = SequenceStream(train, a.groups, a.chunk, rng, row_weight=rw,
                                epochs=(a.epochs if a.epochs > 0 else None),
                                train_row_factor=a.train_row_factor,
                                extra_row_factor=a.extra_row_factor, mix_lambda=a.mix_lambda, fam0_mix=a.fam0_mix,
                                fam_mix=(seq_fam_shares if a.family_mix else None),
                                fam_subset=[int(v) for v in a.fam_subset.split(",") if v] or None,
                                sign_balance=[int(v) for v in a.sign_balance.split(",") if v] or None,
                                big_emph=a.big_emph,
                                seq_subset=(np.load(ROOT/a.seq_subset) if a.seq_subset else None),
                                fam0_repeat=a.fam0_repeat, subset_all=a.seq_subset_all)
        _z48 = None
        if a.dann > 0:
            # research Z48 (add_z48.py): domain-adversarial regime head on the post-LN head input, gradient reversal.
            if not hasattr(core, "post_ln") or getattr(core, "split", False) or getattr(core, "bidir", False):
                raise SystemExit("--dann needs a single-tower causal refgru core (post_ln head input)")
            if a.dann_groups == "period":
                if not stream.has_extra:
                    raise SystemExit("--dann-groups period needs --extra-cache (primary = 0, extra = 1)")
                _z48l = np.r_[np.zeros(stream.n_primary, np.int64), np.ones(stream.n_total - stream.n_primary, np.int64)]
            else:
                _z48p = Path(a.dann_groups) if Path(a.dann_groups).is_absolute() else ROOT / a.dann_groups
                if not _z48p.exists():
                    raise SystemExit(f"--dann-groups: {_z48p} is missing")
                _z48f = np.load(_z48p)
                if _z48f.ndim != 1 or len(_z48f) not in (stream.n_primary, stream.n_total) or not np.issubdtype(_z48f.dtype, np.integer):
                    raise SystemExit(f"--dann-groups: {_z48p} is {_z48f.dtype} {_z48f.shape}; expected ints ({stream.n_primary},) "
                                     f"or ({stream.n_total},)")
                _z48l = np.full(stream.n_total, -1, np.int64); _z48l[:len(_z48f)] = _z48f
            _z48K = int(_z48l.max()) + 1
            _z48c = np.bincount(_z48l[_z48l >= 0], minlength=max(_z48K, 1))
            if _z48K < 2 or (_z48c == 0).any():
                raise SystemExit(f"--dann-groups: need >= 2 non-empty groups, got counts {_z48c.tolist()}")
            if a.dann_stride < 1:
                raise SystemExit("--dann-stride must be >= 1")
            _z48D = int(core.post_ln.normalized_shape[0])
            with torch.random.fork_rng(devices=[]):                 # the core's RNG stream stays untouched
                _z48adv = torch.nn.Sequential(torch.nn.Linear(_z48D, a.dann_hidden), torch.nn.GELU(),
                                              torch.nn.Linear(a.dann_hidden, _z48K))
            _z48gen = torch.Generator().manual_seed(int(a.seed) + 48000)
            with torch.no_grad():
                for _z48m in _z48adv:
                    if isinstance(_z48m, torch.nn.Linear):
                        _z48b = 1.0 / float(_z48m.in_features) ** 0.5
                        _z48m.weight.copy_(torch.rand(_z48m.weight.shape, generator=_z48gen) * 2 * _z48b - _z48b)
                        _z48m.bias.zero_()
            _z48adv = _z48adv.to(dev)

            class _Z48GRL(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x_, lam_):
                    ctx.lam = lam_
                    return x_.view_as(x_)

                @staticmethod
                def backward(ctx, g_):
                    return -ctx.lam * g_, None

            _z48 = dict(lab=_z48l, cw=torch.tensor(_z48c.sum() / (_z48K * _z48c), dtype=torch.float32, device=dev),
                        adv=_z48adv, opt=torch.optim.AdamW(_z48adv.parameters(), lr=a.dann_lr, weight_decay=0.0),
                        grl=_Z48GRL, K=_z48K, n=0, ce=0.0, hit=0.0, rows=0)
            core._z48_keep = True
            print(f"z48 dann: weight {a.dann:g}, groups '{a.dann_groups}' K={_z48K} counts {_z48c.tolist()} "
                  f"(class weights {[round(float(v), 3) for v in _z48['cw']]}), adversary {_z48D}->{a.dann_hidden}->{_z48K} "
                  f"(lr {a.dann_lr:g}, own init seed {int(a.seed) + 48000}), rows every {a.dann_stride} post-warm-up, "
                  f"GRL ramp 2/(1+exp(-10p))-1; train-only, never saved or exported (research Z48)", flush=True)
        if a.extra_fam_repeat or a.seq_repeat:
            # research Z46 (add_z46.py): per-sequence repeat counts in the epoch draw pool (SequenceStream._z46_pool). The
            # step count is unchanged (it counts len(train) sequences), so repeated sequences displace other visits.
            if a.extra_weight > 1.0:
                raise SystemExit("--extra-fam-repeat / --seq-repeat with --extra-weight > 1: two oversampling rules; not implemented")
            _z46r = np.ones(stream.n_total, np.int64)
            if a.seq_repeat:
                if a.seq_subset_all:
                    raise SystemExit("--seq-repeat with --seq-subset-all: the union index is not implemented")
                _z46p = Path(a.seq_repeat) if Path(a.seq_repeat).is_absolute() else ROOT / a.seq_repeat
                if not _z46p.exists():
                    raise SystemExit(f"--seq-repeat: {_z46p} is missing")
                _z46s = np.load(_z46p)
                if (_z46s.ndim != 1 or len(_z46s) != stream.n_primary or not np.issubdtype(_z46s.dtype, np.integer)
                        or _z46s.min() < 0 or _z46s.max() > 16 or _z46s.sum() == 0):
                    raise SystemExit(f"--seq-repeat: {_z46p} is {_z46s.dtype} {_z46s.shape} (min {_z46s.min()}, max {_z46s.max()}); "
                                     f"expected integers 0..16, shape ({stream.n_primary},), not all zero")
                _z46r[:stream.n_primary] = _z46s
            _z46f = None
            if a.extra_fam_repeat:
                if not stream.has_extra:
                    raise SystemExit("--extra-fam-repeat needs --extra-cache")
                _z46f = [int(v) for v in a.extra_fam_repeat.split(",")]
                if len(_z46f) != 4 or min(_z46f) < 1 or max(_z46f) > 16:
                    raise SystemExit(f"--extra-fam-repeat {a.extra_fam_repeat}: expected four integers 1..16")
                for _z46i in range(stream.n_primary, stream.n_total):
                    _z46fi = stream.family(_z46i)
                    if _z46fi >= 0:
                        _z46r[_z46i] = _z46f[_z46fi]
            stream.z46_rep = _z46r
            # redo the constructor's first draw so the counts hold from the first epoch (the constructor drew before they existed)
            stream._queue = []; stream.epoch = 0; stream.drawn[:] = 0
            stream.seq = np.array([stream._draw() for _ in range(a.groups)])
            stream.new_epoch = False; stream.lane_epoch[:] = stream.epoch
            _z46np, _z46nt = stream.n_primary, stream.n_total
            _z46ex = ""
            if _z46f is not None:
                _z46fx = np.array([stream.family(i) for i in range(_z46np, _z46nt)])
                _z46ex = ", extra pool by family " + str([int(_z46r[_z46np:][_z46fx == f].sum()) for f in range(4)]) + \
                         f" (sequences {np.bincount(_z46fx[_z46fx >= 0], minlength=4).tolist()}, x{_z46f})"
            print(f"z46: epoch draw pool {int(_z46r.sum())} = primary {int(_z46r[:_z46np].sum())} ({int((_z46r[:_z46np] != 1).sum())} "
                  f"of {_z46np} sequences re-counted) + extra {int(_z46r[_z46np:].sum())} ({_z46nt - _z46np} sequences){_z46ex}; "
                  f"optimizer steps unchanged ({a.steps}), repeats displace other visits (research Z46)", flush=True)
        if getattr(a, "stagger", False):
            _cps = -(-SEQUENCE_LENGTH // a.chunk)
            stream.pos[:] = (np.arange(a.groups) * _cps // a.groups) * a.chunk
            print(f"stagger: lane start chunks {sorted(set((stream.pos // a.chunk).tolist()))[:6]}... of {_cps}", flush=True)
        if a.chunk_jitter:                             # add_z15.py: chunk-boundary phase jitter
            if getattr(a, "stagger", False): raise SystemExit("--chunk-jitter needs the lanes in lock-step; drop --stagger")
            if a.chunk < 3: raise SystemExit("--chunk-jitter needs --chunk >= 3")
            if a.epochs <= 0: print("WARNING: --chunk-jitter without --epochs: --steps is taken as given (41 chunks per sequence)", flush=True)
            stream.set_chunk_jitter(np.random.default_rng(a.seed + 70_000))
            print(f"chunk-jitter: the first chunk of every sequence is 1..{a.chunk - 1} rows (rng seed+70000, data order "
                  f"unchanged), then {a.chunk}-row chunks and a tail; TBPTT boundaries move every sequence and epoch", flush=True)
        if a.extra_weight > 1.0:
            stream.extra_repeat = int(round(a.extra_weight))    # seqrun: whole-sequence oversampling of the extra cache
        print("row factor / sampling:\n" + stream.describe(), flush=True)
        if a.distill:
            _td = Path(a.distill)
            _tp = np.load(_td/"train_full.npy", mmap_mode="r")
            # The extra-cache teacher file must match the extra cache ACTUALLY in use. This was
            # hard-coded to valid_retrain.npy, so a full-data member (--extra-cache cache/valid_all,
            # 1,873 sequences) loaded the 1,073-sequence file and died on the shape check -- which is
            # exactly what should happen, but it meant no distilled member could ever be trained on
            # the full data, i.e. no distilled model could ever ship.
            _tename = Path(a.extra_cache).name if a.extra_cache else "valid_retrain"
            _tef = _td/f"{_tename}.npy"
            if a.extra_cache and not _tef.exists():
                raise SystemExit(
                    f"distillation: teacher {a.distill} has no {_tename}.npy, but --extra-cache is "
                    f"{a.extra_cache}. Dump the teacher over that cache first:\n"
                    f"  python src/teacher_dump.py --tags ... --raw --caches {_tename} --out {a.distill}")
            _te = np.load(_tef, mmap_mode="r") if _tef.exists() else None
            stream.set_teacher(_tp, _te)
            print(f"distillation: teacher {a.distill} weight {a.distill_weight} "
                  f"(extra: {_tename})", flush=True)
        if a.distill2 and a.distill2_weight > 0:
            _td2 = ROOT / a.distill2
            _tp2 = np.load(_td2 / "train_full.npy", mmap_mode="r")
            _tename2 = Path(a.extra_cache).name if a.extra_cache else "valid_retrain"
            _tef2 = _td2 / f"{_tename2}.npy"
            if a.extra_cache and not _tef2.exists():
                raise SystemExit(f"distill2: teacher {a.distill2} has no {_tename2}.npy for --extra-cache {a.extra_cache}")
            _te2 = np.load(_tef2, mmap_mode="r") if _tef2.exists() else None
            stream.set_teacher2(_tp2, _te2)
            print(f"distillation #2: teacher {a.distill2} weight {a.distill2_weight} (extra: {_tename2})", flush=True)
        _z37 = None                                    # --dens-mid (add_z37.py, research Z37)
        _z38_mult, _z38u = None, None                  # --dens-fam-mult (add_z38.py, research Z38)
        if a.dens_fam_mult:
            _z38g = [[float(v) for v in g.split(",")] for g in a.dens_fam_mult.split("/")]
            _z38_mult = np.asarray(_z38g if len(_z38g) == 2 else _z38g * 2, dtype=np.float64)        # (2 instruments, 4 families)
            print(f"dens-fam-mult: exact densaux target x per-family multiplier (family = SequenceStream.family, the FM0 rule; "
                  f"0 = masked, unknown family masked): i0 {_z38_mult[0].tolist()}, i1 {_z38_mult[1].tolist()} (research Z38)", flush=True)
        if a.dens_mid:
            _Z37_SENT = -32768
            _zd = Path(a.dens_mid) if Path(a.dens_mid).is_absolute() else ROOT / a.dens_mid
            _zparts = [(Path(a.train_cache).name, stream.c.a if stream.has_extra else stream.c, stream.n_primary)]
            if stream.has_extra:
                _zparts.append((Path(a.extra_cache).name, stream.c.b, stream.n_total - stream.n_primary))
            _zarr = []
            for _znm, _zcache, _znn in _zparts:
                _zf, _zmf = _zd / f"{_znm}.npy", _zd / f"{_znm}_meta.json"
                if not (_zf.exists() and _zmf.exists()):
                    raise SystemExit(f"--dens-mid: {_zf} or {_zmf.name} is missing (build: exact-mid dump builder, not included in this repo)")
                _zaa = np.load(_zf, mmap_mode="r")
                if _zaa.dtype != np.int16 or _zaa.shape != (_znn, SEQUENCE_LENGTH, 2):
                    raise SystemExit(f"--dens-mid: {_zf} is {_zaa.dtype} {_zaa.shape}, expected int16 ({_znn}, {SEQUENCE_LENGTH}, 2)")
                if json.load(open(_zmf)).get("seq_ix") != list(_zcache.meta["seq_ix"]):
                    raise SystemExit(f"--dens-mid: {_zmf} seq_ix differs from the {_znm} cache manifest (wrong cache or order)")
                _zarr.append(_zaa)
            _z37 = (_zarr[0], _zarr[1] if len(_zarr) > 1 else None)
            if a.dens_cross_chunk:                   # add_z55.py
                print(f"dens-cross-chunk: each chunk reads the exact mid path {max(dens_h)} rows past its end, so every row "
                      f"gets every horizon {dens_h}; masked only past row {SEQUENCE_LENGTH} or across an unknown row (research Z55)",
                      flush=True)
            print(f"dens-mid: densaux targets from the EXACT mid paths in {_zd} ({', '.join(n for n, _, _ in _zparts)}; int16 "
                  f"half-ticks, {_Z37_SENT} = unknown); horizon-h differences only where every row of [t, t+h] is decoded, "
                  f"standardised over those entries; the p-column means are no longer used (research Z37)", flush=True)
        _z39, _z39_mult, _z39u = None, None, None      # --paux (add_z39.py, research Z39)
        if a.paux:
            _z9d = Path(a.paux) if Path(a.paux).is_absolute() else ROOT / a.paux
            _z9parts = [(Path(a.train_cache).name, stream.c.a if stream.has_extra else stream.c, stream.n_primary)]
            if stream.has_extra:
                _z9parts.append((Path(a.extra_cache).name, stream.c.b, stream.n_total - stream.n_primary))
            _z9arr, _z9chs = [], []
            for _z9nm, _z9cache, _z9nn in _z9parts:
                _z9f, _z9mf = _z9d / f"{_z9nm}.npy", _z9d / f"{_z9nm}_meta.json"
                if not (_z9f.exists() and _z9mf.exists()):
                    raise SystemExit(f"--paux: {_z9f} or {_z9mf.name} is missing (build: rule-target dump builder, not included in this repo)")
                _z9aa = np.load(_z9f, mmap_mode="r"); _z9meta = json.load(open(_z9mf))
                if _z9aa.dtype != np.float16 or _z9aa.shape != (_z9nn, SEQUENCE_LENGTH, _paux_k):
                    raise SystemExit(f"--paux: {_z9f} is {_z9aa.dtype} {_z9aa.shape}, expected float16 "
                                     f"({_z9nn}, {SEQUENCE_LENGTH}, {_paux_k})")
                if _z9meta.get("seq_ix") != list(_z9cache.meta["seq_ix"]):
                    raise SystemExit(f"--paux: {_z9mf} seq_ix differs from the {_z9nm} cache manifest (wrong cache or order)")
                _z9chs.append([c.get("name") for c in _z9meta.get("channels", [])])
                _z9arr.append(_z9aa)
            if any(c != _z9chs[0] for c in _z9chs) or len(_z9chs[0]) != _paux_k:
                raise SystemExit(f"--paux: channel lists differ between the dumps or from K = {_paux_k}: {_z9chs}")
            _z39 = (_z9arr[0], _z9arr[1] if len(_z9arr) > 1 else None)
            if a.paux_fam_mult:
                _z9g = [[float(v) for v in g.split(",")] for g in a.paux_fam_mult.split("/")]
                _z39_mult = np.asarray(_z9g * _paux_k if len(_z9g) == 1 else _z9g, dtype=np.float64).T     # (4 families, K)
            print(f"paux: channels {_z9chs[0]} from {_z9d} ({', '.join(n for n, _, _ in _z9parts)}); NaN = masked"
                  + (f"; per-family multipliers (rows F0..F3, columns = channels) {_z39_mult.tolist()}, unknown family masked"
                     if _z39_mult is not None else "") + " (research Z39)", flush=True)
        if a.elr > 0:                                  # add_z12.py: ELR self-target store
            if getattr(core, "bidir", False) or a.lattice_head or a.row_drop > 0:
                raise SystemExit("--elr is not implemented for --bidir, --lattice-head or --row-drop")
            if 0 < a.epochs <= 1:
                raise SystemExit("--elr needs --epochs > 1 (a row's target is recorded on its first visit and used on later visits)")
            if not (0.0 <= a.elr_beta < 1.0):
                raise SystemExit("--elr-beta must be in [0, 1)")
            _elr_store = np.full((stream.n_total, SEQUENCE_LENGTH, 2), np.nan, np.float16)
            stream.set_elr(_elr_store)
            print(f"elr: per-row temporal-ensemble self-target over {stream.n_total} sequences "
                  f"({_elr_store.nbytes / 2 ** 20:.0f} MB float16), weight {a.elr}, beta {a.elr_beta}, recording from step "
                  f"{int(a.elr_warm * a.steps) + 1} of {a.steps}; |clip(y)| x row-weighted MSE to the ensemble of the model's "
                  f"own earlier-pass predictions (ELR, Liu et al. 2020)", flush=True)
        if a.mask_aux > 0:
            if a.split_towers or a.bidir: raise SystemExit("--mask-aux: single-tower causal GRU only")
            print(stream.enable_mask_out(), flush=True)
        if a.curric > 0:
            if not a.distill or stream.teacher is None:
                raise SystemExit("--curric needs --distill: the difficulty threshold comes from the teacher dump")
            if a.epochs <= 0:
                print("WARNING: --curric without --epochs: uniform sampling never leaves epoch 0, so the curriculum never ends", flush=True)
            print(stream.set_curriculum(a.curric), flush=True)
        if a.trim > 0:
            print(f"trimmed anchor: the top {a.trim:.1%} of supervised rows by |p - clip(y)| leave the anchor term each step "
                  f"(Pearson and distillation terms unchanged)", flush=True)
        rstat = RunningWP(a.groups, 2, dev)
        rstat_auxh = [RunningWP(a.groups, 2, dev) for _ in aux_off] if _auxh_base is not None else []
        rstat_dm = RunningWP(a.groups, 2, dev) if a.dualmode else None
        rstat_mh = [RunningWP(a.groups, 2, dev) for _ in range(a.mh)]   # add_z59.py: one running WP per --mh sub-head
        if a.dualmode and a.dualmode_init:
            _sdb = torch.load(ROOT / a.dualmode_init, map_location="cpu", weights_only=False)["state"]
            with torch.no_grad():
                for _k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                    getattr(core.aux_bwd, _k).copy_(_sdb[f"gru.{_k}_reverse"].to(dev))
            print(f"dualmode: aux_bwd warm-started from {a.dualmode_init} (*_reverse)", flush=True)
        core2 = opt2 = rstat2 = h2 = None
        if a.ncl > 0:
            import copy as _c2
            torch.manual_seed(a.seed + 10_000)                 # a different draw, same data order
            core2 = build(a.preset, n_out=n_out, **kw).to(dev)
            core2.set_norm(core.mu.cpu().numpy(), core.sd.cpu().numpy())
            _b2 = _betas.get("betas", (0.9, 0.98))
            _d2 = [q for n, q in core2.named_parameters()
                   if q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias")]
            _n2 = [q for n, q in core2.named_parameters()
                   if not (q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias"))]
            opt2 = torch.optim.AdamW([{"params": _d2, "weight_decay": a.wd},
                                      {"params": _n2, "weight_decay": 0.0}], lr=a.lr, betas=_b2)
            sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, _lr_mult) if a.ref_opt else \
                     torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=[g['lr'] for g in opt2.param_groups],
                                                         total_steps=a.steps, pct_start=0.15, final_div_factor=30)
            h2 = core2.init_state(a.groups, dev)
            rstat2 = RunningWP(a.groups, 2, dev)
            print(f"NCL: two members trained jointly, lambda {a.ncl}", flush=True)
        h = core.init_state(a.groups, dev)
        hint_core = hint_h = None
        if a.hint and a.hint_weight > 0:
            from score_ckpt import load as _hload
            _hp = Path(a.hint)
            hint_core = _hload(_hp.parent.name, _hp.name)[0].to(dev).eval()
            for q in hint_core.parameters(): q.requires_grad_(False)
            if not getattr(hint_core, "bidir", False): raise SystemExit("--hint needs a --bidir teacher")
            hint_h = hint_core.init_state(a.groups, dev)
            print(f"hint distillation: teacher {a.hint} (forward GRU output, {hint_core.hidden} dims) "
                  f"weight {a.hint_weight}, adapter {tuple(core.hint_adapter.weight.shape)}", flush=True)
        if a.pretrain_steps > 0:
            # SELF-SUPERVISED PRETRAINING (offline/patches/add_pretrain.py). The same core is trained
            # for --pretrain-steps chunks on a label-free target with a separate Linear head that is
            # discarded afterwards; the supervised loop below then starts from the pretrained weights
            # with its own (still untouched) optimiser and schedule. The main stream and rng are not
            # consumed here, so the supervised data order equals the control's at the same seed.
            import math as _ptm
            _pt_rng = np.random.default_rng(a.seed + 20_000)
            _pt_stream = SequenceStream(train, a.groups, a.chunk, _pt_rng, epochs=None,
                                        train_row_factor="one", extra_row_factor="one",
                                        seq_subset=(np.load(ROOT / a.seq_subset) if a.seq_subset else None),
                                        subset_all=a.seq_subset_all)
            _pt_F = int(core.mu.shape[0]); _pt_K = max(1, int(a.pretrain_horizon))
            _pt_head = torch.nn.Linear(core.hidden, _pt_F).to(dev)
            _pt_lr = a.pretrain_lr if a.pretrain_lr > 0 else a.lr
            _pt_dec = [q for n, q in core.named_parameters()
                       if q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias")]
            _pt_nodec = [q for n, q in core.named_parameters()
                         if not (q.ndim >= 2 and "norm" not in n.lower() and not n.endswith("bias"))]
            _pt_opt = torch.optim.AdamW([{"params": _pt_dec + [_pt_head.weight], "weight_decay": a.wd},
                                         {"params": _pt_nodec + [_pt_head.bias], "weight_decay": 0.0}],
                                        lr=_pt_lr, betas=_betas.get("betas", (0.9, 0.98)))
            _pt_warm = max(1, int(0.05 * a.pretrain_steps))
            def _pt_mult(step, _w=_pt_warm, _T=a.pretrain_steps):
                if step < _w: return (step + 1) / _w
                _pp = min((step - _w) / max(1, _T - _w), 1.0)
                return 0.02 + 0.98 * 0.5 * (1 + _ptm.cos(_ptm.pi * _pp))
            _pt_sched = torch.optim.lr_scheduler.LambdaLR(_pt_opt, _pt_mult)
            _pt_h = core.init_state(a.groups, dev)
            _pt_mode = (f"masked-row reconstruction (mask {a.pretrain_mask:.2f})" if a.pretrain_mask > 0
                        else f"next-row prediction (horizon {_pt_K})")
            print(f"pretraining: {a.pretrain_steps} steps of {_pt_mode} on {len(train)} sequences, "
                  f"lr {_pt_lr:.1e}, head {core.hidden}->{_pt_F} (discarded after)", flush=True)
            _pt_t0 = time.time(); core.train(); _pt_last = float("nan")
            for _ps in range(1, a.pretrain_steps + 1):
                _X, _, _W, _reset = _pt_stream.next()
                if _reset.any():
                    _pt_h = _pt_h.clone(); _pt_h[:, np.where(_reset)[0]] = 0.0
                if keep_cols is not None: _X = _X[..., keep_cols]
                # valid rows: every real row, warm-up included (W is 0 there and on end-padding;
                # the warm-up rows of a lane are the first WARMUP rows of the chunk that resets it)
                _vm = _W > 0
                if _reset.any(): _vm[_reset, :WARMUP] = True
                _vm = torch.from_numpy(_vm).to(dev).permute(1, 0).unsqueeze(-1)            # (T,B,1)
                _x = torch.from_numpy(_X).permute(1, 0, 2).to(dev)                              # (T,B,F) raw
                _xs = ((_x - core.mu) / core.sd).clamp(-core.clamp, core.clamp)                  # target
                if a.pretrain_mask > 0:
                    _mk = torch.rand(_x.shape[0], _x.shape[1], 1, device=dev) < a.pretrain_mask
                    _x = torch.where(_mk, core.mu.expand_as(_x), _x)
                _, _pt_h = core(_x, _pt_h.detach())
                _pred = _pt_head(core._gru_out)
                if a.pretrain_mask > 0:
                    _pt_m = (_vm & _mk).to(_pred.dtype); _tg = _xs
                else:
                    _pred, _tg = _pred[:-_pt_K], _xs[_pt_K:]; _pt_m = _vm[_pt_K:].to(_pred.dtype)
                _hub = torch.nn.functional.huber_loss(_pred, _tg, reduction="none", delta=1.0)
                _pl = (_hub * _pt_m).sum() / (_pt_m.sum() * _pt_F).clamp_min(1.0)
                _pt_opt.zero_grad(set_to_none=True); _pl.backward()
                torch.nn.utils.clip_grad_norm_(list(core.parameters()) + list(_pt_head.parameters()), a.clip_norm)
                _pt_opt.step(); _pt_sched.step(); _pt_last = _pl.item()
                if _ps % 100 == 0 or _ps == a.pretrain_steps:
                    _el = time.time() - _pt_t0
                    print(f"  pretrain {_ps:5d}/{a.pretrain_steps}  loss {_pt_last:.4f}  "
                          f"lr {_pt_sched.get_last_lr()[0]:.2e}  {_ps/max(_el,1e-6):.2f} steps/s  {_el/60:.1f} min", flush=True)
            a.pretrain_final_loss = _pt_last
            del _pt_head, _pt_opt, _pt_sched, _pt_stream, _pt_h
            for _at in ("_gru_out", "_x_std", "_aux_out", "_grid_logits"):   # forward-time graph
                if hasattr(core, _at): setattr(core, _at, None)              # tensors block deepcopy (EMA)
            print(f"pretraining done in {(time.time() - _pt_t0)/60:.1f} min; supervised loop starts "
                  f"from the pretrained weights (fresh optimiser)", flush=True)
        _z18_cols_t = None
        if _z18_cols is not None:
            # add_z18.py: --drop-cols / --keep-cols. The dropped RAW columns are removed by zeroing their columns of the first
            # input-projection matrix and masking their gradient, so the graph, n_features and the exported weights keep the
            # 112-wide input (the fused builders consume plain weights); mu/sd/clamp buffers are untouched. Done AFTER any
            # warm start / pretraining and BEFORE the EMA copy, so both trajectories start at exactly zero.
            _ip = getattr(core, "in_proj", None)
            # add_da_dropbc.py (research DA): the raw 112 columns are always the FIRST 112 inputs of in_proj.0; the linear
            # re-reads (bookconv / tradeconv / instshare / densein / xinst, add_s1 / add_s1b / add_struct) append their
            # features after them and read the raw x directly, so zeroing raw columns composes with them. Any other
            # input-changing flag leaves the width different from this sum and is refused as before.
            _z18_w = 112 + 2 * 11 * int(getattr(a, "bookconv", 0) or 0) + 2 * 4 * int(getattr(a, "tradeconv", 0) or 0) \
                + 2 * int(getattr(a, "instshare", 0) or 0) + int(getattr(a, "densein", 0) or 0) + 11 * int(getattr(a, "xinst", 0) or 0)
            if not (isinstance(_ip, torch.nn.Sequential) and isinstance(_ip[0], torch.nn.Linear) and _ip[0].in_features == _z18_w):
                raise SystemExit("--drop-cols/--keep-cols need the plain refgru input projection on the raw 112 columns "
                                 "(+ the linear re-reads bookconv/tradeconv/instshare/densein/xinst; not --proj-gated, --diff, "
                                 "--ewma, --nlfeat, --ple-bins, split towers, --rankgauss or --family-input)")
            if _z18_w != 112:
                print(f"drop-cols: in_proj.0 is {_ip[0].in_features} wide (raw 112 + {_z18_w - 112} linear re-read features); "
                      f"only the RAW columns are zeroed, the re-reads keep reading them (add_da_dropbc.py)", flush=True)
            _z18_cols_t = torch.as_tensor(_z18_cols, device=dev, dtype=torch.long)
            with torch.no_grad(): _ip[0].weight[:, _z18_cols_t] = 0.0
            _ip[0].weight.register_hook(lambda g, _c=_z18_cols_t: g.index_fill(1, _c, 0.0))
            print(f"drop-cols: {len(_z18_cols)} of 112 raw input columns zeroed in in_proj.0.weight and gradient-masked "
                  f"(train, eval and export see them as absent; graph unchanged): {_z18_fmt(_z18_cols)}", flush=True)
        ema = None
        if a.ema > 0:
            ema = _copy.deepcopy(core).eval()
            for q in ema.parameters(): q.requires_grad_(False)
            print(f"EMA of weights at decay {a.ema}, evaluated beside the raw weights", flush=True)
        _ems = _ems_parse(a.ema_extra, a.steps) if a.ema_extra else []     # add_ems.py (research EMS)
        if _ems and getattr(a, "branch_epoch", 0) > 0:
            raise SystemExit("--ema-extra with --branch-epoch: the start fractions would refer to the branch, not the run")
        for _ex in _ems:
            _ex["m"] = _copy.deepcopy(core).eval()
            for q in _ex["m"].parameters(): q.requires_grad_(False)
        if _ems:
            print("ema-extra (never used for training; saved as ema_<name>_final.pt): " + "; ".join(
                f"{_ex['name']} = " + (f"uniform mean from step {_ex['s0']}" if _ex["kind"] == "swa" else
                                       f"EMA {_ex['d0']:g}" + (f"->{_ex['d1']:g} (linear from step {_ex['r0']})"
                                                               if _ex["d1"] != _ex["d0"] else "")
                                       + (f" reset at step {_ex['s0']}" if _ex["s0"] > 0 else " from the initial weights"))
                for _ex in _ems), flush=True)
        _sd_h = None                                   # add_z9.py: the EMA teacher's carried state (--self-distill)
        if a.self_distill > 0:
            if ema is None: raise SystemExit("--self-distill needs --ema > 0 (the EMA copy is the teacher)")
            if getattr(core, "bidir", False) or a.lattice_head:
                raise SystemExit("--self-distill is not implemented for --bidir or --lattice-head")
            _sd_h = ema.init_state(a.groups, dev)
            print(f"self-distill: mean-teacher consistency to the EMA({a.ema}) copy at weight {a.self_distill}, "
                  f"|clip(y)| x row-weighted MSE, applied from step {int(a.self_distill_start * a.steps) + 1} of {a.steps} "
                  f"(teacher state carried from step 1)", flush=True)
        _la_slow = None                                # add_z9.py: Lookahead slow weights (--lookahead-k)
        if a.lookahead_k > 0:
            _la_slow = [q.detach().clone() for q in core.parameters()]
            print(f"lookahead: k {a.lookahead_k} alpha {a.lookahead_alpha} on {len(_la_slow)} tensors "
                  f"(slow weights synced after every {a.lookahead_k}-th optimizer step, before the EMA update)", flush=True)

        def _ckpt(model, path, extra=None):
            d = {"state": _z14_export_state(model),          # add_z14.py: identical dict when no z14 tensor exists
                 "cfg": vars(a), "params": core.n_params()}
            if a.hlg > 0 or a.mh > 0:                         # add_z59.py: export format; the raw training state kept aside
                d["z59_train_state"] = d["state"]; d["state"] = _z59_export_state(d["state"])
            if extra: d.update(extra)
            torch.save(d, path)

        def _eval_point(step, n, label=""):
            nonlocal best
            sc = calibrate_scale(core, train, two_head=a.two_head,
                                 common_head=a.common_head, device=dev, keep_cols=keep_cols,
                                 xform=(_rg if rg is not None else None), famv=(fam if a.family_input else None), fnorm=fnorm, famv_norm=fam, lat=lat_tr)
            r = evaluate(core, ev, n=n, two_head=a.two_head,
                         common_head=a.common_head, scale=sc, device=dev, keep_cols=keep_cols,
                         xform=(_rg if rg is not None else None), famv=(fam_ev if a.family_input else None), fnorm=fnorm, famv_norm=fam_ev, lat=lat_ev,
                         reset_every=a.state_reset)
            r["scale"] = sc
            if ema is not None:
                sce = calibrate_scale(ema, train, two_head=a.two_head,
                                      common_head=a.common_head, device=dev, keep_cols=keep_cols,
                                      xform=(_rg if rg is not None else None), famv=(fam if a.family_input else None), fnorm=fnorm, famv_norm=fam, lat=lat_tr)
                re_ = evaluate(ema, ev, n=n, two_head=a.two_head,
                               common_head=a.common_head, scale=sce, device=dev, keep_cols=keep_cols,
                               xform=(_rg if rg is not None else None), famv=(fam_ev if a.family_input else None), fnorm=fnorm, famv_norm=fam_ev, lat=lat_ev,
                               reset_every=a.state_reset)
                r["wp_ema"], r["scale_ema"] = re_["wp"], sce
            if label: r["label"] = label
            hist.append({"step": step, **r})
            flag = ""
            if r["wp"] > best:
                best = r["wp"]; flag = "  <-- best"
                _ckpt(core, outdir/"best.pt", {"scale": sc, "wp": r["wp"]})
            ema_txt = f"  ema {r['wp_ema']:+.4f}" if ema is not None else ""
            print(f"  EVAL step {step}{(' ' + label) if label else ''}: WP {r['wp']:+.4f} "
                  f"POOLED {r.get('wp_pooled', float('nan')):+.4f} "
                  f"+/- {r['se']:.4f}{ema_txt}  (n={r['n']}, scale x{r['scale']:.3f} -> "
                  f"pred_std {r['pred_std']:.3f}){flag}", flush=True)
            return r

        t0 = time.time()
        n_updates = 0
        _elr_n = 0                                     # add_z12.py: steps on which the ELR term was active
        _HAS_H = True
        _nk = {}                                    # cached AR(1) noise kernel for --in-noise-hl
        _sa_cols = None                             # --seq-affine (add_seqaffine.py)
        if a.seq_affine > 0 or a.seq_affine_shift > 0:
            _sa_rng = np.random.default_rng(a.seed + 50_000)   # own rng, data order unchanged
            _sa_cols = np.concatenate([np.arange(int(u), int(v)) for u, v in
                                       (c.split(":") for c in a.seq_affine_cols.split(",") if c)])
            _sa_cols_t = torch.as_tensor(_sa_cols, device=dev, dtype=torch.long)
            _sa_g = np.ones((a.groups, 1), np.float32); _sa_b = np.zeros((a.groups, 1), np.float32)
            print(f"seq-affine: per-sequence gain exp({a.seq_affine}*N) and shift {a.seq_affine_shift}*N*sd "
                  f"on {len(_sa_cols)} columns ({a.seq_affine_cols}), train-only", flush=True)
        _ck_cols = None                             # --clock-aug (add_z36clk.py, research Z36)
        if a.clock_aug > 0:
            _ck_rng = np.random.default_rng(a.seed + 36_000)   # own rng, data order unchanged
            _ck_cols = np.array([104, 106, 107, 109, 110])      # a0, a2, a3, a5, a6 (log-time columns, raw)
            _ck_r = np.array([float(v) for v in a.clock_ratios.split(",")], np.float32)
            _CK_CLIP = np.float32(5.19934)                      # the data's global clip; a5/a6 log(0) rows sit at -_CK_CLIP
            _ck_u = _ck_rng.uniform(-a.clock_aug, a.clock_aug, a.groups).astype(np.float32)   # covers --stagger lanes
            _ck_n = 0
            print(f"clock-aug: per-sequence log-dilation u ~ U(-{a.clock_aug:g}, {a.clock_aug:g}) (a0 sd units) on raw "
                  f"cols {_ck_cols.tolist()} with ratios {_ck_r.tolist()}; floor rows kept, clip +-{float(_CK_CLIP)}; "
                  f"train-only", flush=True)
        _b3_pc = None                               # --pcgrad (add_b3.py, research B3)
        _b3_kdm = None                              # --distill-fam-mult
        _b3_fl = None                               # --anchor-floor running stats
        if a.pcgrad != "off" or a.distill_fam_mult:
            def _b3_lane_fam():
                """tick family of each lane's CURRENT sequence (-1 = unknown), read as add_z38 reads it."""
                return np.array([stream.family(int(_i)) for _i in stream.cur[:, 0].tolist()], np.int64)
        if a.distill_fam_mult:
            _b3_kdm = np.array([float(v) for v in a.distill_fam_mult.split(",")], np.float32)
            print(f"distill-fam-mult: KD row weight x {_b3_kdm.tolist()} by lane family (unknown family x1) (research B3)", flush=True)
        if a.anchor_floor > 0:
            _b3_fl = {"off_mass": 0.0, "mass": 0.0, "steps": 0, "plain": 0.0, "floored": 0.0}
            print(f"anchor-floor: row anchor = max(0, (p - clip y)^2 - {a.anchor_floor:g} (t - clip y)^2), t = the lookahead teacher; "
                  "the share of anchor mass below the floor is logged every 1000 steps (research B3)", flush=True)
        if a.pcgrad != "off":
            _b3_pc = {"params": [_q for _q in core.parameters() if _q.requires_grad], "rng": np.random.default_rng(a.seed + 73_000),
                      "steps": 0, "pairs": 0, "conflicts": 0, "cos_sum": 0.0, "cos_n": 0, "ratio": 0.0, "check": float("nan"),
                      "names": (["F0", "F1", "F2", "F3"] if a.pcgrad == "fam" else ["y-terms", "KD"])}
            print(f"pcgrad: {a.pcgrad} -- {len(_b3_pc['names'])} task gradients ({'/'.join(_b3_pc['names'])}) taken through the head "
                  "output; conflicting pairs projected (seeded order); the parts are checked against the plain gradient at step 1 and "
                  "every 1000 steps (research B3)", flush=True)

            def _b3_pcgrad_backward(_loss, _o, _loss_y, _step):
                """replaces loss.backward(): per-task gradients through o, PCGrad projection, parameters' .grad set to the sum."""
                _P = _b3_pc["params"]
                if a.pcgrad == "fam":
                    _f = _b3_lane_fam()
                    _go = torch.autograd.grad(_loss, _o, retain_graph=True)[0]
                    _gos = [_go * torch.from_numpy(((_f == _k) | ((_f < 0) & (_k == 0))).astype(np.float32)).to(_go.device)[:, None, None]
                            for _k in range(4)]
                else:
                    _gy = torch.autograd.grad(_loss_y, _o, retain_graph=True)[0]
                    _gt = torch.autograd.grad(_loss, _o, retain_graph=True)[0]
                    _gos = [_gy, _gt - _gy]
                _do_check = (_step == 1 or _step % 1000 == 0)
                _G = []
                for _i, _g in enumerate(_gos):
                    for _q in _P: _q.grad = None
                    torch.autograd.backward(_o, grad_tensors=_g, retain_graph=(_do_check or _i + 1 < len(_gos)))
                    _G.append(torch.cat([(_q.grad.detach().reshape(-1).clone() if _q.grad is not None else torch.zeros(_q.numel(), device=_q.device, dtype=_q.dtype))
                                         for _q in _P]))
                if _do_check:
                    for _q in _P: _q.grad = None
                    _loss.backward()
                    _full = torch.cat([(_q.grad.detach().reshape(-1) if _q.grad is not None else torch.zeros(_q.numel(), device=_q.device, dtype=_q.dtype)) for _q in _P])
                    _b3_pc["check"] = float((sum(_G) - _full).norm() / _full.norm().clamp_min(1e-12))
                    if _b3_pc["check"] > 1e-3:
                        raise SystemExit(f"--pcgrad: the task gradients do not sum to the plain gradient (rel {_b3_pc['check']:.2e}): "
                                         "a loss term bypasses the head output; refusing to continue")
                _n = len(_G); _proj = [_g.clone() for _g in _G]
                _b3_pc["steps"] += 1
                for _i in range(_n):
                    for _j in _b3_pc["rng"].permutation(_n):
                        if _j == _i: continue
                        _d = torch.dot(_proj[_i], _G[_j]); _b3_pc["pairs"] += 1
                        if _d < 0:
                            _b3_pc["conflicts"] += 1
                            _proj[_i] = _proj[_i] - _d / torch.dot(_G[_j], _G[_j]).clamp_min(1e-20) * _G[_j]
                with torch.no_grad():
                    _nr = [_g.norm().clamp_min(1e-12) for _g in _G]
                    for _i in range(_n):
                        for _j in range(_i + 1, _n):
                            _b3_pc["cos_sum"] += float(torch.dot(_G[_i], _G[_j]) / (_nr[_i] * _nr[_j])); _b3_pc["cos_n"] += 1
                    _tot = sum(_proj); _plain = sum(_G)
                    _b3_pc["ratio"] += float(_tot.norm() / _plain.norm().clamp_min(1e-12))
                    _off = 0
                    for _q in _P:
                        _nq = _q.numel(); _q.grad = _tot[_off:_off + _nq].view_as(_q).clone(); _off += _nq
                if _do_check:
                    print(f"        pcgrad[{a.pcgrad}]: mean pairwise cos {_b3_pc['cos_sum'] / max(1, _b3_pc['cos_n']):+.3f}, conflicting pairs "
                          f"{100.0 * _b3_pc['conflicts'] / max(1, _b3_pc['pairs']):.1f}%, |g_proj|/|g_plain| {_b3_pc['ratio'] / _b3_pc['steps']:.3f}, "
                          f"sum-check rel {_b3_pc['check']:.1e}", flush=True)
        _g1_cols = None                             # --vaug (add_g1.py, research G1)
        if a.vaug > 0 or a.vaug_gain > 0:
            _g1_rng = np.random.default_rng(a.seed + 61_000)   # own rng, data order unchanged
            _g1_cols = np.concatenate([np.arange(int(u), int(v)) for u, v in (c.split(":") for c in a.vaug_cols.split(",") if c)]).astype(np.int64)
            _g1_mu = core.mu.detach().cpu().numpy().reshape(-1).astype(np.float32)[_g1_cols]
            _g1_sd = core.sd.detach().cpu().numpy().reshape(-1).astype(np.float32)[_g1_cols]
            _g1_fams = {int(v) for v in a.vaug_fams.split(",") if v != ""} or None
            _g1_nc = 1 if a.vaug_mode == "common" else len(_g1_cols)
            _g1_b = np.zeros((a.groups, _g1_nc), np.float32); _g1_g = np.ones((a.groups, _g1_nc), np.float32)
            _g1_n = {"steps": 0, "on": 0, "off": 0}
            def _g1_draw(lanes, seqs):
                """fresh (b, g) for these lanes, whose current sequences are `seqs` (combined-cache indices)."""
                _n = len(lanes)
                _b = (a.vaug * _g1_rng.standard_normal((_n, _g1_nc))).astype(np.float32) if a.vaug > 0 else np.zeros((_n, _g1_nc), np.float32)
                _g = np.exp(a.vaug_gain * _g1_rng.standard_normal((_n, _g1_nc))).astype(np.float32) if a.vaug_gain > 0 else np.ones((_n, _g1_nc), np.float32)
                if _g1_fams is not None:
                    _off = np.array([stream.family(int(_s)) not in _g1_fams for _s in seqs], bool)
                    _b[_off] = 0.0; _g[_off] = 1.0
                    _g1_n["off"] += int(_off.sum()); _g1_n["on"] += int((~_off).sum())
                else:
                    _g1_n["on"] += _n
                _g1_b[lanes] = _b; _g1_g[lanes] = _g
            _g1_draw(np.arange(a.groups), stream.seq)       # covers --stagger lanes; lock-step lanes are redrawn at step 1's reset
            print(f"vaug: per-sequence volume-level augmentation, mode {a.vaug_mode}: shift b ~ N(0, {a.vaug:g}) train-sd"
                  + (f", gain exp({a.vaug_gain:g} N)" if a.vaug_gain > 0 else "") + f" on {len(_g1_cols)} raw cols ({a.vaug_cols})"
                  + (f", families {sorted(_g1_fams)} only" if _g1_fams else ", all families")
                  + "; redrawn at every sequence start; labels / teacher / densaux / eval / export untouched (research G1)", flush=True)
        _sp_rng = np.random.default_rng(a.seed + 40_000)   # --splice: own rng, data order unchanged
        _sp_n = 0                                         # boundaries spliced so far
        _rd_rng = np.random.default_rng(a.seed + 30_000)   # --row-drop: own rng, data order unchanged
        # add_z18.py: --pw-schedule / --pw-start / --anchor-end. Per-step weights of the running-Pearson term and of the
        # anchor; the default returns exactly (a.pearson_weight, 1.0 - a.pearson_weight). The running-WP statistics are
        # accumulated by rstat.update() after every step regardless of the weight, so a late switch-on sees the full prefix.
        _z18_t1 = int(round(a.pw_start * a.steps)) if a.pw_schedule != "const" else 0
        _z18_t2 = int(round(a.anchor_end * a.steps)) if a.anchor_end > 0 else 0
        if a.pw_schedule != "const" or a.anchor_end > 0:
            if a.pw_schedule != "const" and not (0.0 < a.pw_start < 1.0): raise SystemExit("--pw-start must be in (0, 1) with --pw-schedule ramp/step")
            if a.anchor_end > 0 and not (0.0 < a.anchor_end < 1.0): raise SystemExit("--anchor-end must be in (0, 1)")
            if a.pw_schedule == "step" and 0 < _z18_t2 < _z18_t1: raise SystemExit("--anchor-end before --pw-start under --pw-schedule step leaves steps with no y-loss")
            if a.anchor_end > 0 and a.pearson_weight <= 0: raise SystemExit("--anchor-end needs --pearson-weight > 0")
            _z18_desc = {"step": f"0 for steps 1..{_z18_t1}, then {a.pearson_weight:g} for the last {100 * (a.steps - _z18_t1) / max(1, a.steps):.1f}% of training",
                         "ramp": f"ramps 0 -> {a.pearson_weight:g} over steps 1..{_z18_t1}, then {a.pearson_weight:g}",
                         "const": f"{a.pearson_weight:g} throughout"}[a.pw_schedule]
            print(f"pw-schedule {a.pw_schedule}: Pearson weight {_z18_desc}; anchor weight 1 - pw"
                  + (f", then 0 after step {_z18_t2} ({a.anchor_end:g})" if _z18_t2 else "")
                  + "; distillation unchanged; running-WP prefix accumulated every step (dualmode/NCL side losses keep the constant mix)", flush=True)
        def _z18_pw(step, _pwc=a.pearson_weight, _sch=a.pw_schedule, _t1=_z18_t1, _t2=_z18_t2):
            if _sch == "const" and _t2 == 0:
                return _pwc, 1.0 - _pwc
            pw = (_pwc if step > _t1 else 0.0) if _sch == "step" else (_pwc * min(1.0, step / max(1, _t1)) if _sch == "ramp" else _pwc)
            return pw, (0.0 if (_t2 and step > _t2) else 1.0 - pw)
        # add_z20.py: student-side KD mechanics (--distill-look / --distill-dis / --distill-step / --distill-clip). Every branch
        # below and in the loop is gated by its flag; with all flags off the distillation line is the unpatched one (bitwise).
        _z20_on = (a.distill_look != "none") or a.distill_dis > 0 or a.distill_step > 0 or a.distill_clip
        _z20_lookw = _z20_lookah = None; _z20_t1 = 0
        if _z20_on:
            if not (a.distill and a.distill_weight > 0):
                raise SystemExit("--distill-look/--distill-dis/--distill-step/--distill-clip need --distill DIR and --distill-weight > 0")
            if a.distill_mode != "mse":
                raise SystemExit("--distill-look/--distill-dis/--distill-step/--distill-clip are defined for --distill-mode mse only")
            if a.distill_look != "none":
                if a.chunk % 100 != 0 or SEQUENCE_LENGTH % a.chunk != 0:
                    raise SystemExit("--distill-look needs --chunk a multiple of 100 that divides 20,000: the teacher's 100-row dump "
                                     "windows start at row 0 and the chunk grid must stay aligned to them")
                if a.chunk_jitter or a.stagger or a.row_drop > 0 or a.splice > 0:
                    raise SystemExit("--distill-look: --chunk-jitter/--stagger/--row-drop/--splice move rows off the 100-row window grid")
                if a.distill_look_pow <= 0: raise SystemExit("--distill-look-pow must be > 0")
                _j = np.arange(a.chunk) % 100                       # position inside the dump window; the teacher saw 99 - _j future rows
                _fr = (100.0 - _j) / 100.0 if a.distill_look == "hi" else (1.0 + _j) / 100.0
                _z20_lookw = torch.as_tensor(_fr ** a.distill_look_pow, dtype=torch.float32, device=dev).view(1, -1, 1)
                _z20_lookah = torch.as_tensor(99 - _j, dtype=torch.float32, device=dev).view(1, -1, 1)
            if a.distill_dis > 0 and a.distill_dis_tau <= 0: raise SystemExit("--distill-dis-tau must be > 0")
            if a.distill_step > 0:
                if not (0.0 < a.distill_step < 1.0): raise SystemExit("--distill-step must be in (0, 1)")
                if a.distill_w2 < 0: raise SystemExit("--distill-w2 must be >= 0")
                if a.distill_ramp != "none": raise SystemExit("--distill-step and --distill-ramp are mutually exclusive")
                _z20_t1 = int(round(a.distill_step * a.steps))
            print("z20-kd: " + "; ".join(
                ([f"row weight x ((100-k)/100)^{a.distill_look_pow:g}" if a.distill_look == "hi" else f"row weight x ((1+k)/100)^{a.distill_look_pow:g}"]
                 + [f"k = row % 100 in the teacher's dump window (future rows seen = 99-k), {a.chunk}-row chunks aligned to the grid"]
                 if a.distill_look != "none" else [])
                + ([f"disagreement boost x (1 + {a.distill_dis:g} * r2/({a.distill_dis_tau:g} + r2)), r2 = mean_k (clip(y)-clip(t))^2"] if a.distill_dis > 0 else [])
                + ([f"KD weight {a.distill_weight:g} for steps 1..{_z20_t1}, then {a.distill_w2:g} for the last {100 * (a.steps - _z20_t1) / max(1, a.steps):.1f}%"] if a.distill_step > 0 else [])
                + (["target clip(t, +-2) instead of raw t"] if a.distill_clip else []))
                + "; KD denominator re-normalised, anchor/Pearson/distill2 unchanged", flush=True)
        _tr_rng = np.random.default_rng(a.seed + 60_000)   # add_z15.py: --train-reset-p: own rng, data order unchanged
        _tr_n, _tr_lanes = 0, 0
        if a.train_reset_p > 0:
            if not (0.0 < a.train_reset_p < 1.0): raise SystemExit("--train-reset-p must be in (0, 1)")
            print(f"train-reset: at every within-sequence chunk boundary each lane's carried state is zeroed with probability "
                  f"{a.train_reset_p:g} (expected run between resets {a.chunk / a.train_reset_p:.0f} rows); the running-WP "
                  f"statistics are kept; evaluation carries the state over the whole sequence (train-only)", flush=True)
        _sam_params = [q for q in core.parameters() if q.requires_grad]
        _sam_g, _sam_gn, _sam_eps = None, None, None   # --sam: previous gradient, its norm, current perturbation
        _rowpos = np.zeros(a.groups, np.int64)      # rows consumed per lane since its sequence start
        for step in range(1, a.steps+1):
            X, Yc, W, reset = stream.next()
            _T_out, _T2_out = stream.T_out, stream.T2_out
            if _z37 is not None:
                # --dens-mid (add_z37.py): exact-mid rows of exactly the lane-chunks next() just emitted (stream.cur = (seq, a, b)
                # per lane, combined-cache index); rows past b (padding) stay unknown
                _z55 = a.dens_cross_chunk                 # add_z55.py: also read H = max(dens_h) rows past a full chunk
                _z37m = np.full((X.shape[0], X.shape[1] + (max(dens_h) if _z55 else 0), 2), _Z37_SENT, np.int16)
                for _zg, (_zi, _za, _zb) in enumerate(stream.cur.tolist()):
                    _zr = min(_zb - _za, X.shape[1])
                    if _z55 and _zr == X.shape[1]:
                        _zr = min(_za + _z37m.shape[1], SEQUENCE_LENGTH) - _za
                    _z37m[_zg, :_zr] = (_z37[0][_zi] if _zi < stream.n_primary else _z37[1][_zi - stream.n_primary])[_za:_za + _zr]
                if _z38_mult is not None:
                    # --dens-fam-mult (add_z38.py): per-lane multiplier and validity for (i0, i1) from each lane's family
                    _z38f = np.array([stream.family(int(_zi)) for _zi in stream.cur[:, 0].tolist()], np.int64)
                    _z38q = np.where(_z38f[None, :] >= 0, _z38_mult[:, np.clip(_z38f, 0, 3)], 0.0)          # (2, G)
                    _z38u = (torch.from_numpy(_z38q.astype(np.float32)).to(dev),
                             torch.from_numpy((_z38q > 0).astype(np.float32)).to(dev))
            if _z39 is not None:
                # --paux (add_z39.py): exact aux-target rows of exactly the lane-chunks next() just emitted (stream.cur = (seq, a, b)
                # per lane, combined-cache index); rows past b (padding) stay NaN = masked
                _z39m = np.full((X.shape[0], X.shape[1], _paux_k), np.nan, np.float32)
                for _z9l, (_z9i, _z9a, _z9b) in enumerate(stream.cur.tolist()):
                    _z9r = min(_z9b - _z9a, X.shape[1])
                    _z39m[_z9l, :_z9r] = (_z39[0][_z9i] if _z9i < stream.n_primary else _z39[1][_z9i - stream.n_primary])[_z9a:_z9a + _z9r]
                if _z39_mult is not None:
                    _z9fam = np.array([stream.family(int(_z9i)) for _z9i in stream.cur[:, 0].tolist()], np.int64)
                    _z9q = np.where(_z9fam[:, None] >= 0, _z39_mult[np.clip(_z9fam, 0, 3)], 0.0)                  # (G, K)
                    _z39u = (torch.from_numpy(_z9q[:, None, :].astype(np.float32)).to(dev),
                             torch.from_numpy((_z9q[:, None, :] > 0).astype(np.float32)).to(dev))
            if _z43 is not None:
                # --extra-in (add_z43.py): the K decoded channels of exactly the lane-chunks next() just emitted (stream.cur =
                # (seq, a, b) per lane, combined-cache index), NaN -> fill, appended after the 112 raw columns; padding rows 0
                _z4x = np.zeros((X.shape[0], X.shape[1], _z43[2]), np.float32)
                for _z4l, (_z4i, _z4a, _z4b) in enumerate(stream.cur.tolist()):
                    _z4r = min(_z4b - _z4a, X.shape[1])
                    _z4x[_z4l, :_z4r] = (_z43[0][_z4i] if _z4i < stream.n_primary else _z43[1][_z4i - stream.n_primary])[_z4a:_z4a + _z4r]
                _z4x[np.isnan(_z4x)] = _z43[3]
                X = np.concatenate([X, _z4x], -1)
            _h0_lanes = np.arange(a.groups) if (a.learn_h0 and step == 1) else None   # add_learn_h0.py
            if a.row_drop > 0:
                # ROW DROPOUT / TIME WARP (offline/patches/add_rowdrop.py): remove a random subset of
                # the chunk's rows from the input stream and from every per-row target, so the GRU
                # sees this sequence at a randomly faster clock. Same rows for all lanes (one tensor
                # shape per step); the recurrent state is carried unchanged.
                _p = a.row_drop * (2.0 * _rd_rng.random() if a.row_drop_warp else 1.0)
                _keep = np.flatnonzero(_rd_rng.random(X.shape[1]) >= _p)
                if len(_keep) < 2:
                    _keep = np.sort(_rd_rng.choice(X.shape[1], 2, replace=False))
                X, Yc, W = X[:, _keep], Yc[:, _keep], W[:, _keep]
                if _T_out is not None: _T_out = _T_out[:, _keep]
                if _T2_out is not None: _T2_out = _T2_out[:, _keep]
            if reset.any():
                idx = np.where(reset)[0]
                _zidx = idx
                if a.splice > 0 and step > 1:
                    # SEQUENCE SPLICING (offline/patches/add_splice.py): lanes drawn here keep the
                    # state of the sequence they just finished; the others restart from zero.
                    _zidx = idx[_sp_rng.random(len(idx)) >= a.splice]
                    _sp_n += len(idx) - len(_zidx)
                h = h.clone()
                if a.learn_h0 and len(_zidx):   # lanes restart from the LEARNED state (gradient flows from this chunk)
                    h[:, _zidx] = core.init_state(len(_zidx), dev); _h0_lanes = _zidx
                else:
                    h[:, _zidx] = 0.0
                _rowpos[idx] = 0                    # a new sequence restarts the reset grid
                if _sa_cols is not None:            # new sequence, new per-sequence affine draw
                    _sa_g[idx, 0] = np.exp(a.seq_affine * _sa_rng.standard_normal(len(idx)))
                    _sa_b[idx, 0] = a.seq_affine_shift * _sa_rng.standard_normal(len(idx))
                if _ck_cols is not None:            # add_z36clk.py: new sequence, new log-dilation draw
                    _ck_u[idx] = _ck_rng.uniform(-a.clock_aug, a.clock_aug, len(idx)).astype(np.float32)
                if _g1_cols is not None:            # add_g1.py: new sequence, new per-column level draw (stream.cur = the new sequence)
                    _g1_draw(idx, stream.cur[idx, 0])
                if not a.global_pool:
                    # Under --global-pool the statistics deliberately carry across
                    # sequence boundaries: the metric has ONE centre for the whole
                    # dataset, not one per sequence.
                    rstat.reset(torch.from_numpy(idx).to(dev))
                    for _r in rstat_auxh: _r.reset(torch.from_numpy(idx).to(dev))
                    if rstat_dm is not None: rstat_dm.reset(torch.from_numpy(idx).to(dev))
                    for _r in rstat_mh: _r.reset(torch.from_numpy(idx).to(dev))      # add_z59.py
                if hint_h is not None:
                    hint_h = hint_h.clone(); hint_h[:, idx] = 0.0
                if _sd_h is not None:                  # add_z9.py: the teacher restarts on the same lanes as the student
                    _sd_h = _sd_h.clone(); _sd_h[:, _zidx] = 0.0
                if core2 is not None:
                    h2 = h2.clone(); h2[:, idx] = 0.0
                    if not a.global_pool: rstat2.reset(torch.from_numpy(idx).to(dev))
            if _ck_cols is not None:
                # CLOCK DILATION (add_z36clk.py, research Z36): raw space, before column selection / famnorm / the model's
                # (x - mu)/sd. X is a fresh array from SequenceStream.next(); one u per lane for its whole sequence.
                _ckv = X[:, :, _ck_cols]
                _ckd = _ck_u[:, None, None] * _ck_r[None, None, :]
                X[:, :, _ck_cols] = np.where(_ckv <= -_CK_CLIP + 1e-4, _ckv,
                                             np.clip(_ckv + _ckd, -_CK_CLIP, _CK_CLIP)).astype(X.dtype)
                _ck_n += 1
                if step % 1000 == 0:
                    print(f"        clock-aug: lane u mean {float(_ck_u.mean()):+.3f} sd {float(_ck_u.std()):.3f}", flush=True)
            if _g1_cols is not None:
                # VOLUME-LEVEL AUGMENTATION (add_g1.py, research G1): raw space, per lane (= per sequence) per column:
                # x' = mu + g (x - mu) + b sd on the volume-like columns; padding rows past a short chunk's end stay as they are.
                # The i0 price-code block (the label's source), the targets, the weights and the teacher rows are untouched.
                _g1v = X[:, :, _g1_cols]
                _g1w = (_g1_mu + _g1_g[:, None, :] * (_g1v - _g1_mu) + _g1_b[:, None, :] * _g1_sd).astype(X.dtype)
                _g1ok = (np.arange(X.shape[1])[None, :] < (stream.cur[:, 2] - stream.cur[:, 1])[:, None])[:, :, None]
                X[:, :, _g1_cols] = np.where(_g1ok, _g1w, _g1v)
                _g1_n["steps"] += 1
                if step % 1000 == 0:
                    print(f"        vaug: lane |b| mean {float(np.abs(_g1_b).mean()):.3f} (sd units), lanes on/off so far "
                          f"{_g1_n['on']}/{_g1_n['off']}", flush=True)
            if keep_cols is not None: X = X[..., keep_cols]
            # Family conditioning, which the seqrun path never applied. --fam-norm changes no
            # feature count, so it was a SILENT NO-OP here: every FAM_norm arm was a baseline
            # run wearing a family tag. --family-input appends 4 one-hot dims, so it crashed
            # instead ("tensor a (112) must match tensor b (116)") because mu/sd were widened
            # at train.py:849 while X never was. Both are applied in the same order xprep uses.
            # stream.seq is the per-lane sequence id, which is what indexes `fam`.
            if fnorm is not None or a.family_input:
                _sidx = np.asarray(stream.seq, dtype=np.int64)
                if fnorm is not None: X = famnorm.apply(X, fam[_sidx], *fnorm)
                X = _fam(X, _sidx)
            if a.state_reset:
                # Periodic reset WITHIN a sequence. Chunks are a.chunk rows, so a lane crosses a
                # boundary whenever its running row count reaches a multiple of N. Applied BEFORE the
                # forward, so the chunk after a boundary starts from a zero state -- exactly how the
                # served driver would be driven if it carries the same grid.
                _rowpos += X.shape[1]                # rows per lane this chunk (X is (G, L, F)); was X.shape[0] = lanes -- never fired
                _hit = np.where(_rowpos % a.state_reset == 0)[0]
                if len(_hit):
                    h = h.clone(); h[:, _hit] = 0.0
            if a.train_reset_p > 0:
                # add_z15.py: TRAIN-ONLY stochastic state reset. Lanes starting a new sequence this step are already at zero
                # (excluded); the running-WP statistics are untouched, so the objective stays whole-sequence.
                _tr_hit = np.flatnonzero((_tr_rng.random(a.groups) < a.train_reset_p) & ~reset)
                _tr_lanes += int((~reset).sum())
                if len(_tr_hit):
                    h = h.clone(); h[:, _tr_hit] = 0.0
                    if _sd_h is not None: _sd_h = _sd_h.clone(); _sd_h[:, _tr_hit] = 0.0
                    _tr_n += len(_tr_hit)
                if step % 1000 == 0:
                    print(f"        train-reset: {_tr_n} resets over {_tr_lanes} within-sequence lane-chunks "
                          f"({_tr_n / max(_tr_lanes, 1):.3f})", flush=True)
            x = torch.from_numpy(X).permute(1, 0, 2).to(dev)
            if _sa_cols is not None:
                # PER-SEQUENCE RANDOM AFFINE (offline/patches/add_seqaffine.py) on the price-shape
                # columns, in raw units around the column mean: x' = mu + g*(x - mu) + b*sd, one
                # (g, b) per lane for the whole sequence. x is (T,G,F); core normalises afterwards.
                _g = torch.from_numpy(_sa_g).to(dev).unsqueeze(0); _b = torch.from_numpy(_sa_b).to(dev).unsqueeze(0)
                _mu = core.mu[_sa_cols_t]; _sd = core.sd[_sa_cols_t]
                x = x.clone(); x[..., _sa_cols_t] = _mu + _g * (x[..., _sa_cols_t] - _mu) + _b * _sd
            if a.in_noise > 0:
                _e = torch.randn_like(x)
                if a.in_noise_hl > 0:
                    # causal AR(1) filter along the chunk: one (T x T) Toeplitz matmul, cached per T
                    _T = x.shape[0]
                    if _nk.get("T") != _T:
                        _al = 0.5 ** (1.0 / a.in_noise_hl)
                        _tt = torch.arange(_T, device=x.device, dtype=x.dtype)
                        _K = torch.tril(_al ** (_tt[:, None] - _tt[None, :])) * (1.0 - _al * _al) ** 0.5
                        _nk.update(T=_T, K=_K)
                    _e = (_nk["K"] @ _e.reshape(_T, -1)).reshape(x.shape)
                x = x + a.in_noise * core.sd * _e                        # sd buffer: noise in raw units = SIGMA train-sd
            if a.block_drop > 0 and torch.rand(()).item() < a.block_drop:
                _B = [(0, 22), (22, 44), (44, 48), (48, 52), (52, 74), (74, 96), (96, 100), (100, 104), (104, 112)]
                lo, hi = _B[int(torch.randint(len(_B), ()).item())]
                x = x.clone(); x[..., lo:hi] = 0.0
            y = torch.from_numpy(Yc).to(dev)
            w = torch.from_numpy(W).to(dev)
            # teacher targets for this chunk, hoisted so the Pearson and aux terms can use them
            tt = torch.from_numpy(_T_out).to(dev) if (a.distill and _T_out is not None) else None
            if getattr(core, "bidir", False):
                h = h.clone(); h[1:2 * core.layers:2] = 0.0       # backward direction restarts each chunk
            _sam_eps = None
            if a.sam > 0 and _sam_g is not None:
                # SAM ascent along the previous step's gradient (offline/patches/add_sam.py)
                with torch.no_grad():
                    _sam_eps = [a.sam * g / _sam_gn for g in _sam_g]
                    for q_, e_ in zip(_sam_params, _sam_eps): q_.add_(e_)
            _z14_wm = _z14_materialise(core, _z14, grad=True) if _z14 is not None else None   # add_z14.py
            _wpert = (a.wdrop > 0 or a.wnoise > 0)
            if _wpert:
                # DropConnect / weight noise on the recurrent matrix, one draw per chunk
                # (offline/patches/add_wdrop_cf.py). In place under no_grad, so cuDNN reads the
                # perturbed values through the same storage; restored right after backward.
                if isinstance(core.gru, torch.nn.ModuleList):
                    raise SystemExit("--wdrop/--wnoise need a single GRU (not --split-towers)")
                _whh = core.gru.weight_hh_l0
                _raw_whh = _whh.detach().clone()
                with torch.no_grad():
                    if a.wdrop > 0:
                        _wmask = (torch.rand_like(_whh) >= a.wdrop).to(_whh.dtype) / (1.0 - a.wdrop)
                        _whh.mul_(_wmask)
                    else:
                        _wmask = None
                        _whh.add_(a.wnoise * _raw_whh.abs().mean() * torch.randn_like(_whh))
            _need_h = (a.ns_reg > 0 or a.ar_reg > 0 or a.tar_h > 0)
            # the GRU's own carried state, i.e. h_{t-1} for the first row of this chunk, so the
            # first difference is against the previous chunk instead of being silently dropped
            h_carry = h[0].detach() if _need_h else None
            _h_in = h.detach()
            if a.learn_h0 and _h0_lanes is not None and len(_h0_lanes):
                _h_in = _h_in.clone(); _h_in[:, _h0_lanes] = core.init_state(len(_h0_lanes), dev)   # graph to h0 survives the detach
            if _need_h:
                out, h, _hs = core(x, _h_in, return_states=True)
            else:
                out, h = core(x, _h_in)
            o = out.permute(1, 0, 2)
            if a.lattice_head:
                # LATTICE HEAD, seqrun path. The whole lattice objective lived only in the
                # windowed loop (train.py:1414-1424), so under --loss seqrun -- the only loss
                # any current recipe uses -- `--lattice-head K` built a (2K+1)-wide head and
                # then `pr = o[..., :2]` used two arbitrary logits as predictions. The
                # cross-entropy and the w-tilted readout never ran. S2B_dp_lat8_s801 was that.
                # The head is SINGLE-signal (k_of = (y0-y1)/2q), so it is incompatible with
                # --two-head and reads out as (s, -s) like the other single-signal heads.
                _K = a.lattice_head
                _qs = _BASES_T.index_select(0, torch.as_tensor(
                    fam[np.asarray(stream.seq, dtype=np.int64)], device=dev, dtype=torch.long))
                _sig = _lat.readout(o, _qs, _K)                  # (G,L) in target units
                pr = torch.stack([_sig, -_sig], -1)
            elif a.two_head:
                pr = o[..., :2]                      # predict t0 and t1 independently
            elif a.common_head:
                sg, cc = o[..., 0], o[..., 1]
                pr = torch.stack([sg + cc, -sg + cc], -1)
            else:
                sg = o[..., 0]; pr = torch.stack([sg, -sg], -1)
            if a.soft_clip:
                pr = 2.0 * torch.tanh(pr / 2.0)
            # A pure correlation objective leaves the output SCALE unconstrained, and it
            # drifts: the export calibration fell monotonically 14.4 -> 0.275 over three
            # epochs on the run without this term. The reference implementation carries
            # the weighted MSE at 0.1 for exactly that reason.
            y_anchor = y
            if a.anchor_target == "t":
                if tt is None: raise SystemExit("--anchor-target t needs --distill")
                y_anchor = tt.clamp(-2.0, 2.0)          # research G5: the anchor pins to the teacher
            if a.smooth_anchor > 0:
                k = 2 * a.smooth_anchor + 1
                # centred box filter along time; (G,L,2) -> (G*2,1,L) -> pool -> back
                yt = y.permute(0, 2, 1).reshape(-1, 1, y.shape[1])
                ys = torch.nn.functional.avg_pool1d(yt, kernel_size=k, stride=1,
                                                    padding=a.smooth_anchor, count_include_pad=False)
                y_anchor = ys.reshape(y.shape[0], 2, y.shape[1]).permute(0, 2, 1).contiguous()
                if a.smooth_both:
                    y = y_anchor
            w_anchor = w
            if a.trim > 0:
                # research Z6 (add_signals.py): trimmed anchor. The top-Q fraction of this step's supervised rows by
                # max_k |p_k - clip(y_k)| gets anchor weight 0; the running Pearson and distillation terms still see them.
                with torch.no_grad():
                    _err = (pr.detach() - y_anchor).abs().max(-1).values          # (G,L)
                    _val = w > 0
                    _k = int(round(a.trim * int(_val.sum())))
                    if _k > 0:
                        _cut = torch.topk(_err[_val], _k).values[-1]
                        w_anchor = w * (_err < _cut).to(w.dtype)
                        if step % 1000 == 0:
                            print(f"        trim: {int((_val & (_err >= _cut)).sum())} of {int(_val.sum())} rows left the anchor "
                                  f"(cut |p - y| >= {float(_cut):.3f})", flush=True)
            if a.anchor == "mse" and _b3_fl is not None:
                # REALISABILITY FLOOR (add_b3.py, research B3): the lookahead teacher saw this row's future; a causal student that
                # fits a trained-on row better than K x that teacher is recalling the sequence, so the anchor stops pulling there.
                # Same weights and clipped target as weighted_mse; the teacher residual is a constant (no gradient through it).
                _b3yc = y_anchor.clamp(-2.0, 2.0)
                _b3w = _b3yc.abs() * w_anchor.unsqueeze(-1)
                _b3es = ((clip_ste(pr) if a.anchor_clip else pr) - _b3yc) ** 2
                _b3et = ((tt.clamp(-2.0, 2.0) if a.distill_clip else tt) - _b3yc).detach() ** 2
                _b3h = torch.clamp_min(_b3es - a.anchor_floor * _b3et, 0.0)
                anchor = (_b3w * _b3h).sum() / _b3w.sum().clamp_min(1e-6)
                with torch.no_grad():
                    _b3_fl["steps"] += 1; _b3_fl["mass"] += float(_b3w.sum()); _b3_fl["off_mass"] += float((_b3w * (_b3h <= 0)).sum())
                    _b3_fl["plain"] += float((_b3w * _b3es).sum() / _b3w.sum().clamp_min(1e-6)); _b3_fl["floored"] += float(anchor)
                if step == 1 or step % 1000 == 0:
                    print(f"        anchor-floor: {100.0 * _b3_fl['off_mass'] / max(1e-6, _b3_fl['mass']):.1f}% of anchor mass below the floor so far; "
                          f"anchor plain {_b3_fl['plain'] / _b3_fl['steps']:.4f} -> floored {_b3_fl['floored'] / _b3_fl['steps']:.4f}", flush=True)
            elif a.anchor == "mse":
                anchor = weighted_mse(clip_ste(pr) if a.anchor_clip else pr, y_anchor, w_anchor)   # --anchor-clip: STE clip (add_losses2.py)
            else:
                # robust anchors: |y|-weighted Huber (delta) or L1 -- the conditional MEDIAN/trimmed mean
                # instead of the mean; a regulariser against memorising noisy rows
                wm = y.abs() * w_anchor.unsqueeze(-1); err = pr - y_anchor
                if a.anchor == "l1":
                    pen = err.abs()
                else:
                    d = a.huber_delta; pen = torch.where(err.abs() <= d, 0.5 * err * err / d, err.abs() - 0.5 * d)
                anchor = (wm * pen).sum() / wm.sum().clamp_min(1e-6)
            _rn = None
            if a.resnorm_w > 0 or a.resnorm_ed > 0:
                # research H12 (add_h12_resnorm.py): terms on z = (target - p) / sigma(x) from the train-only log-sigma head
                _rn_s = getattr(core, '_resnorm_out', None)
                if _rn_s is None: raise SystemExit('--resnorm-*: the core produced no log-sigma (needs --preset refgru, training mode)')
                core._resnorm_out = None
                _rn_fam = (torch.as_tensor([stream.family(int(_i)) for _i in stream.cur[:, 0].tolist()], device=dev, dtype=torch.long)
                           if (a.resnorm_famnorm or a.resnorm_ed_fam) else None)
                _rn_y = (y_anchor, w_anchor, True) if a.resnorm_target == 'y' else (tt, w, False)
                _rn = resnorm_terms(pr, _rn_s.permute(1, 0, 2), _rn_y[0], _rn_y[1], beta=a.resnorm_beta, weighting=a.resnorm_weighting,
                                    fam=_rn_fam, famnorm=a.resnorm_famnorm, ed=a.resnorm_ed > 0, ed_fam=a.resnorm_ed_fam,
                                    ed_grad_p=a.resnorm_ed_p, clip_y=_rn_y[2])
                if a.resnorm_mode == 'replace' and a.resnorm_w > 0:
                    _rn_anchor_plain = float(anchor.detach())
                    anchor = a.resnorm_w * _rn['nll']           # the NLL takes the anchor's slot and (1 - pearson-weight) weight
            if a.cosine_loss:
                wm = (y.abs() * w.unsqueeze(-1)).reshape(-1)
                a1 = pr.reshape(-1); t1 = y.reshape(-1)
                S = wm.sum().clamp_min(1e-6)
                mp = (wm * a1).sum() / S; mt = (wm * t1).sum() / S
                cov = (wm * (a1 - mp) * (t1 - mt)).sum() / S
                sdp = ((wm * (a1 - mp) ** 2).sum() / S).clamp_min(1e-8).sqrt()
                corr_term = 1.0 - cov / sdp / y.abs().mean().clamp_min(1e-6)
            else:
                if a.corr_target == "t" and tt is not None:
                    corr_term = rstat.loss(tt.clamp(-2.0, 2.0), pr, w, gamma=a.focal, pooled=a.global_pool)
                else:
                    corr_term = rstat.loss(y, pr, w, gamma=a.focal, pooled=a.global_pool, rescale=a.wp_rescale)
            _pw, _aw = _z18_pw(step)                     # add_z18.py: default = (a.pearson_weight, 1.0 - a.pearson_weight)
            if dens_h:
                # forward mid change of each instrument, built from the raw chunk itself. Valid only
                # where the horizon fits inside the chunk, hence the mask; standardised per batch so
                # the auxiliary term cannot dominate through scale alone.
                base_out = 2 + (3 if ssl_w is not None else 0)
                if _z37 is None:
                    mids = [0.5 * (x[..., b:b + 11].mean(-1) + x[..., b + 11:b + 22].mean(-1)) for b in (0, 52)]
                else:
                    # --dens-mid (add_z37.py): the EXACT mid of i0 / i1 in half-ticks instead of the p-column means (which track
                    # the spread: research Z37). Unknown rows carry 0 here; _zc counts them so a window holding one is masked.
                    _zm = torch.from_numpy(_z37m).to(dev).permute(1, 0, 2)                 # (T, G, 2) int16
                    _zk = _zm != _Z37_SENT
                    _zc = torch.cat([torch.zeros_like(_zk[:1], dtype=torch.int32),
                                     torch.cumsum((~_zk).to(torch.int32), 0)], 0)             # (T+1, G, 2) unknown-row prefix counts
                    mids = [torch.where(_zk[..., c], _zm[..., c].to(x.dtype), torch.zeros((), dtype=x.dtype, device=dev))
                            for c in (0, 1)]
                aux_loss = 0.0
                _z55x = _z37 is not None and a.dens_cross_chunk   # add_z55.py: mids carry max(dens_h) rows past the chunk
                for hi, hh in enumerate(dens_h):
                    _hk = abs(hh)   # add_sx.py: |h|; h < 0 = BACKWARD target mid[t] - mid[t-|h|] read at row t (same tensors as forward |h|)
                    if x.shape[0] <= _hk and not _z55x: continue
                    for mi, mid in enumerate(mids):
                        if _z55x:
                            tgt = mid[hh:hh + x.shape[0]] - mid[:x.shape[0]]          # (T, G): every chunk row (add_z55.py)
                        else:
                            tgt = mid[_hk:] - mid[:-_hk]                     # (T-|h|, G)
                        if _z37 is None:
                            tgt = tgt / tgt.std().clamp_min(1e-6)
                        elif _z55x:
                            _zok = ((_zc[hh + 1:hh + 1 + x.shape[0], :, mi] - _zc[:x.shape[0], :, mi]) == 0).to(tgt.dtype)
                        else:
                            _zok = ((_zc[_hk + 1:, :, mi] - _zc[:-_hk - 1, :, mi]) == 0).to(tgt.dtype)   # 1 = all |h|+1 rows known
                        if _z37 is not None:
                            if _z38u is not None:                               # --dens-fam-mult (add_z38.py)
                                tgt = tgt * _z38u[0][mi]; _zok = _zok * _z38u[1][mi]                     # (T-h, G) x (G,)
                            _zn = _zok.sum()
                            _zmu = (tgt * _zok).sum() / _zn.clamp_min(1.0)
                            _zsd = ((((tgt - _zmu) ** 2) * _zok).sum() / (_zn - 1.0).clamp_min(1.0)).sqrt()
                            tgt = tgt * _zok / _zsd.clamp_min(1e-6)
                        # o is (G, T, n_out) after the permute above, while mid is (T, G):
                        # take the column, trim to the valid horizon, and transpose back to (T-h, G)
                        pred_a = (o[:, _hk:, base_out + 2 * hi + mi] if hh < 0 else
                                  o[:, :tgt.shape[0], base_out + 2 * hi + mi]).permute(1, 0)   # add_sx.py: backward targets read at row t
                        if _z37 is None:
                            aux_loss = aux_loss + ((pred_a - tgt) ** 2).mean()
                        else:
                            aux_loss = aux_loss + (((pred_a - tgt) ** 2) * _zok).sum() / _zn.clamp_min(1.0)
                loss = _pw * corr_term + _aw * anchor + a.densaux * aux_loss      # add_z18.py: scheduled weights
            else:
                loss = _pw * corr_term + _aw * anchor                             # add_z18.py: scheduled weights
            if _z39 is not None:
                # --paux (add_z39.py, research Z39): K exact aux targets from the label's own construction, one head row each
                # (channels _paux_base.. of n_out, dropped at export). Per channel: NaN (unknown / censored / family x0) entries
                # are masked; the valid ones are centred and divided by their batch std (Bessel); masked MSE averaged over the
                # valid entries; the K terms are summed (the --densaux convention) and weighted by --paux-weight.
                _z9t = torch.from_numpy(_z39m).to(dev)                                            # (G, T, K), NaN = masked
                _z9ok = ~torch.isnan(_z9t)
                if _z39u is not None:
                    _z9t = _z9t * _z39u[0]; _z9ok = _z9ok & (_z39u[1] > 0)                       # per-lane family multiplier
                _z9okf = _z9ok.to(_z9t.dtype)
                _z9t = torch.where(_z9ok, _z9t, torch.zeros((), dtype=_z9t.dtype, device=dev))
                _z9n = _z9okf.sum((0, 1))                                                          # (K,) valid entries
                _z9mu = _z9t.sum((0, 1)) / _z9n.clamp_min(1.0)
                _z9sd = ((((_z9t - _z9mu) ** 2) * _z9okf).sum((0, 1)) / (_z9n - 1.0).clamp_min(1.0)).sqrt()
                _z9z = (_z9t - _z9mu) * _z9okf / _z9sd.clamp_min(1e-6)
                _z9p = o[:, :_z9t.shape[1], _paux_base:_paux_base + _paux_k]
                _z9loss = (((_z9p - _z9z) ** 2) * _z9okf).sum((0, 1)) / _z9n.clamp_min(1.0)      # (K,) per-channel masked MSE
                loss = loss + a.paux_weight * _z9loss.sum()
                if step % 1000 == 0:
                    print(f"        paux: per-channel loss {[round(v, 4) for v in _z9loss.detach().cpu().tolist()]} (1 - R2 per "
                          f"channel on standardised targets)", flush=True)
            if _rn is not None:
                # research H12 (add_h12_resnorm.py): add-mode NLL and the energy-distance calibration term
                if a.resnorm_mode == 'add' and a.resnorm_w > 0:
                    loss = loss + a.resnorm_w * _rn['nll']
                if a.resnorm_ed > 0:
                    loss = loss + a.resnorm_ed * _rn['ed']
                if step == 1 or step % 1000 == 0:
                    _rs = _rn['stats']
                    print(f"        h12: nll {float(_rn['nll'].detach()):+.5f} x {a.resnorm_w:g} ({a.resnorm_mode}, beta {a.resnorm_beta:g}, target "
                          f"{a.resnorm_target}{', famnorm' if a.resnorm_famnorm else ''})"
                          + (f"  energy {_rs['ed']:.5f} x {a.resnorm_ed:g}" if a.resnorm_ed > 0 else "")
                          + f"  mean_w z^2 {_rs['z2_w']:.4f}  mean_w log-sigma {_rs['logsig_w']:+.4f}  sigma p10/p90 {_rs['sig_p10']:.3f}/{_rs['sig_p90']:.3f}"
                          + f"  | pearson {float(corr_term.detach()):.5f} x {_pw:g}  anchor {float(anchor.detach()):.5f} x {_aw:g}"
                          + (f" (plain MSE anchor would be {_rn_anchor_plain:.5f})" if (a.resnorm_mode == 'replace' and a.resnorm_w > 0) else ""), flush=True)
            if a.hetero > 0:
                # heteroscedastic anchor (research Z5, add_losses2.py): the log-variance head reweights the regression
                # residual row by row; ADDED at weight W, the plain anchor above is kept
                _het = getattr(core, "_hetero_out", None)
                if _het is None: raise SystemExit("--hetero needs --preset refgru (RefGRUCore.hetero_head) and --loss seqrun")
                loss = loss + a.hetero * hetero_nll(pr, _het.permute(1, 0, 2), y_anchor, w)
            if a.xlane_pw > 0:
                # cross-lane pooled Pearson of THIS chunk, one centre across the 64 lanes (research Z5, add_losses2.py)
                loss = loss + a.xlane_pw * xlane_wp_loss(pr, y, w)
            _z59_pk = []                                   # add_z59.py: detached per-head predictions for rstat_mh
            if a.mh > 0:
                # TabM (research Z59): mean over heads of the SAME pearson + anchor terms, each head on its own running stats
                _mho = core._mh_out.permute(1, 0, 2, 3); core._mh_out = None             # (G,T,K,n_out)
                _mhl = 0.0
                for _k59, _rk in enumerate(rstat_mh):
                    _pk = _mho[:, :, _k59, :2]
                    if a.soft_clip: _pk = 2.0 * torch.tanh(_pk / 2.0)
                    if a.corr_target == "t" and tt is not None:
                        _ck59 = _rk.loss(tt.clamp(-2.0, 2.0), _pk, w, gamma=a.focal, pooled=a.global_pool)
                    else:
                        _ck59 = _rk.loss(y, _pk, w, gamma=a.focal, pooled=a.global_pool, rescale=a.wp_rescale)
                    _ak59 = weighted_mse(clip_ste(_pk) if a.anchor_clip else _pk, y_anchor, w_anchor)
                    _mhl = _mhl + _pw * _ck59 + _aw * _ak59
                    _z59_pk.append(_pk.detach())
                loss = loss + a.mh_w * _mhl / len(rstat_mh)
            if a.y_weight != 1.0:
                loss = a.y_weight * loss        # scale the y-based terms; 0 = teacher-only student
            if a.grid_head > 0 and a.grid_ce > 0:
                # two-hot cross-entropy on the value grid, weighted like the metric (|clip(y)| x row weight)
                _lg = core._grid_logits.permute(1, 0, 2, 3)                      # (B,T,2,K)
                _v = core.grid_v; _K = _v.shape[0]; _d = float(_v[1] - _v[0])
                _yc = y.clamp(-2.0, 2.0); _pos = (_yc + 2.0) / _d
                _lo = _pos.floor().clamp(0, _K - 2).long(); _fr = (_pos - _lo.to(_pos.dtype)).clamp(0, 1)
                _logp = torch.log_softmax(_lg, -1)
                _ce = -((1 - _fr) * _logp.gather(-1, _lo.unsqueeze(-1)).squeeze(-1) + _fr * _logp.gather(-1, (_lo + 1).unsqueeze(-1)).squeeze(-1))
                _wce = _yc.abs() * w.unsqueeze(-1)
                loss = loss + a.grid_ce * (_wce * _ce).sum() / _wce.sum().clamp_min(1e-6)
            if a.hlg > 0:
                # HL-Gauss (research Z58 C1 / Z59): |clip(y)| x row-weighted CE against Gaussian-smoothed bins of clip(y)
                _lg59 = core._hlg_logits.permute(1, 0, 2, 3); core._hlg_logits = None   # (G,T,2,K)
                _K59 = _lg59.shape[-1]
                _e59 = torch.linspace(-2.2, 2.2, _K59 + 1, device=_lg59.device, dtype=_lg59.dtype)
                _s59 = a.hlg_sigma * (4.4 / _K59)
                _yc59 = y.clamp(-2.0, 2.0)
                _cdf59 = 0.5 * (1.0 + torch.erf((_e59 - _yc59.unsqueeze(-1)) / (_s59 * 2.0 ** 0.5)))   # (G,T,2,K+1)
                _q59 = _cdf59[..., 1:] - _cdf59[..., :-1]
                _q59 = _q59 / _q59.sum(-1, keepdim=True).clamp_min(1e-12)
                _ce59 = -(_q59 * torch.log_softmax(_lg59, -1)).sum(-1)                      # (G,T,2)
                _w59 = _yc59.abs() * w.unsqueeze(-1)
                loss = loss + a.hlg * (_w59 * _ce59).sum() / _w59.sum().clamp_min(1e-6)
            if a.aux_next > 0:
                # next-row feature prediction on the rows whose successor is in this chunk; masked by the
                # row weights of the successor row (w is (B,T))
                _xn = core._x_std[1:].detach()                                  # (T-1,B,F) targets
                _pn = core.aux_next_head(core._gru_out[:-1])                    # (T-1,B,F)
                _mn = (w[:, 1:] > 0).to(_pn.dtype).permute(1, 0).unsqueeze(-1)  # (T-1,B,1)
                _hub = torch.nn.functional.huber_loss(_pn, _xn, reduction="none", delta=1.0)
                loss = loss + a.aux_next * (_hub * _mn).sum() / (_mn.sum() * _xn.shape[-1]).clamp_min(1.0)
            if a.mask_aux > 0:
                # research Z6 (add_signals.py): train-only head predicts P(is_scored) from the GRU output; BCE on the
                # post-warm-up rows whose source cache carries a mask (stream.MV_out), weight W. The mask depends on
                # the label path (research T), so the state is pushed to carry the label-relevant regime.
                _mv = torch.from_numpy(stream.MV_out).to(dev)                              # (G,L) 1 = supervised
                if float(_mv.sum()) > 0:
                    _mt = torch.from_numpy(stream.M_out).to(dev)                           # (G,L) is_scored 0/1
                    _ml = core.mask_head(core._gru_out).squeeze(-1).permute(1, 0)          # (G,L) logits
                    _bce = torch.nn.functional.binary_cross_entropy_with_logits(_ml, _mt, reduction="none")
                    loss = loss + a.mask_aux * (_bce * _mv).sum() / _mv.sum()
                    if step % 1000 == 0:
                        with torch.no_grad():
                            _mp = torch.sigmoid(_ml); _np_ = (_mt * _mv).sum().clamp_min(1.0); _nn_ = ((1 - _mt) * _mv).sum().clamp_min(1.0)
                            print(f"        mask-aux: bce {float((_bce * _mv).sum() / _mv.sum()):.4f} on {int(_mv.sum())} rows "
                                  f"({float((_mt * _mv).sum() / _mv.sum()):.3f} scored); mean p scored {float((_mp * _mt * _mv).sum() / _np_):.3f} "
                                  f"vs unscored {float((_mp * (1 - _mt) * _mv).sum() / _nn_):.3f}", flush=True)
            if hint_core is not None:
                with torch.no_grad():
                    hint_h = hint_h.clone(); hint_h[1:2 * hint_core.layers:2] = 0.0   # backward restarts per chunk
                    _, hint_h = hint_core(x, hint_h)
                    _tg = hint_core._gru_out[..., :hint_core.hidden].permute(1, 0, 2)     # (B,T,Ht) forward direction
                _sg = core.hint_adapter(core._gru_out.permute(1, 0, 2))
                _hm = (w > 0).to(_sg.dtype)
                loss = loss + a.hint_weight * (((_sg - _tg) ** 2).mean(-1) * _hm).sum() / _hm.sum().clamp_min(1.0)
            if _sd_h is not None:
                # MEAN-TEACHER SELF-DISTILLATION (add_z9.py): the EMA copy walks the same chunk with its own state (no
                # grad, eval mode) and reads out exactly like the student; the student is pulled to it where the metric looks.
                with torch.no_grad():
                    _oe, _sd_h = ema(x, _sd_h)
                    _oe = _oe.permute(1, 0, 2)
                    if a.two_head:
                        _pe = _oe[..., :2]
                    elif a.common_head:
                        _pe = torch.stack([_oe[..., 0] + _oe[..., 1], -_oe[..., 0] + _oe[..., 1]], -1)
                    else:
                        _pe = torch.stack([_oe[..., 0], -_oe[..., 0]], -1)
                    if a.soft_clip: _pe = 2.0 * torch.tanh(_pe / 2.0)
                _sd_from = int(a.self_distill_start * a.steps) + 1
                if step >= _sd_from:
                    _wsd = y.abs() * w.unsqueeze(-1)
                    _sdt = (_wsd * (pr - _pe) ** 2).sum() / _wsd.sum().clamp_min(1e-6)
                    loss = loss + a.self_distill * _sdt
                    if step == _sd_from or step % 1000 == 0:
                        _sdv = float(_sdt.detach())
                        print(f"        self-distill: term {_sdv:.5f} (weighted rms student-EMA gap {max(_sdv, 0.0) ** 0.5:.4f}) "
                              f"x {a.self_distill}", flush=True)
            _ao = None
            if a.dualmode:
                # DUAL-MODE: the aux (non-causal) head learns y with the same objective; the causal head
                # is pulled toward its detached output (in-place lookahead distillation).
                _ao = core._aux_out.permute(1, 0, 2)[..., :2]
                _dm = (a.pearson_weight * rstat_dm.loss(y, _ao, w, gamma=a.focal, pooled=a.global_pool)
                       + (1.0 - a.pearson_weight) * weighted_mse(_ao, y, w))
                loss = loss + a.dualmode_weight * _dm
                if a.distill_weight > 0:
                    _wmd = y.abs() * w.unsqueeze(-1)
                    loss = loss + a.distill_weight * ((_wmd * (pr - _ao.detach()) ** 2).sum() / _wmd.sum().clamp_min(1e-6))
            _auxh_pairs = []
            if _auxh_base is not None and a.aux_weight > 0:
                # MULTI-HORIZON AUXILIARY HEADS, seqrun port (offline/patches/add_auxh_seqrun.py).
                # Head j at row t is correlated with the clipped target at row t+k through its own
                # RunningWP, i.e. the metric's own whole-sequence weighted Pearson; the last k rows
                # of the chunk have no in-chunk label and are skipped.
                _al = 0.0
                for _j, _k in enumerate(aux_off):
                    if y.shape[1] <= _k: continue
                    _c0 = _auxh_base + _auxh_c * _j
                    if _auxh_c == 2:
                        _pa = o[:, :-_k, _c0:_c0 + 2]
                    else:
                        _sg = o[:, :-_k, _c0]; _pa = torch.stack([_sg, -_sg], -1)
                    _ysrc = tt.clamp(-2.0, 2.0) if (a.aux_teacher and tt is not None) else y
                    _al = _al + rstat_auxh[_j].loss(_ysrc[:, _k:], _pa, w[:, _k:],
                                                    gamma=a.focal, pooled=a.global_pool)
                    _auxh_pairs.append((_j, _k, _pa.detach()))
                if _auxh_pairs:
                    loss = loss + a.aux_weight * _al / len(_auxh_pairs)
            if a.levelaux > 0:
                # Per-sequence level and scale of THIS chunk, |y|-weighted exactly as the metric is.
                # Computed over the whole chunk, so at row t it depends on rows after t: acausal, and
                # unavailable at inference, which is what makes it privileged rather than just another
                # feature. Supervised at EVERY row, so the state has to carry it forward rather than
                # the head reconstructing it once at the end.
                _b = 2 + (3 if ssl_w is not None else 0) + 2 * len(dens_h)
                wm_l = y.abs() * w.unsqueeze(-1)                      # (G,L,2)
                S_l = wm_l.sum(1).clamp_min(1e-6)                     # (G,2)
                lvl = (wm_l * y).sum(1) / S_l                         # (G,2)
                var = (wm_l * (y - lvl.unsqueeze(1)) ** 2).sum(1) / S_l
                sd_y = y.std().clamp_min(1e-6)
                t_lvl = (lvl / sd_y).detach()
                t_scl = torch.log((var.clamp_min(1e-12).sqrt() / sd_y).clamp_min(1e-6)).detach()
                lev = ((o[..., _b:_b + 2] - t_lvl.unsqueeze(1)) ** 2).mean() \
                    + ((o[..., _b + 2:_b + 4] - t_scl.unsqueeze(1)) ** 2).mean()
                loss = loss + a.levelaux * lev
            if a.lattice_head and a.lattice_wp > 0:
                # loss = seqwp(tilted readout) + W * CE, exactly as the windowed path does.
                # Labels come from RAW targets: the +-2 clip destroys the integer lattice.
                _lab = _lat.labels(Yc, _BASES[fam[np.asarray(stream.seq, dtype=np.int64)]][:, None], a.lattice_head)
                _tgt = torch.from_numpy(_lab).to(dev)
                _ce = torch.nn.functional.cross_entropy(
                    o.reshape(-1, 2 * a.lattice_head + 1), _tgt.reshape(-1))
                loss = loss + a.lattice_wp * _ce
                if step % 1000 == 0:
                    print(f"        lattice: ce {_ce.item():.4f}", flush=True)
            if a.dir_weight > 0:
                loss = loss + a.dir_weight * weighted_dir(pr, y, w, tau=a.dir_tau,
                                              resolved_only=not a.dir_all_rows)
            if a.rank_weight > 0:
                loss = loss + a.rank_weight * weighted_softrank(
                    pr, y, w, knots=a.rank_knots, tau=a.rank_tau,
                    resolved_only=not a.dir_all_rows)
            if _need_h:
                # _hs is (T,B,H): the state AFTER each row, with the carried state prepended so
                # the first difference is against the previous chunk rather than being dropped.
                _hprev = torch.cat([h_carry.unsqueeze(0), _hs[:-1]], 0)
                if a.ns_reg > 0:
                    _n = _hs.norm(dim=-1); _np = _hprev.norm(dim=-1)
                    loss = loss + a.ns_reg * ((_n - _np) ** 2).mean()
                if a.ar_reg > 0:
                    loss = loss + a.ar_reg * (_hs ** 2).mean()
                if a.tar_h > 0:
                    loss = loss + a.tar_h * ((_hs - _hprev) ** 2).mean()
            if a.tar_seq > 0:
                loss = loss + a.tar_seq * ((pr[:, 1:] - pr[:, :-1]) ** 2).mean()
            if a.zmse_weight > 0:
                # running per-sequence mean/std of the clipped target from the detached prefix
                # statistics (+ this chunk), so the regression target is the metric's own
                # standardised signal for the sequence so far
                with torch.no_grad():
                    S_, Sy_, _, Syy_ = (rstat.s[..., k] for k in range(4))
                    cur = rstat._terms(y, pr.detach(), w).sum(1)
                    S2, Sy2, Syy2 = S_ + cur[..., 0], Sy_ + cur[..., 1], Syy_ + cur[..., 3]
                    my_ = (Sy2 / S2.clamp_min(1e-6)).unsqueeze(1); sd_ = ((Syy2 / S2.clamp_min(1e-6)) - my_.squeeze(1) ** 2).clamp_min(1e-4).sqrt().unsqueeze(1)
                    z = (y - my_) / sd_
                loss = loss + a.zmse_weight * weighted_mse(pr, z, w)
            _b3_loss_y = loss if (_b3_pc is not None and a.pcgrad == "term") else None    # add_b3.py: task 1 = every term so far
            if a.distill and a.distill_weight > 0:
                # research G3: imitation schedule with the MEAN dose held at distill_weight
                _dw = a.distill_weight * ({"up": 2.0 * step / max(1, a.steps),
                                           "down": 2.0 * (1.0 - step / max(1, a.steps))}.get(a.distill_ramp, 1.0))
                if _z20_t1 and step > _z20_t1: _dw = a.distill_w2       # add_z20.py: --distill-step / --distill-w2
                # teacher-matching term with the metric's own weights (|clip(y)| x row factor), so the
                # student copies the ensemble where the metric looks
                if a.distill_wsrc == "t":
                    wm = tt.abs() * w.unsqueeze(-1)
                elif a.distill_wsrc == "geo":
                    wm = (y.abs() * tt.abs()).clamp_min(0.0).sqrt() * w.unsqueeze(-1)
                else:
                    wm = y.abs() * w.unsqueeze(-1)
                if _b3_kdm is not None:                                             # add_b3.py: --distill-fam-mult, per lane
                    _b3f = _b3_lane_fam()
                    wm = wm * torch.from_numpy(np.where(_b3f >= 0, _b3_kdm[np.clip(_b3f, 0, 3)], 1.0).astype(np.float32)).to(dev)[:, None, None]
                if a.noise_w > 0:
                    _r2 = ((y - tt) ** 2).mean(-1, keepdim=True)
                    wm = wm / (1.0 + _r2 / a.noise_w)
                if _z20_lookw is not None:                                      # add_z20.py: --distill-look
                    wm = wm * (_z20_lookw if _z20_lookw.shape[1] == wm.shape[1] else _z20_lookw[:, :wm.shape[1]])
                if a.distill_dis > 0:                                           # add_z20.py: --distill-dis
                    _z20_wm0 = wm
                    _r2d = ((y - tt.clamp(-2.0, 2.0)) ** 2).mean(-1, keepdim=True)
                    wm = wm * (1.0 + a.distill_dis * _r2d / (a.distill_dis_tau + _r2d))
                if a.distill_mode == "corr":
                    # PER TARGET, matching how the metric pools each channel separately.
                    cs = []
                    for _k in range(pr.shape[-1]):
                        _w = wm[..., _k]; _sw = _w.sum().clamp_min(1e-6)
                        _p = pr[..., _k]; _t = tt[..., _k]
                        _a = _p - (_w*_p).sum()/_sw
                        _b = _t - (_w*_t).sum()/_sw
                        _cov = (_w*_a*_b).sum()/_sw
                        _vp = (_w*_a*_a).sum()/_sw
                        _vt = (_w*_b*_b).sum()/_sw
                        cs.append(_cov / (_vp*_vt).clamp_min(1e-12).sqrt())
                    loss = loss + _dw * (1.0 - torch.stack(cs).mean())
                elif _z20_on:                                                   # add_z20.py: same term, flag-modified weight/target
                    _ttk = tt.clamp(-2.0, 2.0) if a.distill_clip else tt
                    _z20_kd = (wm * (pr - _ttk) ** 2).sum() / wm.sum().clamp_min(1e-6)
                    loss = loss + _dw * _z20_kd
                    if step == 1 or step % 1000 == 0:
                        _z20_msg = f"        z20-kd: term {_z20_kd.item():.5f} x {_dw:g}"
                        if _z20_lookah is not None:
                            _la = _z20_lookah if _z20_lookah.shape[1] == wm.shape[1] else _z20_lookah[:, :wm.shape[1]]
                            _z20_msg += f"  mass-weighted mean lookahead {((wm * _la).sum() / wm.sum().clamp_min(1e-6)).item():.1f} rows"
                        if a.distill_dis > 0:
                            _z20_msg += f"  dis mean boost {(wm.sum() / _z20_wm0.sum().clamp_min(1e-6)).item():.3f}"
                        print(_z20_msg, flush=True)
                else:
                    loss = loss + _dw * ((wm * (pr - tt) ** 2).sum() / wm.sum().clamp_min(1e-6))
            if a.distill2 and a.distill2_weight > 0:
                tt2 = torch.from_numpy(_T2_out).to(dev)
                wm2 = y.abs() * w.unsqueeze(-1)
                loss = loss + a.distill2_weight * ((wm2 * (pr - tt2) ** 2).sum() / wm2.sum().clamp_min(1e-6))
            if a.elr > 0 and float(stream.EV_out.sum()) > 0:
                # ELR (add_z12.py): pull toward the per-row temporal ensemble of the model's OWN earlier-pass predictions,
                # on the rows that have one, with the metric's weights. E_out is 0 where EV_out is 0, so no NaN can enter.
                _te = torch.from_numpy(stream.E_out).to(dev)
                _we = y.abs() * (w * torch.from_numpy(stream.EV_out).to(dev)).unsqueeze(-1)
                _elt = (_we * (pr - _te) ** 2).sum() / _we.sum().clamp_min(1e-6)
                loss = loss + a.elr * _elt
                _elr_n += 1
                if _elr_n == 1 or step % 1000 == 0:
                    _elv = float(_elt.detach())
                    _cov = float((torch.from_numpy(stream.EV_out).to(dev) * (w > 0)).sum()) / max(float((w > 0).sum()), 1.0)
                    print(f"        elr: term {_elv:.5f} (weighted rms gap to self-target {max(_elv, 0.0) ** 0.5:.4f}) x {a.elr}, "
                          f"coverage {_cov:.1%} of the chunk's selected rows", flush=True)
            if core2 is not None:
                out2, h2 = core2(x, h2.detach())
                o2 = out2.permute(1, 0, 2)
                pr2 = o2[..., :2] if a.two_head else torch.stack([o2[..., 0], -o2[..., 0]], -1)
                if a.soft_clip: pr2 = 2.0 * torch.tanh(pr2 / 2.0)
                loss2 = (a.pearson_weight * rstat2.loss(y, pr2, w, gamma=a.focal, pooled=a.global_pool)
                         + (1.0 - a.pearson_weight) * weighted_mse(pr2, y, w))
                # NCL: subtract the |y|-weighted variance across members, which is exactly the
                # ensemble's error-correlation term. Members are pushed apart only where the metric
                # looks (weight w * |clip(y)|), so the penalty cannot buy diversity on unscored rows.
                # The penalty must be BOUNDED: an unweighted variance bonus is unbounded above, so the
                # members simply inflate their scale (lambda 1.0 drove the loss to -5e5 and lambda 3.0
                # produced degenerate, anti-correlated members). Penalise their weighted CORRELATION
                # instead, which lives in [-1, 1] and is exactly the quantity the ensemble ceiling
                # depends on.
                wm = (y.abs() * w.unsqueeze(-1)).reshape(-1)
                a1 = pr.reshape(-1); a2 = pr2.reshape(-1)
                S = wm.sum().clamp_min(1e-6)
                m1 = (wm * a1).sum() / S; m2 = (wm * a2).sum() / S
                v1 = ((wm * (a1 - m1) ** 2).sum() / S).clamp_min(1e-8)
                v2 = ((wm * (a2 - m2) ** 2).sum() / S).clamp_min(1e-8)
                corr = (wm * (a1 - m1) * (a2 - m2)).sum() / S / torch.sqrt(v1 * v2)
                pen = corr if a.ncl_target >= 1.0 else torch.relu(corr - a.ncl_target)
                lam = a.ncl
                if a.ncl_warm > 0:
                    _f = step / max(1, a.steps)
                    lam = a.ncl * min(1.0, max(0.0, (_f - a.ncl_warm) / max(1e-6, a.ncl_warm)))
                loss = loss + loss2 + lam * pen
            if _z48 is not None:
                # add_z48.py: class-balanced CE of the adversary on the head input (GRL: the core gets -lam x the gradient)
                _z48h = core._z48_hin.permute(1, 0, 2)[:, ::a.dann_stride]                  # (G, T', D)
                _z48y = torch.as_tensor(_z48["lab"][np.asarray(stream.cur[:, 0], np.int64)], device=dev)    # (G,) lane labels
                _z48v = (w[:, ::a.dann_stride] > 0) & (_z48y[:, None] >= 0)
                if bool(_z48v.any()):
                    _z48p_ = step / max(1, a.steps)
                    _z48lam = float(2.0 / (1.0 + np.exp(-10.0 * _z48p_)) - 1.0)
                    _z48x = _z48["grl"].apply(_z48h[_z48v], _z48lam)
                    _z48t = _z48y[:, None].expand(-1, _z48h.shape[1])[_z48v]
                    _z48o = _z48["adv"](_z48x)
                    _z48ce = torch.nn.functional.cross_entropy(_z48o, _z48t, weight=_z48["cw"])
                    loss = loss + a.dann * _z48ce
                    with torch.no_grad():
                        _z48["n"] += 1; _z48["ce"] += float(_z48ce)
                        _z48["hit"] += float((_z48o.argmax(-1) == _z48t).float().mean()); _z48["rows"] += int(_z48v.sum())
                    if step % 1000 == 0:
                        print(f"        z48 dann: lam {_z48lam:.3f}  adversary CE {_z48['ce'] / max(1, _z48['n']):.4f} "
                              f"(chance {float(np.log(_z48['K'])):.4f} unweighted)  acc {_z48['hit'] / max(1, _z48['n']):.3f}  "
                              f"rows/step {_z48['rows'] / max(1, _z48['n']):.0f}", flush=True)
                        _z48["n"] = 0; _z48["ce"] = 0.0; _z48["hit"] = 0.0; _z48["rows"] = 0
                _z48["opt"].zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            if opt2 is not None: opt2.zero_grad(set_to_none=True)
            if _b3_pc is None:
                loss.backward()
            else:
                _b3_pcgrad_backward(loss, o, _b3_loss_y, step)   # add_b3.py: per-task gradients through o, PCGrad, .grad = projected sum
            if _z14_wm is not None: _z14_chain(_z14_wm)        # add_z14.py: V.grad, g.grad from W.grad; W.grad -> None
            if _wpert:
                with torch.no_grad():
                    _whh.copy_(_raw_whh)                 # the optimiser updates the RAW weights
                    if _wmask is not None and _whh.grad is not None:
                        _whh.grad.mul_(_wmask)           # d/d(raw) = mask * d/d(dropped)
            if a.sam > 0:
                with torch.no_grad():
                    if _sam_eps is not None:
                        for q_, e_ in zip(_sam_params, _sam_eps): q_.sub_(e_)     # back to the unperturbed weights
                    _sam_g = [(q_.grad.detach().clone() if q_.grad is not None else torch.zeros_like(q_)) for q_ in _sam_params]
                    _sam_gn = torch.sqrt(sum((g * g).sum() for g in _sam_g)).clamp_min(1e-12)
            torch.nn.utils.clip_grad_norm_(core.parameters(), a.clip_norm)   # matches the main loop
            if core2 is not None:
                torch.nn.utils.clip_grad_norm_(core2.parameters(), a.clip_norm)
                opt2.step(); sched2.step()
            if a.rec_wd > 0:
                with torch.no_grad():
                    for _n, _q in core.named_parameters():
                        if "weight_hh" in _n: _q.mul_(1.0 - a.lr * a.rec_wd)
                    if core2 is not None:
                        for _n, _q in core2.named_parameters():
                            if "weight_hh" in _n: _q.mul_(1.0 - a.lr * a.rec_wd)
            opt.step(); sched.step(); n_updates += 1
            if _z48 is not None:                       # add_z48.py: the adversary's own update (its grads are not clipped)
                _z48["opt"].step()
            if _z18_cols_t is not None:                 # add_z18.py: dropped columns stay at exactly zero (over the gradient mask)
                with torch.no_grad(): core.in_proj[0].weight[:, _z18_cols_t] = 0.0
            if _la_slow is not None and n_updates % a.lookahead_k == 0:
                with torch.no_grad():                  # Lookahead sync (add_z9.py): phi += alpha (theta - phi); theta = phi
                    for _qs, _qf in zip(_la_slow, core.parameters()):
                        _qs.add_(_qf.detach() - _qs, alpha=a.lookahead_alpha); _qf.copy_(_qs)
            if _z14 is not None:                           # add_z14.py: the EMA copy and every checkpoint hold the plain weight
                _z14_materialise(core, _z14, grad=False)
                if step == 1 or step % 1000 == 0: print(_z14_diag(core, _z14), flush=True)
            if ema is not None:
                with torch.no_grad():
                    for qe, qc in zip(ema.parameters(), core.parameters()):
                        qe.mul_(a.ema).add_(qc.detach(), alpha=1.0 - a.ema)
            if _ems:                                   # add_ems.py: extra averages (written here, never read by training)
                with torch.no_grad():
                    for _ex in _ems:
                        if step < _ex["s0"]: continue
                        if step == _ex["s0"]:
                            for qe, qc in zip(_ex["m"].parameters(), core.parameters()): qe.copy_(qc.detach())
                            continue
                        _exd = _ems_decay(_ex, step, a.steps)
                        for qe, qc in zip(_ex["m"].parameters(), core.parameters()):
                            qe.mul_(_exd).add_(qc.detach(), alpha=1.0 - _exd)
            if a.global_pool and a.pool_decay < 1.0:
                rstat.decay(a.pool_decay)
            rstat.update(tt.clamp(-2.0, 2.0) if (a.corr_target == "t" and tt is not None) else y, pr.detach(), w)
            for _rk, _pk in zip(rstat_mh, _z59_pk):          # add_z59.py: the main rstat's decay + update rule, per head
                if a.global_pool and a.pool_decay < 1.0: _rk.decay(a.pool_decay)
                _rk.update(tt.clamp(-2.0, 2.0) if (a.corr_target == "t" and tt is not None) else y, _pk, w)
            if a.elr > 0 and step > int(a.elr_warm * a.steps):   # add_z12.py: fold this chunk's predictions into the store
                stream.elr_write(pr.detach().float().cpu().numpy(), a.elr_beta)
            if rstat_dm is not None and _ao is not None:
                rstat_dm.update(y, _ao.detach(), w)
            for _j, _k, _pa in _auxh_pairs:
                _ysrc = tt.clamp(-2.0, 2.0) if (a.aux_teacher and tt is not None) else y
                rstat_auxh[_j].update(_ysrc[:, _k:], _pa, w[:, _k:])
            if step % a.log_every == 0:
                el = time.time() - t0
                print(f"  step {step:5d}/{a.steps}  loss {loss.item():.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {step/el:.2f} steps/s  "
                      f"{el/60:.1f} min", flush=True)
            if step % 1000 == 0:
                sm = stream.summary()
                print(f"  drawn: {sm['drawn_primary']} primary + {sm['drawn_extra']} extra "
                      f"(unique {sm['unique_primary']} / {sm['unique_extra']}); selected rows "
                      f"{sm['rows_selected']:,} of {sm['rows_forward']:,} forward "
                      f"({sm['rows_selected']/max(sm['rows_forward'],1):.1%}), weight mass "
                      f"{sm['weight_mass']:.0f}", flush=True)
            if a.ckpt_every and step % a.ckpt_every == 0 and step != a.steps:
                _ckpt(core, outdir/f"ck{step}.pt")
            if stream.new_epoch:
                # the previous epoch's sequences all finished on this step (lanes are in
                # lock-step: every sequence is 20,000 rows)
                ep = stream.epoch - 1
                _eval_point(step, a.eval_n, label=f"epoch{ep}")
                _ckpt(core, outdir/f"ck_ep{ep}.pt", {"scale": hist[-1]["scale"]})
                if ema is not None:
                    _ckpt(ema, outdir/f"ema_ep{ep}.pt", {"scale": hist[-1].get("scale_ema")})
            elif step % a.eval_every == 0 and step != a.steps:
                _eval_point(step, a.eval_n)
        r = _eval_point(a.steps, a.final_eval_n, label="final")
        # The FIXED-ENDPOINT checkpoint is the primary artefact of a recipe test; best.pt
        # is selected on eval and therefore biased upward.
        _ckpt(core, outdir/"final.pt", {"scale": r["scale"], "wp": r["wp"]})
        if core2 is not None:
            # the second NCL member is an ordinary checkpoint in its own run dir, so export,
            # dumping and fusing all work on it unchanged
            out2dir = ROOT/"runs"/f"{a.tag}_m1"; out2dir.mkdir(parents=True, exist_ok=True)
            sc2 = calibrate_scale(core2, train, two_head=a.two_head, common_head=a.common_head,
                                  device=dev, keep_cols=keep_cols,
                                  xform=(_rg if rg is not None else None), famv=(fam if a.family_input else None), fnorm=fnorm, famv_norm=fam, lat=lat_tr)
            r2 = evaluate(core2, ev, n=a.final_eval_n, two_head=a.two_head,
                          common_head=a.common_head, scale=sc2, device=dev, keep_cols=keep_cols,
                          xform=(_rg if rg is not None else None), famv=(fam_ev if a.family_input else None), fnorm=fnorm, famv_norm=fam_ev, lat=lat_ev,
                          reset_every=a.state_reset)
            torch.save({"state": {k: v.cpu() for k, v in core2.state_dict().items()},
                        "cfg": vars(a), "scale": sc2, "wp": r2["wp"]}, out2dir/"final.pt")
            print(f"NCL member 2 -> {out2dir}/final.pt  scale {sc2:.4f}  WP {r2['wp']:+.4f}", flush=True)
        if ema is not None:
            _ckpt(ema, outdir/"ema_final.pt", {"scale": r["scale_ema"], "wp": r["wp_ema"]})
        _ems_res = {}
        for _ex in _ems:                                # add_ems.py: calibrated and evaluated exactly like the --ema copy
            _exm = _ex["m"]
            _exsc = calibrate_scale(_exm, train, two_head=a.two_head,
                                   common_head=a.common_head, device=dev, keep_cols=keep_cols,
                                   xform=(_rg if rg is not None else None), famv=(fam if a.family_input else None), fnorm=fnorm, famv_norm=fam, lat=lat_tr)
            _exr = evaluate(_exm, ev, n=a.final_eval_n, two_head=a.two_head,
                            common_head=a.common_head, scale=_exsc, device=dev, keep_cols=keep_cols,
                            xform=(_rg if rg is not None else None), famv=(fam_ev if a.family_input else None), fnorm=fnorm, famv_norm=fam_ev, lat=lat_ev,
                            reset_every=a.state_reset)
            _ckpt(_exm, outdir/f"ema_{_ex['name']}_final.pt", {"scale": _exsc, "wp": _exr["wp"]})
            _ems_res[_ex["name"]] = {"spec": _ex["spec"], "kind": _ex["kind"], "s0": _ex["s0"], "r0": _ex["r0"],
                                    "scale": _exsc, "wp": _exr["wp"], "wp_pooled": _exr.get("wp_pooled")}
            print(f"ema-extra {_ex['name']} ({_ex['spec']}): scale x{_exsc:.3f}  WP {_exr['wp']:+.4f}  "
                  f"POOLED {_exr.get('wp_pooled', float('nan')):+.4f}  -> ema_{_ex['name']}_final.pt", flush=True)
        if a.splice > 0: print(f"splice: {_sp_n} sequence starts carried a foreign state", flush=True)
        if a.train_reset_p > 0: print(f"train-reset: {_tr_n} resets over {_tr_lanes} within-sequence lane-chunks "
                                      f"({_tr_n / max(_tr_lanes, 1):.3f})", flush=True)
        if a.chunk_jitter: print(f"chunk-jitter: {stream.jit_n} sequence cycles jittered (last first-chunk length {stream.jit_last})", flush=True)
        if _b3_pc is not None:                     # add_b3.py
            print(f"pcgrad[{a.pcgrad}]: {_b3_pc['steps']} steps; mean pairwise cos {_b3_pc['cos_sum'] / max(1, _b3_pc['cos_n']):+.4f}; conflicting "
                  f"pairs {100.0 * _b3_pc['conflicts'] / max(1, _b3_pc['pairs']):.1f}%; mean |g_proj|/|g_plain| {_b3_pc['ratio'] / max(1, _b3_pc['steps']):.3f}", flush=True)
        if _b3_fl is not None:
            print(f"anchor-floor: {100.0 * _b3_fl['off_mass'] / max(1e-6, _b3_fl['mass']):.1f}% of anchor mass below the floor over training; "
                  f"mean anchor plain {_b3_fl['plain'] / max(1, _b3_fl['steps']):.4f} -> floored {_b3_fl['floored'] / max(1, _b3_fl['steps']):.4f}", flush=True)
        if _g1_cols is not None: print(f"vaug: {_g1_n['steps']} augmented steps; sequence starts augmented {_g1_n['on']}, left clean "
                                       f"{_g1_n['off']} (family filter)", flush=True)   # add_g1.py
        if _z18_cols_t is not None:                     # add_z18.py: assert the invariant on the shipped and the EMA weights
            _zmax = core.in_proj[0].weight[:, _z18_cols_t].abs().max().item()
            _zema = ema.in_proj[0].weight[:, _z18_cols_t].abs().max().item() if ema is not None else 0.0
            print(f"drop-cols: {len(_z18_cols)} columns zeroed; max |in_proj.0.weight[:, dropped]| final {_zmax:.1e}  ema {_zema:.1e}  "
                  f"({'OK' if _zmax == 0.0 and _zema == 0.0 else 'VIOLATED'})", flush=True)
        if a.pw_schedule != "const" or a.anchor_end > 0:
            print(f"pw-schedule {a.pw_schedule}: weights at the last step  pearson {_pw:g}  anchor {_aw:g}", flush=True)
        if _z20_on:                                      # add_z20.py
            print(f"z20-kd: KD weight at the last step {_dw:g}; look={a.distill_look} pow={a.distill_look_pow:g} dis={a.distill_dis:g} "
                  f"tau={a.distill_dis_tau:g} step={a.distill_step:g} w2={a.distill_w2:g} clip={a.distill_clip}", flush=True)
        if a.slow_units > 0:                             # add_z25.py: where did the slow lane end up?
            _g = core.gru; _H = _g.hidden_size; _sl = slice(2 * _H - a.slow_units, 2 * _H)
            _bz = (_g.bias_hh_l0[_sl] + _g.bias_ih_l0[_sl]).detach().float().cpu()
            _bf = (_g.bias_hh_l0[_H:2 * _H - a.slow_units] + _g.bias_ih_l0[_H:2 * _H - a.slow_units]).detach().float().cpu()
            _tau = 1.0 / (1.0 - torch.sigmoid(_bz)).clamp_min(1e-6); _tauf = 1.0 / (1.0 - torch.sigmoid(_bf)).clamp_min(1e-6)
            _b0 = torch.log(torch.as_tensor(np.geomspace(a.slow_tmin, a.slow_tmax, a.slow_units), dtype=torch.float32))
            print(f"z25-slow: trained zero-input time constants of the {a.slow_units} slow units: min {float(_tau.min()):.0f} "
                  f"median {float(_tau.median()):.0f} max {float(_tau.max()):.0f} rows (init {a.slow_tmin:g}..{a.slow_tmax:g}); "
                  f"b_z moved by mean |d| {float((_bz - _b0).abs().mean()):.3f} max |d| {float((_bz - _b0).abs().max()):.3f}; "
                  f"fast units' zero-input tau median {float(_tauf.median()):.1f} max {float(_tauf.max()):.1f}", flush=True)
        manifest = {**stream.summary(), "optimizer_updates": n_updates, "steps": a.steps,
                    "minutes": (time.time() - t0) / 60}
        json.dump({"cfg": vars(a), "history": hist, "best": best, "final_wp": r["wp"],
                   "final_wp_ema": r.get("wp_ema"), "params": core.n_params(),
                   "manifest": manifest, **({"ema_extra": _ems_res} if _ems_res else {})}, open(outdir/"history.json", "w"), indent=1)
        print(f"\nmanifest: {json.dumps(manifest)}")
        print(f"\nbest WP {best:+.4f}  |  final {r['wp']:+.4f}"
              + (f"  ema {r['wp_ema']:+.4f}" if ema is not None else "")
              + f"  |  baseline 0.5896  |  {(time.time()-t0)/60:.1f} min", flush=True)
        return

    for step in range(1, a.steps+1):
        if a.loss in ("seqwp", "anchored", "zmse"):
            xb, yb, sidx, mb, wstart = sample_grouped(train, rng, a.groups, a.per_seq, a.body, a.burn,
                                              row_weight=rw,
                                              seq_p=(seq_p_fam if seq_p_fam is not None else seq_p))
            if rw is not None and a.row_weight_power != 1.0:
                mb = mb ** a.row_weight_power
            # NB: weighted Pearson is invariant to a global weight scale, so only the
            # RELATIVE row weights matter -- no normalisation needed here.
        else:
            xb, yb, mb = train.batch(rng, a.batch, a.body, a.burn)
            sidx = None          # train.batch does not report which sequences it drew
        if keep_cols is not None: xb = xb[..., keep_cols]
        if sidx is None and (fnorm is not None or a.family_input):
            raise SystemExit(f"--loss {a.loss} uses SequenceCache.batch, which does not "
                             f"return sequence indices, so --fam-norm/--family-input "
                             f"cannot be applied. Use --loss seqwp.")
        if fnorm is not None: xb = famnorm.apply(xb, fam[sidx], *fnorm)
        if rg is not None: xb = _rg(xb)
        if sidx is not None: xb = _fam(xb, sidx)
        x = torch.from_numpy(xb).permute(1, 0, 2).to(dev, non_blocking=True)   # (T,B,F)
        y = torch.from_numpy(yb).to(dev, non_blocking=True)                    # (B,body,2)
        if a.prefix_state == "off" and not a.detach_burn:
            h0 = core.init_state(a.batch, dev)
            if a.state_pass > 0 and state_bank is not None:
                # a DIFFERENT window's final state, detached: explores attainable states
                # instead of only the ones reachable from zero
                perm = torch.randperm(state_bank.shape[1], device=dev)[:a.batch]
                h0 = state_bank[:, perm].clone()
                zero = torch.rand(a.batch, device=dev) < a.state_pass
                if zero.any(): h0[:, zero] = 0.0
            out, hT = core(x, h0)
            if a.state_pass > 0:
                state_bank = hT.detach()
            out = out[a.burn:].permute(1, 0, 2)                   # (B,body,n_out)
        else:
            # State exposure. Training builds every window's state from zero over `burn`
            # rows; inference carries state from row 0 across 20,000. These arms change
            # only how the state reaching the loss rows is built -- never the loss rows
            # themselves, and never the gradient horizon (the body is identical).
            h0 = core.init_state(a.batch, dev)
            with torch.no_grad():
                _, h_burn = core(x[:a.burn], h0)
                if a.prefix_state != "off":
                    want = (np.ones(len(sidx), bool) if a.prefix_state == "all"
                            else (fam[sidx] == 0))
                    if want.any():
                        h_pre = prefix_states(core, train, sidx[want],
                                              wstart[want] + a.burn, dev, xprep)
                        h_burn = h_burn.clone()
                        h_burn[:, torch.from_numpy(np.where(want)[0]).to(dev), :] = h_pre
            # Detached in BOTH arms: otherwise the candidate would confound history
            # length with a changed gradient path, since the stock recipe backpropagates
            # through its burn rows.
            out, _ = core(x[a.burn:], h_burn.detach())
            out = out.permute(1, 0, 2)
        # --row-weight implies using the weights: otherwise they are silently
        # discarded here and the run is an unmarked duplicate of the control.
        m = torch.from_numpy(mb).to(dev) if (a.mask_weight or rw is not None) else None
        if a.tar > 0:
            # `out` is (B, body, n_out) -- batch first and projected. TAR wants the
            # HIDDEN trajectory differenced along TIME, so pass the pre-head sequence
            # with its time axis named explicitly.
            import gateinit as _gi2
            loss_tar = a.tar * _gi2.tar_penalty(out, time_dim=1)
        if a.lattice_head:
            # Cross-entropy supervises the CLASS; the metric cares about the w-tilted
            # readout. The readout is differentiable (softmax then a weighted mean), so
            # with --lattice-wp both can be trained at once: the metric decides the
            # ranking, the classes shape the distribution it is read out of. Pure CE
            # never sees the metric at all, which is why it starts slow.
            K = a.lattice_head
            qs = np.asarray([BASES[fam[i]] for i in sidx], np.float64)
            lab = np.stack([_lat.labels(yb[k], qs[k], K) for k in range(len(sidx))])
            tgt = torch.from_numpy(lab).to(dev)
            ce = torch.nn.functional.cross_entropy(out.reshape(-1, 2*K+1), tgt.reshape(-1))
            if a.lattice_wp > 0:
                sig = _lat.readout(out, torch.from_numpy(qs).float().to(dev), K)
                wp = seq_pooled_wp_loss(sig, y, a.groups, m)
                loss = wp + a.lattice_wp * ce
                if step % 1000 == 0:
                    print(f"        lattice: wp {wp.item():.4f}  ce {ce.item():.4f}", flush=True)
            else:
                loss = ce
        elif a.loss == "mse":
            tgt = y if (a.two_head or a.common_head) else ((y[..., 0]-y[..., 1])/2).unsqueeze(-1)
            loss = torch.nn.functional.mse_loss(out, tgt)
        elif a.loss == "seqwp":
            # --common-head MUST be tested before the generic head-0 path. Ordered the
            # other way, seq_pooled_wp_loss_common was dead code: head 1 (the shared
            # component c) received no gradient and, being zero-initialised, stayed
            # exactly zero -- so the run emitted (s+0, -s+0) and silently tested nothing.
            loss = (seq_pooled_wp_loss_horizon(out, y, a.groups, aux_off, a.aux_weight, m)
                    if aux_off else
                    seq_pooled_wp_loss_common(out, y, a.groups, m) if a.common_head else
                    seq_pooled_wp_loss(out[..., 0], y, a.groups, m))
            if ssl_w is not None:
                # targets are computed on the FULL window (burn+body) so the burn-in rows
                # can see their own future events, then sliced to the loss region
                at = np.stack([_auxt(xb[k].astype(np.float64), ssl_w)[a.burn:]
                               for k in range(xb.shape[0])])
                att = torch.from_numpy(at).float().to(dev)          # (B,body,6)
                tg, vm = att[..., :3], att[..., 3:]
                # A sampled window is a truncated view: where the next event lies outside
                # it, direction and wait are UNKNOWN, and the last row does not know
                # whether an event follows. Masked out rather than given a default label.
                def _masked(lossmap, mask):
                    den = mask.sum().clamp_min(1.0)
                    return (lossmap * mask).sum() / den
                ev_l = _masked(torch.nn.functional.binary_cross_entropy_with_logits(
                    out[..., 1], tg[..., 0], reduction="none"), vm[..., 0])
                sg_l = _masked(torch.nn.functional.binary_cross_entropy_with_logits(
                    out[..., 2], (tg[..., 1] > 0).float(), reduction="none"), vm[..., 1])
                wt_l = _masked((out[..., 3] - tg[..., 2])**2, vm[..., 2])
                loss = loss + a.ssl_aux * (ev_l + sg_l + 0.1*wt_l)
        elif a.loss in ("anchored", "zmse"):
            smu = torch.from_numpy(seq_mu[sidx]).to(dev)
            ssd = torch.from_numpy(seq_sd[sidx]).to(dev)
            loss = (anchored_loss(out[..., 0], y, a.groups, smu, ssd, a.lam, m)
                    if a.loss == "anchored" else zmse_loss(out[..., 0], y, smu, ssd, m))
        elif a.common_head:
            loss = (seq_pooled_wp_loss_common(out, y, a.groups, m)
                    if a.loss == "seqwp" else wp_loss_common(out, y, m))
        elif a.two_head:
            loss = wp_loss_two_head(out, y, m)
        else:
            loss = wp_loss(out[..., 0], y, m)
        if a.ckpt_every and step % a.ckpt_every == 0:
            torch.save({"state": core.state_dict(), "cfg": vars(a), "step": step},
                       outdir/f"ck{step}.pt")
        if a.tar > 0: loss = loss + loss_tar
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), a.clip_norm)
        opt.step(); sched.step()

        if step % 100 == 0:
            el = time.perf_counter()-t0
            print(f"  step {step:5d}/{a.steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{step/el:.2f} steps/s  {el/60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            n = a.final_eval_n if step == a.steps else a.eval_n
            sc = calibrate_scale(core, train, two_head=a.two_head, common_head=a.common_head,
                                 device=dev, keep_cols=keep_cols,
                                 xform=(_rg if rg is not None else None),
                                 famv=(fam if a.family_input else None),
                                 fnorm=fnorm, famv_norm=fam, lat=lat_tr)
            r = evaluate(core, ev, n=n, two_head=a.two_head, common_head=a.common_head, scale=sc, device=dev, keep_cols=keep_cols, xform=(_rg if rg is not None else None),
                             famv=(fam_ev if a.family_input else None),
                             fnorm=fnorm, famv_norm=fam_ev, lat=lat_ev,
                             reset_every=a.state_reset)
            r["scale"] = sc
            hist.append({"step": step, **r})
            flag = ""
            if r["wp"] > best:
                best = r["wp"]; flag = "  <-- best"
                torch.save({"state": {k: v.cpu() for k, v in core.state_dict().items()}, "cfg": vars(a), "scale": sc,
                            "params": core.n_params(), "wp": r["wp"]}, outdir/"best.pt")
            print(f"  EVAL step {step}: WP {r['wp']:+.4f} +/- {r['se']:.4f}  "
                  f"(n={r['n']}, scale x{r['scale']:.3f} -> pred_std {r['pred_std']:.3f}){flag}", flush=True)
    sc_final = calibrate_scale(core, train, two_head=a.two_head, common_head=a.common_head,
                               device=dev, keep_cols=keep_cols,
                               xform=(_rg if rg is not None else None),
                               famv=(fam if a.family_input else None),
                               fnorm=fnorm, famv_norm=fam, lat=lat_tr)
    torch.save({"state": core.state_dict(), "cfg": vars(a), "scale": sc_final,
                "params": core.n_params(), "wp": hist[-1]["wp"] if hist else float("nan")},
               outdir/"final.pt")
    json.dump({"cfg": vars(a), "history": hist, "best": best, "params": core.n_params()},
              open(outdir/"history.json", "w"), indent=1)
    print(f"\nbest WP {best:+.4f}  |  baseline 0.5896  |  {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
