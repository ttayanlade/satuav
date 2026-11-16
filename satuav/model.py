from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.distributions.dirichlet import Dirichlet

from . import paths  # noqa: F401  (puts MultiMAE on sys.path)
from multimae.input_adapters import PatchedInputAdapter
from multimae.multimae import MultiMAE
from multimae.output_adapters import SpatialOutputAdapter

CHANNELS = {'sat_rgb': 3, 'sat_all': 6, 'uav_rgb': 3}
N_TOK = 196


class SatUAVMAE(MultiMAE):
    def __init__(self, in_domains, out_domains, dec_depth=2, dec_dim=256, **kw):
        input_adapters = {d: PatchedInputAdapter(num_channels=CHANNELS[d], stride_level=1, patch_size_full=16)
                          for d in in_domains}
        output_adapters = {d: SpatialOutputAdapter(num_channels=CHANNELS[d], stride_level=1, patch_size_full=16,
                                                   dim_tokens=dec_dim, depth=dec_depth, num_heads=8,
                                                   use_task_queries=True, task=d, context_tasks=list(in_domains))
                           for d in out_domains}
        super().__init__(input_adapters=input_adapters, output_adapters=output_adapters, num_global_tokens=1,
                         dim_tokens=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kw)
        self.mod_emb = nn.ParameterDict({d: nn.Parameter(torch.zeros(1, 1, 768)) for d in in_domains})
        for p in self.mod_emb.values():
            nn.init.trunc_normal_(p, std=0.02)
        self.in_domains = list(in_domains)

    def no_weight_decay(self):
        return super().no_weight_decay() | {f'mod_emb.{d}' for d in self.in_domains}

    # --------------------------------------------------------------------------------- masking
    def counts(self, B, K, alphas, mode, present, device):
        """Visible-token count per (sample, domain) summing exactly to K per sample."""
        D = len(self.in_domains)
        if mode == 'full':  # all tokens of every present domain visible
            return torch.full((B, D), N_TOK, device=device, dtype=torch.long) * present.long()
        if mode == 'sat_full':  # all sat tokens visible, uav fully masked
            c = torch.zeros(B, D, dtype=torch.long, device=device)
            for i, d in enumerate(self.in_domains):
                if d.startswith('sat'):
                    c[:, i] = N_TOK
            return c
        lam = Dirichlet(torch.tensor(alphas, dtype=torch.float)).sample((B,)).to(device) if D > 1 else \
            torch.ones(B, 1, device=device)
        lam = lam * present.float()
        lam = lam / lam.sum(1, keepdim=True).clamp_min(1e-8)
        assert K <= N_TOK
        c = (lam * K).round().long()
        c[:, -1] = K - c[:, :-1].sum(1)  # exact total; absent last domain gets lam=0 so this is 0
        return c

    def make_masks(self, counts):
        """counts: (B, D) visible tokens per domain -> per-domain masks (1 = masked), ids_keep, ids_restore."""
        B, D = counts.shape
        dev = counts.device
        masks = []
        for i in range(D):
            noise = torch.rand(B, N_TOK, device=dev)
            rank = noise.argsort(1).argsort(1)
            masks.append((rank >= counts[:, i:i + 1]).long())
        mask_all = torch.cat(masks, 1)
        K = int(counts.sum(1)[0])
        assert bool((counts.sum(1) == K).all()), 'all samples in a batch must have the same visible count'
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :K]
        return {d: m for d, m in zip(self.in_domains, masks)}, ids_keep, ids_restore

    # --------------------------------------------------------------------------------- forward
    def encode(self, x, counts):
        B = list(x.values())[0].shape[0]
        H, W = list(x.values())[0].shape[-2:]
        tok = {d: self.input_adapters[d](x[d]) + self.mod_emb[d] for d in self.in_domains}
        input_info = self.generate_input_info(input_task_tokens=tok, image_size=(H, W))
        task_masks, ids_keep, ids_restore = self.make_masks(counts)
        t = torch.cat([tok[d] for d in self.in_domains], 1)
        t = torch.gather(t, 1, ids_keep.unsqueeze(-1).expand(-1, -1, t.shape[2]))
        t = torch.cat([t, repeat(self.global_tokens, '() n d -> b n d', b=B)], 1)
        enc = self.encoder(t)
        return enc, task_masks, input_info, ids_keep, ids_restore

    def forward(self, x, K=66, alphas=(0.9, 0.1), mode='random', present=None, out_domains=None):
        B = list(x.values())[0].shape[0]
        dev = list(x.values())[0].device
        if present is None:
            present = torch.ones(B, len(self.in_domains), dtype=torch.bool, device=dev)
        counts = self.counts(B, K, list(alphas), mode, present, dev)
        enc, task_masks, input_info, ids_keep, ids_restore = self.encode(x, counts)
        out_domains = out_domains or list(self.output_adapters.keys())
        preds = {d: self.output_adapters[d](encoder_tokens=enc, input_info=input_info, ids_keep=ids_keep,
                                            ids_restore=ids_restore) for d in out_domains}
        return preds, task_masks, enc, ids_keep, ids_restore, input_info

    def decoder_features(self, d, enc, input_info, ids_keep, ids_restore):
        """SpatialOutputAdapter.forward up to (excluding) out_proj -> (B, 196, dec_dim)."""
        oa = self.output_adapters[d]
        ctx = oa.proj_context(enc)
        q, ctx = oa.get_queries_and_context(ctx, input_info, ids_keep, ids_restore)
        y = oa.decoder(oa.query_norm(q), oa.context_norm(ctx))
        y = y + oa.mlp(oa.out_norm(y))
        return oa.decoder_transformer(y)


def load_pretrained(model, path, rgb_domains=('sat_rgb', 'uav_rgb', 'sat_all')):
    """Encoder + global token from the MultiMAE checkpoint; every RGB-like input adapter from the RGB patch
    embedding. sat_all = [R,G,B, extra...]: RGB weights copied, extra bands = mean of RGB channel weights, all
    rescaled by 3/C so the summed activation magnitude matches the 3-channel embedding. Decoders: random init."""
    sd = torch.load(path, map_location='cpu')['model']
    own = model.state_dict()
    new = {}
    for k, v in sd.items():
        if k.startswith('encoder.') or k == 'global_tokens':
            new[k] = v
    w, b = sd['input_adapters.rgb.proj.weight'], sd['input_adapters.rgb.proj.bias']
    for d in model.in_domains:
        if d not in rgb_domains:
            continue
        C = CHANNELS[d]
        if C == 3:
            wd = w.clone()
        else:
            extra = w.mean(1, keepdim=True).expand(-1, C - 3, -1, -1)
            wd = torch.cat([w, extra], 1) * (3.0 / C)
        new[f'input_adapters.{d}.proj.weight'] = wd
        new[f'input_adapters.{d}.proj.bias'] = b.clone()
    missing = [k for k in new if k not in own]
    assert not missing, missing
    msg = model.load_state_dict(new, strict=False)
    return msg


def build(cfg, init=True):
    m = SatUAVMAE(cfg['in_domains'], cfg['out_domains'])
    if init and cfg.get('init', 'multimae') == 'multimae':
        load_pretrained(m, paths.WEIGHTS)
    return m


def build_imnet(sat_domain='sat_rgb'):
    """E-imnet: original MultiMAE encoder + RGB patch embedding, no domain pretraining."""
    m = SatUAVMAE([sat_domain], [])
    load_pretrained(m, paths.WEIGHTS)
    with torch.no_grad():
        m.mod_emb[sat_domain].zero_()
    return m
