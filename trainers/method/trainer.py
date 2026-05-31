import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from .prompt_templates import PROMPT_TEMPLATES
from .losses import loss_gad, loss_lgd
from .geometry import cosine_kernel_from_protos, effective_base_kernel_from_all

_BACKBONE = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


@torch.no_grad()
def build_text_prototypes_for_classnames(classnames_all, n_prompts: int = 50, device="cuda"):
    """Average frozen zero-shot text features over ``n_prompts`` templates per class.

    Returns ``[C, D]`` L2-normalized class prototypes.
    """
    text_model, _ = create_model_from_pretrained(_BACKBONE)
    text_model = text_model.float().eval().to(device)
    tok = get_tokenizer(_BACKBONE)

    feats = []
    for cname in classnames_all:
        prompts = [PROMPT_TEMPLATES[cname][i] for i in range(n_prompts)]
        acc = []
        for p in prompts:
            tokens = tok(p).to(device)
            f = text_model.encode_text(tokens, normalize=True)
            acc.append(f)
        f_cls = torch.cat(acc, dim=0).mean(dim=0, keepdim=False)
        f_cls = F.normalize(f_cls, dim=-1)
        feats.append(f_cls)
    return torch.stack(feats, dim=0)


class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype
        self.text_transformer = biomedclip_model.text.transformer
        self.text_projection = biomedclip_model.text.text_projection if hasattr(biomedclip_model.text, 'text_projection') else None

        transformer_name = self.text_transformer.__class__.__name__
        hidden_size = getattr(getattr(self.text_transformer, 'config', None), 'hidden_size', 'unknown')
        proj_info = 'available' if self.text_projection is not None else 'absent'
        print(f"[TextEncoder] transformer={transformer_name}, hidden_size={hidden_size}, text_projection={proj_info}")

    def forward(self, inputs, tokenized_prompts=None, inputs_are_tokens=False):
        if inputs_are_tokens:
            prompts = self.model.text.transformer.embeddings.word_embeddings(inputs).type(self.dtype)
            if tokenized_prompts is None:
                tokenized_prompts = inputs
        else:
            prompts = inputs  # already token embeddings
        return self.model.encode_text(prompts, True, tokenized_prompts)


class TimmVisionWrapper(nn.Module):
    """Vision wrapper exposing the projected global token and per-patch tokens.

    Adds no learnable parameters; reuses the frozen trunk and head.
    """
    def __init__(self, original_timm_model):
        super().__init__()
        self.trunk = original_timm_model.trunk
        self.head = original_timm_model.head

    def forward(self, x):
        x = self.trunk.forward_features(x)  # [B, P+1, 768]
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        return {
            'global_features': self.head.proj(cls_token),       # [B, D]
            'patch_tokens_proj': self.head.proj(patch_tokens),  # [B, P, D]
        }


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.OGKD.N_CTX
        ctx_init = cfg.TRAINER.OGKD.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer(_BACKBONE)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx == 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if cfg.TRAINER.OGKD.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP to precompute the teacher text prototypes
        biomedclip_model_temp, _ = create_model_from_pretrained(_BACKBONE)
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        TextEncoder_temp = TextEncoder(biomedclip_model_temp)

        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            all_512_teacher_features = []
            for i in range(cfg.TRAINER.OGKD.N_PROMPTS):
                x_tokenized = torch.cat([self.tokenizer(PROMPT_TEMPLATES[classname][i]) for classname in classnames])
                text_features_512 = TextEncoder_temp(x_tokenized.cuda(), inputs_are_tokens=True)
                all_512_teacher_features.append(text_features_512.unsqueeze(1))

        self.fixed_embeddings_512 = torch.cat(all_512_teacher_features, dim=1)
        # These token vectors are saved by save_model() but ignored on load_model()
        # (we recompute them from the current class names).
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.OGKD.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        return self.construct_prompts(ctx, prefix, suffix)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = TimmVisionWrapper(biomedclip_model.visual)
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # Student text features from the learnable prompts
        text_features = self.text_encoder(self.prompt_learner(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_all = self.image_encoder(image.type(self.dtype))
        image_features = image_features_all['global_features']
        patch_features = image_features_all['patch_tokens_proj']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        # Prompted logits (global + per-patch)
        logits = logit_scale * image_features @ text_features.t()
        patch_logits_student = logit_scale * patch_features @ text_features.t()

        if not self.prompt_learner.training:
            return logits

        cfg = self.cfg.TRAINER.OGKD

        # Frozen zero-shot teacher prototypes
        fixed_embeddings = self.prompt_learner.fixed_embeddings_512
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

        # Statistics-based teacher prompt selection (from BiomedCoOp)
        with torch.no_grad():
            scores = []
            for i in range(fixed_embeddings.shape[1]):
                temp_logits = logit_scale * image_features @ fixed_embeddings[:, i, :].cuda().t()
                max_logits = torch.max(temp_logits, dim=1).values
                scores.append(torch.mean(max_logits).item())
            s_bar = torch.median(torch.tensor(scores))
            d_bar = torch.median(torch.abs(torch.tensor(scores) - s_bar))
            z = (torch.tensor(scores) - s_bar) / d_bar
            mask = torch.abs((z - torch.mean(z)) / torch.std(z)) <= cfg.TAU
            selected_embeddings = fixed_embeddings[:, mask].mean(dim=1)
            selected_embeddings = selected_embeddings / selected_embeddings.norm(dim=-1, keepdim=True)

        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

        # Teacher logits
        teacher_logits = logit_scale * image_features.cuda() @ selected_embeddings.cuda().t()
        patch_logits_teacher = logit_scale * patch_features.cuda() @ selected_embeddings.cuda().t()

        # Cross-entropy + SCCM (frozen-prototype anchoring)
        loss_ce = F.cross_entropy(logits, label)
        loss_sccm = nn.MSELoss()(text_features, fixed_embeddings.cuda()) * cfg.LAMBDA_SCCM

        # Global Geometry-Aware Distillation (GAD)
        loss_gad_term = loss_gad(
            student_logits=logits,
            teacher_logits=teacher_logits,
            W=self.W_eff_base,
            gamma=cfg.GAMMA,
        ) * cfg.LAMBDA_GAD

        # Label-Guided Geometry Distillation (LGD): keep the top-K attentive patches
        # (highest |teacher score| on the ground-truth class), then distill on them.
        B, N, C = patch_logits_student.shape
        k = max(1, int(round(N * cfg.TOP_K_RATIO)))
        with torch.no_grad():
            label_col = label.view(-1, 1, 1).expand(B, N, 1)
            gt_scores = patch_logits_teacher.gather(-1, label_col).squeeze(-1)  # [B, N]
            top_idx = torch.topk(gt_scores.abs(), k=k, dim=1).indices            # [B, K]
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, C)                        # [B, K, C]
        stu_patches = torch.gather(patch_logits_student, 1, idx_exp)
        tea_patches = torch.gather(patch_logits_teacher, 1, idx_exp)

        loss_lgd_term = loss_lgd(
            stu_patches,
            tea_patches,
            self.W_eff_base,
            label,
            gamma=cfg.GAMMA,
        ) * cfg.LAMBDA_LGD

        return logits, loss_ce, loss_sccm, loss_gad_term, loss_lgd_term


@TRAINER_REGISTRY.register()
class OGKD(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, _ = create_model_from_pretrained(_BACKBONE)
        biomedclip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]
        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)
        enabled = {name for name, param in self.model.named_parameters() if param.requires_grad}
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # only the prompt learner is optimized
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # Build the frozen class-similarity graph W and fold novel structure into
        # the base block (W_eff_base), used by both GAD and LGD.
        device = self.device
        ds = self.dm.dataset
        classnames_all = getattr(ds, "classnames_all", None) or ds.classnames
        classnames_base = getattr(ds, "classnames_base", None) or ds.classnames

        name_to_pos = {name: i for i, name in enumerate(classnames_all)}
        base_positions = [name_to_pos[c] for c in classnames_base if c in name_to_pos]
        base_idx = torch.tensor(base_positions, dtype=torch.long, device=device)
        if base_idx.numel() != len(classnames_base):
            base_idx = torch.arange(len(classnames_base), device=device)

        with torch.no_grad():
            protos_all = build_text_prototypes_for_classnames(
                classnames_all, n_prompts=cfg.TRAINER.OGKD.N_PROMPTS, device=device,
            )
        W_all = cosine_kernel_from_protos(protos_all, alpha=cfg.TRAINER.OGKD.ALPHA)
        W_eff_base = effective_base_kernel_from_all(W_all, base_idx=base_idx, lambda_novel=0.1)

        self.W_eff_base = W_eff_base
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        model_ref.W_eff_base = W_eff_base

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        logits, loss_ce, loss_sccm, loss_gad_term, loss_lgd_term = self.model(image, label)
        loss = loss_ce + loss_sccm + loss_gad_term + loss_lgd_term
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # the token buffers are recomputed from current class names
            state_dict.pop("prompt_learner.token_prefix", None)
            state_dict.pop("prompt_learner.token_suffix", None)

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
