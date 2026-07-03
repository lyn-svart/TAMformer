"""Compute TAMformer parameter count and FLOPs (analytical, no TensorFlow required)."""

import yaml
from pathlib import Path


def mha_params(embed_dim, num_heads=6):
    """Keras MultiHeadAttention(key_dim=embed_dim) weight count."""
    key_dim = embed_dim
    value_dim = key_dim
    # Q, K, V einsum: embed_dim * num_heads * key_dim each
    qkv = 3 * embed_dim * num_heads * key_dim
    # output projection: (num_heads * value_dim) * embed_dim
    out = num_heads * value_dim * embed_dim
    return qkv + out


def ffn_params(embed_dim, ff_dim=1024):
    return embed_dim * ff_dim + ff_dim + ff_dim * embed_dim + embed_dim


def layernorm_params(embed_dim):
    return 2 * embed_dim  # gamma + beta


def transformer_block_params(embed_dim, num_heads=6, ff_dim=1024, normalization=True):
    p = mha_params(embed_dim, num_heads) + ffn_params(embed_dim, ff_dim)
    if normalization:
        p += 2 * layernorm_params(embed_dim)
    return p


def learned_causal_mask_params(input_dim, key_len, hidden=(128, 64, 32)):
    dims = [input_dim] + list(hidden) + [key_len]
    p = 0
    for i in range(len(dims) - 1):
        p += dims[i] * dims[i + 1] + dims[i + 1]
    return p


def position_embedding_params(seq_len, embed_dim):
    return seq_len * embed_dim


def classifier_head_params(embed_dim, num_classes=1, hidden=(64, 32)):
    dims = [embed_dim] + list(hidden) + [num_classes]
    p = 0
    for i in range(len(dims) - 1):
        p += dims[i] * dims[i + 1] + dims[i + 1]
    return p


def self_attention_flops(seq_len, embed_dim, num_heads=6):
    """MACs for one self-attention block (QKV proj + attention + output proj)."""
    key_dim = embed_dim
    inner = num_heads * key_dim
    qkv = 3 * 2 * seq_len * embed_dim * inner
    attn = 2 * 2 * seq_len * seq_len * inner  # QK^T and attn@V
    out_proj = 2 * seq_len * inner * embed_dim
    return qkv + attn + out_proj


def cross_attention_flops(query_len, key_len, embed_dim, num_heads=6):
    key_dim = embed_dim
    inner = num_heads * key_dim
    q = 2 * query_len * embed_dim * inner
    k = 2 * key_len * embed_dim * inner
    v = 2 * key_len * embed_dim * inner
    attn = 2 * 2 * query_len * key_len * inner
    out_proj = 2 * query_len * inner * embed_dim
    return q + k + v + attn + out_proj


def ffn_flops(seq_len, embed_dim, ff_dim=1024):
    return 2 * 2 * seq_len * embed_dim * ff_dim


def transformer_block_flops_self(seq_len, embed_dim, ff_dim=1024, num_heads=6):
    return self_attention_flops(seq_len, embed_dim, num_heads) + ffn_flops(seq_len, embed_dim, ff_dim)


def transformer_block_flops_cross(query_len, key_len, embed_dim, ff_dim=1024, num_heads=6):
    return cross_attention_flops(query_len, key_len, embed_dim, num_heads) + ffn_flops(query_len, embed_dim, ff_dim)


def mask_mlp_flops(seq_len, input_dim, key_len, hidden=(128, 64, 32)):
    dims = [input_dim] + list(hidden) + [key_len]
    flops = 0
    for i in range(len(dims) - 1):
        flops += 2 * seq_len * dims[i] * dims[i + 1]
    return flops


def vgg16_backbone_params():
  # VGG16 without top, ImageNet weights - standard count
  return 14_714_688


def vgg16_backbone_flops_per_frame(h=224, w=224):
    """Approximate FLOPs for VGG16 feature extractor (include_top=False, pooling=max)."""
    # Common estimate: ~15.5 GFLOPs for 224x224 forward pass through full VGG16 conv stack
    return 15.47e9


def analyze_config(name, model_opts):
    input_types = model_opts['obs_input_type']
    feat_sizes = model_opts['feat_size']
    pose_enabled = model_opts.get('pose_enabled', True)
    active_types, active_sizes = [], []
    for t, s in zip(input_types, feat_sizes):
        if t == 'pose' and not pose_enabled:
            continue
        active_types.append(t)
        active_sizes.append(s)

    seq_len = model_opts.get('seq_len', model_opts.get('sequence_length', 136))
    obs_length = model_opts.get('obs_length', seq_len)
    step = model_opts.get('step', 1)
    num_classes = model_opts.get('num_classes', 1)
    prediction_mode = model_opts.get('prediction_mode', 'temporal')
    trainable_backbone = model_opts.get('trainable_backbone', False)
    total_embed = sum(active_sizes)
    num_modalities = len(active_types)

    if num_classes > 1 or prediction_mode == 'final':
        query_len = seq_len
    else:
        query_len = int((seq_len - obs_length) / step)

    params = {}
    flops = {}

    # Position embeddings
    pe_params = sum(position_embedding_params(seq_len, d) for d in active_sizes)
    params['position_embeddings'] = pe_params

    # Learned masks
    concat_dim = total_embed
    mask_enc_p = learned_causal_mask_params(concat_dim, seq_len)
    mask_dec_p = learned_causal_mask_params(concat_dim, seq_len)
    params['learned_causal_masks'] = mask_enc_p + mask_dec_p

    # Modality self-attention encoders
    enc_p = sum(transformer_block_params(d) for d in active_sizes)
    params['modality_encoders'] = enc_p

    # Cross-attention decoder blocks (2 blocks)
    dec_p = 2 * transformer_block_params(total_embed)
    params['decoder_transformers'] = dec_p

    # Classifier
    if num_classes > 1:
        if prediction_mode == 'sequence':
            cls_p = query_len * classifier_head_params(total_embed, num_classes)
        else:
            cls_p = classifier_head_params(total_embed, num_classes)
    else:
        cls_p = query_len * classifier_head_params(total_embed, 1)
    params['classifier_heads'] = cls_p

    # VGG backbone (optional)
    backbone_p = 0
    backbone_flops = 0
    if trainable_backbone and 'local_context' in active_types:
        backbone_p = vgg16_backbone_params()
        target_dim = tuple(model_opts.get('target_dim', (224, 224)))
        backbone_flops = seq_len * vgg16_backbone_flops_per_frame(*target_dim)

    params['vgg16_backbone'] = backbone_p
    total_params = sum(params.values())

    # FLOPs (MACs * 2 convention -> we already use 2* for matmul)
    flops['position_embeddings'] = 0  # lookup only
    flops['mask_encoder'] = mask_mlp_flops(seq_len, concat_dim, seq_len)
    flops['mask_decoder'] = mask_mlp_flops(query_len, concat_dim, seq_len)
    flops['modality_encoders'] = sum(
        transformer_block_flops_self(seq_len, d) for d in active_sizes
    )
    flops['decoder_transformers'] = (
        transformer_block_flops_cross(query_len, seq_len, total_embed)
        + transformer_block_flops_cross(query_len, seq_len, total_embed)
    )
    if num_classes > 1 and prediction_mode == 'sequence':
        cls_flops = query_len * (2 * total_embed * 64 + 2 * 64 * 32 + 2 * 32 * num_classes)
    elif num_classes > 1:
        cls_flops = 2 * total_embed * 64 + 2 * 64 * 32 + 2 * 32 * num_classes
    else:
        cls_flops = query_len * (2 * total_embed * 64 + 2 * 64 * 32 + 2 * 32 * 1)
    flops['classifier_heads'] = cls_flops
    flops['vgg16_backbone'] = backbone_flops

    total_flops = sum(flops.values())
    gflops = total_flops / 1e9

    return {
        'name': name,
        'modalities': list(zip(active_types, active_sizes)),
        'seq_len': seq_len,
        'obs_length': obs_length,
        'step': step,
        'query_len': query_len,
        'embed_dim': total_embed,
        'num_heads': 6,
        'ff_dim': 1024,
        'num_classes': num_classes,
        'prediction_mode': prediction_mode,
        'trainable_backbone': trainable_backbone,
        'params_breakdown': params,
        'total_params': total_params,
        'flops_breakdown': flops,
        'total_gflops': gflops,
    }


def print_report(r):
    print('=' * 72)
    print(r['name'])
    print('=' * 72)
    print('Inputs:')
    for m, d in r['modalities']:
        if m == 'local_context' and r['trainable_backbone']:
            td = '(224, 224, 3) per frame via VGG16'
            print(f"  - {m}: ({r['seq_len']}, 224, 224, 3) raw crops -> {d}-D features")
        else:
            print(f"  - {m}: ({r['seq_len']}, {d}) pre-extracted features")
    print(f"Sequence length T = {r['seq_len']}, observation length = {r['obs_length']}, step = {r['step']}")
    print(f"Prediction queries T_q = {r['query_len']}")
    print(f"Fused embedding dim = {r['embed_dim']}")
    print(f"Transformer: {len(r['modalities'])} modality encoder(s) + 2 cross-attention decoder block(s)")
    print(f"  heads = {r['num_heads']}, FFN dim = {r['ff_dim']}, LayerNorm = True")
    print()
    print('Parameters:')
    for k, v in r['params_breakdown'].items():
        if v:
            print(f"  {k:28s} {v:>12,} ({v/1e6:.3f} M)")
    print(f"  {'TOTAL':28s} {r['total_params']:>12,} ({r['total_params']/1e6:.3f} M)")
    print()
    print('FLOPs (1 forward pass, batch=1):')
    for k, v in r['flops_breakdown'].items():
        if v:
            print(f"  {k:28s} {v/1e9:>10.3f} GFLOPs")
    print(f"  {'TOTAL':28s} {r['total_gflops']:>10.3f} GFLOPs")
    print()


def main():
    configs = [
        ('JAAD (configs_all.yaml)', Path('configs/configs_all.yaml')),
        ('JAAD beh (configs_beh.yaml)', Path('configs/configs_beh.yaml')),
        ('PIE (configs_pie.yaml)', Path('configs/configs_pie.yaml')),
        ('Car motion (configs_car_motion.yaml)', Path('configs/configs_car_motion.yaml')),
    ]
    for name, path in configs:
        with open(path) as f:
            opts = yaml.safe_load(f)['model_opts']
        print_report(analyze_config(name, opts))


if __name__ == '__main__':
    main()
