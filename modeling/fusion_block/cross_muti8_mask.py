#Other code will be uploaded after the paper is officially accepted and published.

def AFF(x,y,keep_rate,enhance_weights,csnum_heads):
    attn = EVAD_fusion(dim=1024, num_heads=8, csnum_heads=csnum_heads,qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0.,
                     norm_layer=nn.LayerNorm, mlp_ratio=4., act_layer=nn.GELU, drop=0.,drop_path=0.,
                     init_values=0, enhance_weights=enhance_weights, keep_rate=keep_rate)
    if x.device != 'cuda':
        x = x.to('cuda')
    if y.device != 'cuda':
        y = y.to('cuda')
    output = attn(x, y)
    return output
