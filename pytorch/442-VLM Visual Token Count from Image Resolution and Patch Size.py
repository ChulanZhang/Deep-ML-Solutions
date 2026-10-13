def vlm_visual_tokens(img_h: int, img_w: int, patch_size: int) -> int:
    h_tokens = img_h // patch_size
    w_tokens = img_w // patch_size
    return int(h_tokens * w_tokens)
