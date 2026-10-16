def video_latent_memory(frames: int, channels: int, height: int, width: int, precision_bytes: int = 2) -> float:
    # Memory for uncompressed latents
    # shape = (frames, channels, height, width)
    
    total_elements = frames * channels * height * width
    total_bytes = total_elements * precision_bytes
    
    return float(round(total_bytes / (1024 ** 2), 4))
