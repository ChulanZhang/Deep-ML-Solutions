def video_chunk_fps(num_frames: int, chunk_size: int, time_per_chunk_s: float) -> float:
    import math
    total_steps = math.ceil(num_frames / chunk_size)
    total_time = total_steps * time_per_chunk_s
    fps = num_frames / total_time
    return round(fps, 4)
