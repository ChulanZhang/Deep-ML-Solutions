def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    """
    from collections import deque
    
    queue = deque()
    # Deep copy and state tracking
    all_reqs = []
    for r in requests:
        all_reqs.append({
            'arrival': r['arrival_time'],
            'needed': r['tokens_needed'],
            'start_time': None,
            'completion_time': None
        })
        
    all_reqs.sort(key=lambda x: x['arrival'])
    
    time = 0
    active_slots = []
    req_idx = 0
    completed = 0
    total_reqs = len(all_reqs)
    
    while completed < total_reqs:
        # 1. Arrive
        while req_idx < total_reqs and all_reqs[req_idx]['arrival'] <= time:
            queue.append(all_reqs[req_idx])
            req_idx += 1
            
        # 2. Fill
        while len(active_slots) < max_batch_size and queue:
            req = queue.popleft()
            if req['start_time'] is None:
                req['start_time'] = time
            active_slots.append(req)
            
        # 3. Generate & 4. Complete
        next_active = []
        for req in active_slots:
            req['needed'] -= 1
            if req['needed'] == 0:
                req['completion_time'] = time + 1
                completed += 1
            else:
                next_active.append(req)
        active_slots = next_active
        
        # 5. Advance
        if not active_slots and not queue and req_idx < total_reqs:
            time = all_reqs[req_idx]['arrival']
        else:
            time += 1
            
    total_tokens = sum(r['tokens_needed'] for r in requests)
    latencies = [r['completion_time'] - r['arrival'] for r in all_reqs]
    ttfts = [r['start_time'] - r['arrival'] + 1 for r in all_reqs]
    
    return {
        'total_time': time,
        'avg_latency': round(sum(latencies) / total_reqs, 4),
        'avg_ttft': round(sum(ttfts) / total_reqs, 4),
        'throughput': round(total_tokens / time, 4) if time > 0 else 0.0
    }
