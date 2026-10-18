def kv_cache_eviction(kv_cache: list, new_tokens: list, max_budget: int) -> list:
    # LRU or FIFO baseline. Let's do simple FIFO eviction if exceeding budget
    cache = list(kv_cache)
    for token in new_tokens:
        cache.append(token)
        if len(cache) > max_budget:
            cache.pop(0)  # Evict oldest
    return cache
