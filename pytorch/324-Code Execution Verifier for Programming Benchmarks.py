def verify_code(code_string: str, test_cases: list) -> float:
    passed = 0
    loc = {}
    try:
        exec(code_string, {}, loc)
        funcs = [f for f in loc.values() if callable(f)]
        if not funcs:
            return 0.0
        func = funcs[0]
        
        for case in test_cases:
            inputs, expected = case
            if func(*inputs) == expected:
                passed += 1
    except Exception:
        return 0.0
    
    if len(test_cases) == 0: return 0.0
    return passed / len(test_cases)
