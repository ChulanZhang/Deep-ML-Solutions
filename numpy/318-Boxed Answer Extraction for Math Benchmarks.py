def extract_boxed_answer(text: str) -> str:
    start_idx = text.find('\\boxed{')
    if start_idx == -1:
        return ""
    start_idx += 7
    brace_count = 1
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        if brace_count == 0:
            return text[start_idx:i]
    return ""
