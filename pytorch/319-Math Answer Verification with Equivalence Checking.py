import re

def verify_math_answer(pred: str, ground_truth: str) -> bool:
    def normalize(s):
        s = s.replace(' ', '').lower()
        s = re.sub(r'\.0+$', '', s)
        return s
    
    return normalize(pred) == normalize(ground_truth)
