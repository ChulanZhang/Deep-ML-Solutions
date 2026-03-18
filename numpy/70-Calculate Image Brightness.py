def calculate_brightness(img):
    if not img or not img[0]:
        return -1
    
    cols = len(img[0])
    total = 0
    count = 0
    
    for row in img:
        if len(row) != cols:
            return -1
        for p in row:
            if not (0 <= p <= 255):
                return -1
            total += p
            count += 1
            
    if count == 0:
        return -1
    return round(total / count, 2)
