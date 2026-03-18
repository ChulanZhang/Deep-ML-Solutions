def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    '''
    Calculate Mean by Row or Column
    '''
    if not matrix or not matrix[0]:
        return []
        
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    
    if mode == 'row':
        # 行模式：遍历每一行，计算该行所有列的和
        means = []
        for i in range(n_rows):
            row_sum = sum(matrix[i])
            means.append(row_sum / n_cols)
        return means
        
    elif mode == 'column':
        # 列模式（面试标准思路）：
        # 外层循环选定“第 j 列”，内层循环跑完“所有 i 行”
        means = []
        for j in range(n_cols): # 每一列
            col_sum = 0
            for i in range(n_rows): # 每一行
                col_sum += matrix[i][j] # 累加第 j 列的元素
            means.append(col_sum / n_rows)
        return means
        
    else:
        raise ValueError("Mode must be 'row' or 'column'")