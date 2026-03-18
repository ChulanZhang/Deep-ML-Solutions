def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	'''
	Calculate Eigenvalues of a Matrix (2x2)
	'''
	# 对于 2x2 矩阵: [[a, b], [c, d]]
	a = matrix[0][0]
	b = matrix[0][1]
	c = matrix[1][0]
	d = matrix[1][1]

	# 1. 计算迹 (Trace): tr(A) = a + d
	trace = a + d

	# 2. 计算行列式 (Determinant): det(A) = ad - bc
	determinant = a * d - b * c

	# 3. 特征方程: λ^2 - trace*λ + determinant = 0
	# 使用求根公式: λ = [trace ± sqrt(trace^2 - 4*determinant)] / 2
	discriminant = (trace**2 - 4 * determinant)**0.5

	lambda1 = (trace + discriminant) / 2
	lambda2 = (trace - discriminant) / 2

	return [lambda1, lambda2]