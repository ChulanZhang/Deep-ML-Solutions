import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine similarity of two vectors.
	原理: 计算两个向量夹角的余弦值。公式为 (v1 · v2) / (||v1|| * ||v2||)
	Args:
		v1 (numpy.ndarray): 1D array representing the first vector.
		v2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		float: The cosine similarity of the two vectors (range: -1 to 1).
	"""
	# 1. 计算点积 (Dot Product): 分子部分
	# 反映了向量在方向上的重合程度
	dot_product = np.dot(v1, v2)
	
	# 2. 计算向量的 L2 范数 (模长/长度): 分母部分
	norm_v1 = np.linalg.norm(v1)
	norm_v2 = np.linalg.norm(v2)
	
	# 3. 处理零向量情况
	# 如果任一向量长度为 0，则无法计算夹角，返回 0
	if norm_v1 == 0 or norm_v2 == 0:
		return 0.0
		
	# 4. 计算余弦相似度
	# 余弦值 = 点积 / (长度1 * 长度2)
	return dot_product / (norm_v1 * norm_v2)