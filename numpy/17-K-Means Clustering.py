import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    """
    Implement k-Means clustering algorithm.
    """
    points_arr = np.array(points)
    centroids = np.array(initial_centroids)
    
    for _ in range(max_iterations):
        # Compute distances from each point to each centroid
        # points_arr: (N, D), centroids: (K, D) -> (N, K, D)
        diffs = points_arr[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sum(diffs ** 2, axis=2)
        
        # Find closest centroid for each point
        assignments = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        # Update centroids
        for i in range(k):
            cluster_points = points_arr[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = centroids[i] # keep the same if empty
                
        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-5, atol=1e-5):
            centroids = new_centroids
            break
            
        centroids = new_centroids

    # Format output
    return [tuple(np.round(c, 4)) for c in centroids]
