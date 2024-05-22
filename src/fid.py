import numpy as np
import torch
from scipy.linalg import sqrtm

def calculate_statistics(vectors):
    mean = np.mean(vectors, axis=0)
    cov = np.cov(vectors, rowvar=False)
    return mean, cov


def calculate_fid(vectors1, vectors2):
    mean1, cov1 = calculate_statistics(vectors1)
    mean2, cov2 = calculate_statistics(vectors2)

    # Calculate the mean difference
    diff = mean1 - mean2
    
    # Calculate the covariance mean product
    covmean, _ = sqrtm(cov1.dot(cov2), disp=False)
    
    # Handle numerical issues with complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate the FID score
    fid = diff.dot(diff) + np.trace(cov1 + cov2 - 2 * covmean)
    return fid



