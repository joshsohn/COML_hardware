import numpy as np

if __name__ == "__main__":

    # Define the matrix R
    R = np.array([
        [9.9999988e-01, -1.0091504e-05, 5.7971262e-04],
        [-1.0659045e-05, -9.9999952e-01, 9.7901304e-04],
        [5.7970244e-04, -9.7901921e-04, -9.9999934e-01]
    ])

    # Check orthogonality by verifying R * R^T is the identity matrix
    orthogonal_check = np.allclose(R @ R.T, np.eye(3))

    # Check determinant is +1
    determinant_check = np.isclose(np.linalg.det(R), 1.0)

    print(orthogonal_check, determinant_check)