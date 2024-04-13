# import numpy as np

if __name__ == "__main__":

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Generate a random rotation matrix
    random_rotation = R.random().as_matrix()
    print(random_rotation)