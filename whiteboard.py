import jax                                          # noqa: E402
import jax.numpy as jnp

if __name__ == "__main__":
    u = jnp.array([1, 2, 3])
    f_d = jnp.linalg.norm(u)
    b_3d = -u / jnp.linalg.norm(u)
    b_1d = jnp.array([1, 0, 0])
    cross = jnp.cross(b_3d, b_1d)
    b_2d = cross / jnp.linalg.norm(cross)

    R_d = jnp.column_stack((jnp.cross(b_2d, b_3d), b_2d, b_3d))

    print(-f_d*R_d@jnp.array([0, 0, 1]))