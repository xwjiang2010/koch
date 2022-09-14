# from prediction.src.nn import inv_z_transform, test, z_transform
import pytest
from sklearn.utils._testing import assert_allclose_dense_sparse, assert_array_equal


@pytest.mark.skip
def test_inv_z_transform():
    X = np.array([1, 4, 9, 16]).reshape((2, 2))

    # Test that inv_z_transform works correctly
    F = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        inv_kw_args=dict(decimals=3),
    )
    assert_array_equal(
        inv_z_transform(F.transform(X)),
        np.around(np.sqrt(X), decimals=3),
    )
