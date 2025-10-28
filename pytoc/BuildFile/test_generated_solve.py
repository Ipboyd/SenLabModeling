import numpy as np
from numpy.testing import assert_array_equal
import pytest
from generated_solve_file import solve_run

def test_solve_run_basic():
    """Basic test with zero inputs"""
    shape = (1, 29800, 1)
    on_input = np.zeros(shape, dtype=np.float64)
    off_input = np.zeros(shape, dtype=np.float64)
    noise_token = np.zeros(shape, dtype=np.float64)
    
    result = solve_run(on_input, off_input, noise_token)
    assert result.shape == (1, 10, 1, 29800)
    assert result.dtype == np.int8

def test_solve_run_input_validation():
    """Test input validation"""
    wrong_shape = (2, 29800, 1)
    with pytest.raises(ValueError):
        solve_run(np.zeros(wrong_shape), np.zeros((1, 29800, 1)), np.zeros((1, 29800, 1)))
    
    wrong_type = np.zeros((1, 29800, 1), dtype=np.float32)
    with pytest.raises(TypeError):
        solve_run(wrong_type, np.zeros((1, 29800, 1)), np.zeros((1, 29800, 1)))

def test_solve_run_with_input():
    """Test with non-zero input"""
    shape = (1, 29800, 1)
    on_input = np.random.rand(*shape).astype(np.float64)
    off_input = np.random.rand(*shape).astype(np.float64)
    noise_token = np.random.rand(*shape).astype(np.float64)
    
    result = solve_run(on_input, off_input, noise_token)
    assert result.shape == (1, 10, 1, 29800)
    assert result.dtype == np.int8
    # Values should be either 0 or 1 since it's a spike train
    assert np.all(np.logical_or(result == 0, result == 1))

if __name__ == '__main__':
    # Run tests
    print("Running basic test...")
    test_solve_run_basic()
    print("Running input validation test...")
    test_solve_run_input_validation()
    print("Running input test...")
    test_solve_run_with_input()
    print("All tests passed!")