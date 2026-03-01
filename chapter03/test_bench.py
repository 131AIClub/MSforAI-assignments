
import pytest
import numpy as np
import sys
import os

# Dynamically import based on pytest mode (handled in conftest.py via sys.path)
# However, if we hardcode 'from startup.model import ...', it defeats the purpose.
# We need to import 'model' which will resolve to either startup/model.py or solution/model.py
try:
    from model import Linear, Sigmoid, Softmax, CrossEntropyLoss
except ImportError:
    # Fallback or error handling if model is not found in path
    # This might happen if conftest.py hasn't run yet or path isn't set
    # But conftest.py runs before test collection usually.
    # Let's try to be robust.
    pass

@pytest.fixture
def layer(request, mode):
    # Dynamically import the class based on request.param (class name)
    # and the mode (startup vs solution)
    
    # Reload module to ensure we get the right one
    if 'model' in sys.modules:
        del sys.modules['model']
        
    try:
        import model
        from importlib import reload
        reload(model)
        
        layer_cls = getattr(model, request.param)
        
        if request.param == 'Linear':
            return layer_cls(5, 2)
        else:
            return layer_cls()
    except ImportError:
        pytest.fail(f"Could not import model from path: {sys.path}")

@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")

# Numerical gradient checking utility
def compute_numerical_gradient(layer, x, epsilon=1e-5):
    """
    Computes numerical gradient for a layer using finite differences.
    For layers with parameters (like Linear), it checks gradients w.r.t input.
    Use similar logic for weights if needed, but here we focus on input gradients
    passed back to previous layers.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    # We need a dummy gradient from next layer for backprop
    # For simplicity in numerical check of activation functions:
    # dL/dx = (dL/dy) * (dy/dx)
    # If we set dL/dy = 1, then we are checking dy/dx directly.
    # However, backpropagation usually takes grad_output.
    # So we compare layer.backpropagation(grad_output) with numerical approximation of (layer(x+eps) - layer(x-eps))/2eps * grad_output
    
    # Actually, standard numerical gradient check for backprop:
    # 1. Choose a random grad_output (dL/dy)
    # 2. Compute analytical grad_input (dL/dx) using layer.backpropagation(grad_output)
    # 3. Compute numerical grad_input:
    #    For each element x_i:
    #      x_plus = x.copy(); x_plus[i] += epsilon
    #      y_plus = layer(x_plus)
    #      loss_plus = np.sum(y_plus * grad_output) # We treat 'loss' as dot product for verification
    #      
    #      x_minus = x.copy(); x_minus[i] -= epsilon
    #      y_minus = layer(x_minus)
    #      loss_minus = np.sum(y_minus * grad_output)
    #      
    #      num_grad_i = (loss_plus - loss_minus) / (2 * epsilon)
    # 4. Compare analytical and numerical gradients.
    
    return grad

def check_gradient(layer, x, grad_output, epsilon=1e-5):
    # 1. Numerical gradient
    # We compute this FIRST because layer.backpropagation might update weights (e.g. Linear layer).
    # If we ran backpropagation first, the numerical gradient would be computed using updated weights,
    # leading to a mismatch with the analytical gradient (which was computed using original weights).
    numerical_grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        original_val = x[idx]
        
        # f(x + eps)
        x[idx] = original_val + epsilon
        y_plus = layer(x)
        l_plus = np.sum(y_plus * grad_output)
        
        # f(x - eps)
        x[idx] = original_val - epsilon
        y_minus = layer(x)
        l_minus = np.sum(y_minus * grad_output)
        
        # Restore
        x[idx] = original_val
        
        # Central difference
        numerical_grad[idx] = (l_plus - l_minus) / (2 * epsilon)
        
        it.iternext()

    # 2. Analytical gradient
    # Ensure forward pass is called first with the clean x to set cache correctly
    # (The numerical loop's last iteration left the cache corresponding to a perturbed input)
    _ = layer(x) 
    analytical_grad = layer.backpropagation(grad_output, lr=0.01)
        
    return analytical_grad, numerical_grad

# --- Linear Layer Tests ---
class TestLinear:
    @pytest.fixture
    def layer(self, mode):
        # Dynamically import Linear based on mode
        # Reload module to ensure we get the right one
        if 'model' in sys.modules:
            del sys.modules['model']
            
        try:
            import model
            from importlib import reload
            reload(model)
            return model.Linear(3, 2)
        except ImportError:
             pytest.fail(f"Could not import model from path: {sys.path}")

    def test_forward_shape(self, layer):
        x = np.random.randn(5, 3)
        out = layer(x)
        assert out.shape == (5, 2), f"Expected shape (5, 2), got {out.shape}"

    def test_forward_value(self, layer):
        x = np.array([[1.0, 2.0, 3.0]])
        layer.w = np.ones((3, 2))
        layer.b = np.zeros(2)
        # 1*1 + 2*1 + 3*1 = 6
        expected = np.array([[6.0, 6.0]])
        out = layer(x)
        np.testing.assert_allclose(out, expected, rtol=1e-5, err_msg="Forward pass calculation incorrect")

    def test_backward_gradient(self, layer):
        x = np.random.randn(4, 3)
        grad_output = np.random.randn(4, 2)
        
        # Initial forward to set cache
        layer(x)
        
        ana_grad, num_grad = check_gradient(layer, x, grad_output)
        
        # Linear layer gradient is usually very precise, so tight tolerance
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-4, atol=1e-5, 
                                   err_msg="Gradient check failed for Linear layer")

    def test_parameter_update(self, layer):
        # Check if weights and bias are updated correctly
        x = np.array([[1.0, 2.0, 3.0]])
        grad_output = np.array([[0.1, 0.2]])
        lr = 0.1
        
        # Store initial
        w_old = layer.w.copy()
        b_old = layer.b.copy()
        
        # Run backward
        _ = layer(x) # Cache input
        layer.backpropagation(grad_output, lr)
        
        # Expected updates
        # dL/dw = x.T @ grad_output
        grad_w = x.T @ grad_output
        grad_b = np.sum(grad_output, axis=0)
        
        expected_w = w_old - lr * grad_w
        expected_b = b_old - lr * grad_b
        
        np.testing.assert_allclose(layer.w, expected_w, atol=1e-5, err_msg="Weight update incorrect")
        np.testing.assert_allclose(layer.b, expected_b, atol=1e-5, err_msg="Bias update incorrect")

    def test_numerical_stability_large_input(self, layer):
        # Very large inputs shouldn't crash (though they might cause overflow in subsequent layers)
        x = np.array([[1e10, 1e10, 1e10]])
        try:
            out = layer(x)
            assert not np.isnan(out).any(), "Output contains NaN with large input"
            assert not np.isinf(out).any(), "Output contains Inf with large input"
        except Exception as e:
            pytest.fail(f"Large input caused exception: {e}")

# --- Sigmoid Layer Tests ---
class TestSigmoid:
    @pytest.fixture
    def layer(self, mode):
        # Dynamically import based on mode
        if 'model' in sys.modules:
            del sys.modules['model']
            
        try:
            import model
            from importlib import reload
            reload(model)
            return model.Sigmoid()
        except ImportError:
             pytest.fail(f"Could not import model from path: {sys.path}")

    def test_forward_value(self, layer):
        x = np.array([[0.0], [100.0], [-100.0]])
        out = layer(x)
        # sig(0) = 0.5, sig(100) -> 1, sig(-100) -> 0
        expected = np.array([[0.5], [1.0], [0.0]])
        np.testing.assert_allclose(out, expected, atol=1e-5, err_msg="Sigmoid output incorrect")

    def test_gradient(self, layer):
        x = np.random.randn(5, 3)
        grad_output = np.random.randn(5, 3)
        
        ana_grad, num_grad = check_gradient(layer, x, grad_output)
        np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-4, atol=1e-5)


# --- Softmax Layer Tests ---
class TestSoftmax:
    @pytest.fixture
    def layer(self, mode):
        # Dynamically import based on mode
        if 'model' in sys.modules:
            del sys.modules['model']
            
        try:
            import model
            from importlib import reload
            reload(model)
            return model.Softmax()
        except ImportError:
             pytest.fail(f"Could not import model from path: {sys.path}")

    def test_output_sum_to_one(self, layer):
        x = np.random.randn(5, 10)
        out = layer(x)
        sums = np.sum(out, axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6, err_msg="Softmax outputs do not sum to 1")

    def test_stability_large_values(self, layer):
        # Softmax should handle large values by subtracting max (numerical stability trick)
        x = np.array([[1000.0, 1001.0, 1002.0]])
        # Expected: exp(0)/sum(...) approx similar to smaller values shifted
        # If naive implementation exp(1000) will overflow
        try:
            out = layer(x)
            assert not np.isnan(out).any(), "Softmax produced NaN with large inputs (check for overflow protection)"
            np.testing.assert_allclose(np.sum(out, axis=1), 1.0, atol=1e-6)
        except RuntimeWarning:
            pytest.fail("RuntimeWarning caught - likely overflow in Softmax")
        except Exception as e:
            pytest.fail(f"Exception in Softmax with large inputs: {e}")

    def test_gradient(self, layer):
        """Numerical gradient check for Softmax"""
        batch_size = 2
        num_classes = 3
        x = np.random.randn(batch_size, num_classes)
        grad_output = np.random.randn(batch_size, num_classes)
        
        # Analytical gradient
        _ = layer(x) # forward to cache
        analytical_grad = layer.backpropagation(grad_output, lr=0.1)
        
        # Check shape
        assert analytical_grad.shape == x.shape, f"Gradient shape mismatch. Expected {x.shape}, got {analytical_grad.shape}"
        
        # Numerical gradient
        ana_grad_check, num_grad_check = check_gradient(layer, x, grad_output)
        
        # Softmax gradient can be small, so check absolute tolerance as well
        np.testing.assert_allclose(analytical_grad, num_grad_check, rtol=1e-4, atol=1e-5,
                                   err_msg="Softmax gradient check failed")

# --- CrossEntropyLoss Tests ---
class TestCrossEntropyLoss:
    @pytest.fixture
    def layer(self, mode):
        # Dynamically import based on mode
        if 'model' in sys.modules:
            del sys.modules['model']
            
        try:
            import model
            from importlib import reload
            reload(model)
            return model.CrossEntropyLoss()
        except ImportError:
             pytest.fail(f"Could not import model from path: {sys.path}")

    def test_loss_value(self, layer):
        # pred: 0.5, true: 1 -> -log(0.5) approx 0.693
        pred = np.array([[0.5, 0.5]])
        target = np.array([[1, 0]]) # class 0
        loss = layer(pred, target)
        expected = -np.log(0.5)
        np.testing.assert_allclose(loss, expected, atol=1e-5)

    def test_stability_zero_log(self, layer):
        # pred has 0, should rely on epsilon
        pred = np.array([[0.0, 1.0]])
        target = np.array([[1, 0]]) # correct class has 0 prob -> infinite loss
        # Implementation should clamp or add epsilon
        loss = layer(pred, target)
        assert not np.isnan(loss), "Loss is NaN for zero probability"
        assert not np.isinf(loss), "Loss is Inf (should be large finite number)"

    def test_gradient(self, layer):
        # Update test to reflect rigorous implementation of CrossEntropyLoss gradient
        # dL/d_pred = -target / pred / N
        
        pred = np.array([[0.7, 0.3], [0.2, 0.8]])
        target = np.array([[1, 0], [0, 1]])
        
        # Call forward to cache (if needed)
        _ = layer(pred, target)
        
        grad = layer.backpropagation(lr=0.1)
        
        # Expected gradient: - (target / pred) / batch_size
        batch_size = pred.shape[0]
        # pred is clipped in implementation, so we should calculate expected with that in mind
        # or use close values
        expected = - (target / pred) / batch_size
        
        # Note: zeros in target result in 0 gradient contribution for that class in this formula
        # The implementation returns -0.0 for those entries, which is fine.
        
        np.testing.assert_allclose(grad, expected, atol=1e-5, err_msg="Gradient incorrect for separate CrossEntropyLoss")
