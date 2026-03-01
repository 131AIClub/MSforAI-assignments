
import unittest
import numpy as np
from solution.model import Softmax, CrossEntropyLoss

class TestSoftmaxCrossEntropy(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()

    def test_softmax_numerical_stability(self):
        """Test Softmax with large input values to ensure no overflow"""
        # Case 1: Large positive values
        x_large = np.array([[1000.0, 1001.0, 1002.0]])
        output = self.softmax(x_large)
        
        # Check if output is valid (not NaN, sums to 1)
        self.assertFalse(np.isnan(output).any(), "Softmax output contains NaN for large inputs")
        np.testing.assert_allclose(np.sum(output, axis=1), 1.0, atol=1e-6)
        
        # Check if relative probabilities are preserved (exp(x-max))
        # 1000, 1001, 1002 -> shift by -1002 -> -2, -1, 0
        # exp(-2), exp(-1), exp(0) -> 0.135, 0.368, 1.0
        # sum approx 1.503
        # probs approx 0.09, 0.24, 0.66
        expected_ratios = np.exp(x_large - np.max(x_large, axis=1, keepdims=True))
        expected_probs = expected_ratios / np.sum(expected_ratios, axis=1, keepdims=True)
        np.testing.assert_allclose(output, expected_probs, atol=1e-6)

        # Case 2: Extreme values mixed
        x_extreme = np.array([[1e5, -1e5, 0]])
        output = self.softmax(x_extreme)
        self.assertFalse(np.isnan(output).any())
        # Max is 1e5. Shifted: 0, -2e5, -1e5. 
        # exp(0) = 1. Others approx 0.
        # Prob should be [1, 0, 0]
        expected = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(output, expected, atol=1e-6)

        # Case 3: All Zeros
        x_zeros = np.zeros((1, 5))
        output = self.softmax(x_zeros)
        # Shifted: 0, 0, 0, 0, 0. exp -> 1, 1, 1, 1, 1. sum -> 5. prob -> 0.2
        expected = np.full((1, 5), 0.2)
        np.testing.assert_allclose(output, expected, atol=1e-6)

        # Case 4: Very small negative numbers
        x_neg = np.array([[-1e5, -1e5-1, -1e5-2]])
        output = self.softmax(x_neg)
        # Max is -1e5. Shifted: 0, -1, -2.
        expected_ratios = np.exp([0, -1, -2])
        expected_probs = expected_ratios / np.sum(expected_ratios)
        np.testing.assert_allclose(output, expected_probs.reshape(1, -1), atol=1e-6)

    def test_softmax_gradient(self):
        """Numerical gradient check for Softmax"""
        batch_size = 2
        num_classes = 3
        x = np.random.randn(batch_size, num_classes)
        # Random gradient from next layer
        grad_output = np.random.randn(batch_size, num_classes)
        
        # Forward pass to set cache
        _ = self.softmax(x)
        
        # Analytical gradient (using backpropagation)
        analytical_grad = self.softmax.backpropagation(grad_output, lr=0.1)
        
        # Numerical gradient checking
        epsilon = 1e-6
        numerical_grad = np.zeros_like(x)
        
        # Compute numerical gradient for each element of input x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = x[idx]
            
            # f(x + eps)
            x[idx] = orig_val + epsilon
            y_plus = self.softmax(x)
            # We treat 'loss' as dot product with grad_output for chain rule verification
            # If Loss = sum(y * grad_output), then dL/dy = grad_output
            # So dL/dx = sum(dL/dy * dy/dx) which is what backprop computes
            l_plus = np.sum(y_plus * grad_output)
            
            # f(x - eps)
            x[idx] = orig_val - epsilon
            y_minus = self.softmax(x)
            l_minus = np.sum(y_minus * grad_output)
            
            # Central difference
            numerical_grad[idx] = (l_plus - l_minus) / (2 * epsilon)
            
            # Restore original value
            x[idx] = orig_val
            it.iternext()
        
        # Compare analytical and numerical gradients
        # Softmax gradient can be small, so check absolute tolerance as well
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-5,
                                   err_msg="Softmax gradient check failed")

    def test_crossentropy_log_stability(self):
        """Test CrossEntropyLoss with zero or near-zero predictions"""
        # Case 1: Perfect prediction (log(1) = 0)
        # Probabilities [0.0, 1.0], target [0, 1]
        pred = np.array([[0.0, 1.0]])
        target = np.array([[0, 1]])
        loss = self.loss_fn(pred, target)
        self.assertAlmostEqual(loss, 0.0, places=5)
        
        # Case 2: Zero probability for correct class (log(0) -> inf, should be clamped)
        # Probabilities [1.0, 0.0], target [0, 1] -> Correct class has 0 prob
        pred = np.array([[1.0, 0.0]])
        target = np.array([[0, 1]]) 
        loss = self.loss_fn(pred, target)
        
        # Should be roughly -log(epsilon)
        # Implementation uses np.clip(pred, eps, 1-eps)
        # so loss = -log(eps)
        expected_loss = -np.log(self.loss_fn.epsilon)
        
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
        # Allow some floating point error
        self.assertTrue(loss > 10.0, "Loss for zero probability should be large")
        # Check against expected value
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_crossentropy_gradient(self):
        """Numerical gradient check for CrossEntropyLoss"""
        batch_size = 2
        num_classes = 3
        
        # Random probabilities (must sum to 1 for valid softmax output simulation, but CE works generally)
        # Use simple values
        pred = np.array([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        target = np.array([[0, 1, 0], [0, 0, 1]]) # One-hot targets
        
        # Forward pass to set cache
        _ = self.loss_fn(pred, target)
        
        # Analytical gradient
        # dL/d_pred
        analytical_grad = self.loss_fn.backpropagation(lr=0.1)
        
        # Numerical gradient checking
        epsilon = 1e-6
        numerical_grad = np.zeros_like(pred)
        
        it = np.nditer(pred, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = pred[idx]
            
            # f(x + eps)
            pred[idx] = orig_val + epsilon
            l_plus = self.loss_fn(pred, target)
            
            # f(x - eps)
            pred[idx] = orig_val - epsilon
            l_minus = self.loss_fn(pred, target)
            
            # Central difference
            numerical_grad[idx] = (l_plus - l_minus) / (2 * epsilon)
            
            # Restore
            pred[idx] = orig_val
            it.iternext()
            
        # Compare
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-5,
                                   err_msg="CrossEntropyLoss gradient check failed")

    def test_combined_gradient_correctness(self):
        """
        Check if manual chain rule (Softmax backprop -> CE backprop) matches 
        the simplified formula (pred - target)/N.
        This verifies that our separate implementations are mathematically consistent.
        """
        batch_size = 5
        num_classes = 4
        x = np.random.randn(batch_size, num_classes)
        
        # Random one-hot targets
        targets_idx = np.random.randint(0, num_classes, size=batch_size)
        targets = np.zeros((batch_size, num_classes))
        targets[np.arange(batch_size), targets_idx] = 1
        
        # Forward
        pred = self.softmax(x)
        _ = self.loss_fn(pred, targets)
        
        # Backward chain
        grad_loss = self.loss_fn.backpropagation(lr=0.1) # dL/d_pred
        grad_x = self.softmax.backpropagation(grad_loss, lr=0.1) # dL/dx
        
        # Simplified formula for Softmax + CE combination
        # dL/dx = (pred - target) / batch_size
        expected_grad_x = (pred - targets) / batch_size
        
        np.testing.assert_allclose(grad_x, expected_grad_x, rtol=1e-5, atol=1e-6,
                                   err_msg="Combined gradient chain rule mismatch")

if __name__ == '__main__':
    unittest.main()
