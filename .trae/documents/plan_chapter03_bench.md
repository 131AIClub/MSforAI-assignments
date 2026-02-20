# Plan: Implement Bench Function for Chapter 03

This plan outlines the steps to implement the `bench` function in `chapter03/startup/main.py`. The goal is to provide a comprehensive test suite for the model components that students are required to implement: `Linear`, `Sigmoid`, `Softmax`, and `CrossEntropyLoss`.

## Objective

Update `chapter03/startup/main.py` to include a functional `bench()` function that verifies the correctness of:

1. **Linear Layer**: Forward pass, backward pass (gradient computation), and parameter updates.
2. **Sigmoid Activation**: Forward pass and backward pass.
3. **Softmax Activation**: Forward pass and backward pass.
4. **CrossEntropyLoss**: Forward pass (loss calculation) and backward pass (gradient w\.r.t logits).

## Implementation Details

### 1. `bench()` Function Structure

The function will perform sequential tests for each component. Each test will:

* Initialize the component.

* Define deterministic input data and expected outputs (where possible).

* Execute the component's methods (`__call__`, `backpropagation`).

* Compare results with expected values using `np.allclose`.

* Print clear "PASS" or "FAIL" messages with error details.

### 2. Component Tests

#### A. Linear Layer Test

* **Setup**: Create `Linear(input_dim=3, output_dim=2)`. Manually set weights `w` and bias `b` to known values (e.g., all ones or sequence).

* **Forward**: Compute `x @ w + b`. Compare with output.

* **Backward**: Pass a dummy gradient `grad_output` (e.g., all ones).

  * Expected `grad_input` = `grad_output @ w.T`.

  * Check if `w` and `b` gradients are computed internally (implied by update).

* **Update**: Check if parameters change after `backpropagation` (since the method signature implies update might happen there or `Linear` stores grads and `update` is separate? Wait, `Linear.backpropagation` in `model.py` signature is `backpropagation(self, grad_output: np.ndarray, lr: float)`. This implies it updates parameters *inside* this method or returns them. The docstring says "update parameters...". I will assume it updates in place).

#### B. Sigmoid Test

* **Forward**: Input `x = np.array([0, 2])`. Expected `y = 1 / (1 + exp(-x))`.

* **Backward**: Input `grad = np.array([1, 1])`. Expected `grad_input = grad * y * (1 - y)`.

#### C. Softmax Test

* **Forward**: Input `x`. Expected `y = exp(x) / sum(exp(x))`. Check if rows sum to 1.

* **Backward**: (Optional/Informational) Since it's often combined with Loss, we primarily test Forward. If tested, we'll check output shape.

#### D. CrossEntropyLoss Test

* **Forward**: Input `pred` and `target` (one-hot). Calculate `-sum(target * log(pred)) / batch_size`.

* **Backward**: Check if it returns `(pred - target) / batch_size` (standard implementation for Softmax+CE).

## Verification

* The `bench()` function is already called in `main()` before training.

* Running `python chapter03/startup/main.py` will execute the tests.

## File to Edit

* `chapter03/startup/main.py`

