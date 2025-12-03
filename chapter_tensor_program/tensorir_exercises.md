## Exercises for TensorIR

```{.python .input n=0}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
import IPython
```

### Section 1: How to Write TensorIR

In this section, let's try to write TensorIR manually according to high-level instructions (e.g., Numpy or Torch). First, we give an example of element-wise add function, to show what should we do to write a TensorIR function.

#### Example: Element-wise Add

First, let's try to use Numpy to write an element-wise add function.

```{.python .input n=1}
# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
```

```{.python .input n=2}
# numpy version
c_np = a + b
c_np
```

Before we directly write TensorIR, we should first translate high-level computation abstraction (e.g., `ndarray + ndarray`) to low-level python implementation (standard for loops with element access and operation)

Notably, the initial value of the output array (or buffer) is not always `0`. We need to write or initialize it in our implementation, which is important for reduction operator (e.g. matmul and conv)

```{.python .input n=3}
# low-level numpy version
def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
  for i in range(4):
    for j in range(4):
      c[i, j] = a[i, j] + b[i, j]
c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
c_lnumpy
```

Now, let's take a further step: translate low-level NumPy implementation into TensorIR. And compare the result with it comes from NumPy.

```{.python .input n=4}
# TensorIR version
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.compile(MyAdd, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

Here, we have finished the TensorIR function. Please take your time to finish the following exercises

#### Exercise 1: Broadcast Add

Please write a TensorIR function that adds two arrays with broadcasting.

```{.python .input n=5}
# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
```

```{.python .input n=6}
# numpy version
c_np = a + b
c_np
```

Please complete the following Module `MyAdd` and run the code to check your implementation.

```python
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add():
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.compile(MyAdd, target="llvm")
a_tvm = tvm.runtime.tensor(a)
b_tvm = tvm.runtime.tensor(b)
c_tvm = tvm.runtime.tensor(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

#### Exercise 2: 2D Convolution

Then, let's try to do something challenging: 2D convolution, which is a common operation in image processing.

Here is the mathematical definition of convolution with NCHW layout:

$$Conv[b, k, i, j] =
    \sum_{di, dj, q} A[b, q, strides * i + di, strides * j + dj] * W[k, q, di, dj]$$
, where, `A` is the input tensor, `W` is the weight tensor, `b` is the batch index, `k` is the out channels, `i` and `j` are indices for image hight and width, `di` and `dj` are the indices of the weight, `q` is the input channel, and `strides` is the stride of the filter window.

In the exercise, we pick a small and simple case with `stride=1, padding=0`.

```{.python .input n=7}
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)
```

```{.python .input n=8}
# torch version
import torch
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch
```

Please complete the following Module `MyConv` and run the code to check your implementation.

```python
@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv():
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.compile(MyConv, target="llvm")
data_tvm = tvm.runtime.tensor(data)
weight_tvm = tvm.runtime.tensor(weight)
conv_tvm = tvm.runtime.tensor(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
```

### Section 2: How to Transform TensorIR

In the lecture, we learned that TensorIR is not only a programming language but also an abstraction for program transformation. In this section, let's try to transform the program. We take `bmm_relu` (`batched_matmul_relu`) in our studies, which is a variant of operations that common appear in models such as transformers.

#### Parallel, Vectorize and Unroll
First, we introduce some new primitives, `parallel`, `vectorize` and `unroll`. These three primitives operate on loops to indicate how this loop executes. Here is the example:

```{.python .input n=9}
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
IPython.display.Code(sch.mod.script(), language="python")
```

#### Exercise 3: Transform a batch matmul program
Now, let's go back to the `bmm_relu` exercise. First, Let's see the definition of `bmm`:

- $Y_{n, i, j} = \sum_k A_{n, i, k} \times B_{n, k, j}$
- $C_{n, i, j} = \mathbb{relu}(Y_{n,i,j}) = \mathbb{max}(Y_{n, i, j}, 0)$

It's your time to write the TensorIR for `bmm_relu`. We provide the lnumpy func as hint:

```{.python .input n=10}
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
```

```python
@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu():
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    # TODO
    ...

sch = tvm.tir.Schedule(MyBmmRelu)
IPython.display.Code(sch.mod.script(), language="python")
# Also please validate your result
```

In this exercise, let's focus on transform the original program to a specific target. Note that the target program may not be the best one due to different hardware. But this exercise aims to let students understand how to transform the program to a wanted one. Here is the target program:

```{.python .input n=11}
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))
```

Your task is to transform the original program to the target program.

```python
sch = tvm.tir.Schedule(MyBmmRelu)
# TODO: transformations
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
...

# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)
...

# Step 3. Organize the loops
k0, k1 = sch.split(k, ...)
sch.reorder(...)
sch.compute_at/reverse_compute_at(...)
...

# Step 4. decompose reduction
Y_init = sch.decompose_reduction(Y, ...)
...

# Step 5. vectorize / parallel / unroll
sch.vectorize(...)
sch.parallel(...)
sch.unroll(...)
...

IPython.display.Code(sch.mod.script(), language="python")
```

**OPTIONAL** If we want to make sure the transformed program is exactly the same as the given target, we can use `assert_structural_equal`. Note that this step is an optional step in this exercise. It's good enough if you transformed the program **towards** the target and get performance improvement.

```python
tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")
```

#### Build and Evaluate

Finally we can evaluate the performance of the transformed program.

```python
before_rt_lib = tvm.compile(MyBmmRelu, target="llvm")
after_rt_lib = tvm.compile(sch.mod, target="llvm")
a_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.runtime.tensor(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.mod.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
```
