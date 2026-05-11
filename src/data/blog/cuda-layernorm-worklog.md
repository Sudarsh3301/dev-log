---
title: "Optimizing a Layer Normalization Kernel with CUDA: a Worklog"
author: Sudarsh
pubDatetime: 2025-02-17T00:00:00.000Z
tags:
  - cuda
  - gpu
  - kernels
description: An iterative guide to writing and optimizing a CUDA layer normalization kernel — from a naive single-thread implementation to vectorized loads — benchmarked against PyTorch.
---

Layer normalization is a data preprocessing technique used in deep learning to stabilize training data. When we train a neural network on a dataset, most of the time, the data is on different scales. For example, let's take a dataset of employees at some company, where the two input features are age and salary. Age data ranges from 20–50 while salary data can range from 50,000 to 100,000. Totally different scales. Normalizing helps the input features be at the same scale.

In this blog, I will iteratively optimize a layer normalization kernel written in CUDA, from scratch, by learning and using GPU optimizing techniques including memory coalescing, shuffling and vectorized loading. Let's see if we can beat PyTorch's implementation of layer norm. I'm using NVIDIA GeForce RTX 4050 GPU for this implementation. You can find full code in my [GitHub](https://github.com/aryagxr/cuda/tree/main/layernorm).

This is meant to be a fun, code-along blog. Let's get started!

## Layer Norm Math: Under the Hood

The math for layer norm is fairly simple. We calculate the mean $\mu$ and variance $\sigma^2$ for each input feature $X_{ij}$ sequentially.

Consider each row of a matrix $X$ to be a feature. Layer norm ensures each feature has a mean of $0$ and variance of $1$.

A very small value $\epsilon$ is added to prevent division by zero.

The formula is as follows:

$$
X_{ij,\text{normalized}}
=
\frac{X_{ij} - \mu_{\text{row}_i}}
{\sqrt{\sigma^2_{\text{row}_i} + \epsilon}}
$$

To compute mean and variance for each row, we apply the formulas:

$$
\mu_i
=
\frac{1}{n}
\sum_{j=1}^{n} X_{ij}
$$

$$
\sigma_i^2
=
\frac{1}{n}
\sum_{j=1}^{n}
(X_{ij} - \mu_i)^2
$$

Visually, assume a $3 \times 3$ input matrix:

$$
X
=
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

To calculate the layer norm on the first row:

$$
X_{\text{row1}} = [1,2,3]
$$

Find the mean:

$$
\mu_1
=
\frac{1+2+3}{3}
=
2
$$

Compute the variance:

$$
\sigma_1^2
=
\frac{
(1-2)^2 + (2-2)^2 + (3-2)^2
}{3}
=
\frac{2}{3}
$$

Normalize each value:

$$
\hat{X}_{11}
=
\frac{1-2}
{\sqrt{\frac{2}{3}+\epsilon}}
$$

$$
\hat{X}_{12}
=
\frac{2-2}
{\sqrt{\frac{2}{3}+\epsilon}}
$$

$$
\hat{X}_{13}
=
\frac{3-2}
{\sqrt{\frac{2}{3}+\epsilon}}
$$

Similarly, compute mean, variance and layer norm for every row to achieve the following output matrix.

Throughout this worklog, assume epsilon:

$$
\epsilon = 10^{-6}
$$

Final normalized output:

$$
X_{\text{norm}}
=
\begin{bmatrix}
-1.2247 & 0 & 1.2247 \\
-1.2247 & 0 & 1.2247 \\
-1.2247 & 0 & 1.2247
\end{bmatrix}
$$
## PyTorch Benchmark

To begin, let's see how fast a layer norm implementation is in PyTorch for a 1024 × 1024 matrix. We will use this same input dimension for all kernels.

```python
import torch
import torch.nn as nn
import time

m, n = 1024, 1024
# input matrix is 1,2,3,4...1048576
input = torch.arange(1, m*n+1).reshape(m,n).float()

# LayerNorm
layer_norm = nn.LayerNorm(n, elementwise_affine=False, eps=1e-6).cuda()

# warm up GPU
for i in range(10):
    output = layer_norm(input.cuda())

# measure time
start = time.time()
for i in range(1000):
    output = layer_norm(input.cuda())
torch.cuda.synchronize()
end = time.time()

pytorch_time = (end - start)/1000
print(f"PyTorch LayerNorm time: {pytorch_time * 1000:.4f} ms")
```

**Output:** PyTorch takes around 0.4 ms to compute layer norm on a 1024 × 1024 matrix.

```
PyTorch LayerNorm time: 0.4447 ms
```


## Kernel 1: Naive Layer Normalization

The first kernel is going to be a naive implementation, where we replicate the formulas shown in the example above. For this approach, one thread in a block normalizes one row. When we invoke the kernel with `__global__`, the execution launches a grid of threads where each thread processes a single row of the input matrix.

This line of code assigns the row index that the current thread will process:

```cpp
int row = threadIdx.x + (blockDim.x * blockIdx.x);
```

![Naive kernel thread-to-row mapping](/ln_images/image1.png)

Once the threads are assigned to their rows, there are three stages to the layer norm computation — mean, variance, and putting them both together to find the norm. Let's analyse the three loops:

- **Loop 1:** The thread loops over all elements in its assigned row. It reads each element from global memory and accumulates the sum. The sum is then divided by `n` total number of columns to find the mean.
- **Loop 2:** The same thread loops through the row again, this time to compute variance.
- **Loop 3:** Finally, the thread loops through the row again, subtracting the mean and dividing by the standard deviation — which is the square root of variance.

Normalizing code:

```cpp
// compute mean
for(int col = 0; col < n; col++){
    int idx = row * n + col;
    mean += X[idx]; // reading from global memory
}
mean /= n;

// compute variance
for(int col = 0; col < n; col++){
    int idx = row * n + col;
    var += (X[idx] - mean) * (X[idx] - mean);
}
var /= n;

// normalize each row
float stddev = sqrt(var + EPSILON);
for(int col = 0; col < n; col++){
    int idx = row * n + col;
    P[idx] = (X[idx] - mean) / stddev;
}
```

As you may notice, at each for loop, the thread reads each element from global memory `X[idx]`. This means the thread accesses the input row from global memory three times, causing high traffic. We will learn how to optimize this in the next kernel.

Since we have 1024 rows, we need 1024 threads per block. We can launch our kernel as:

```cpp
dim3 threadsPerBlock(1024); // 1024 rows
dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);

// kernel launch
naive_layernorm<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
```

The kernel performance is as follows:

```
Naive Kernel Execution time: 2.3886 ms
```

As expected, this naive implementation is around 2 ms slower than PyTorch. Let's do better.


## Kernel 2: Shared Memory Reductions and Coalescing

In this kernel, let's walk through how to reduce frequent global memory access by using shared memory instead.

In GPU memory architecture, it is faster to access shared memory due to lower latency. Each block has its own shared memory and all threads in a block can access it. And all blocks can access the same global memory. Each thread has access to its own unique register.

For this implementation, let's assign one block per row (as opposed to one thread per row in the naive kernel). These lines of code do that:

```cpp
// one block per row
int row = blockIdx.x;
int tidx = threadIdx.x;
```

![One block per row mapping](/ln_images/image2.png)

A more efficient way to load from global memory is to use **memory coalescing** — having consecutive threads access consecutive memory addresses.

Take a look at the diagram below. We will be using 256 threads per block. So threads `t0–t255` will load consecutive elements 1–256. In the second iteration, threads `t0–t255` will then load consecutive elements 257–511 and so on, until each thread has loaded 4 elements spaced by `blockDim.x = 256`.

![Memory coalescing diagram](/ln_images/image3.png)

Note that this diagram illustrates one input row; there are `m = 1024` of these blocks processing each row of the input. This is done in the following code snippet:

```cpp
for(int i = tidx; i < n; i+=blockDim.x){
    float a = row_in[i]; // load from global mem into register
    lmean += a;          // sum for now
    lvar += (a*a);
}

smem[tidx] = lmean; // contains the sum of all values loaded by thread tidx
```

Each thread accumulates the sum of the elements it loaded and stores it in shared memory `smem`.

![Shared memory accumulation](/ln_images/image4.png)

Now that we have all the local sums, it's time for **reductions**. We need to find the total sum of all elements in shared memory so we can compute the mean. This is done in log(n) steps as we reduce hierarchically. Eventually, the global sum ends up in the first index `smem[0]`. In the example below, assume the 8 random values are the local sums each thread just computed.

![Reduction diagram](/ln_images/image5.png)

`stride` begins as half of `blockDim.x`, and is halved as we iterate. The thread elements within the first stride are added to the thread elements at `tidx + stride`. As you can see, the final sum accumulates at the first index of shared memory. We can then divide by `n` to compute the global mean for each row, followed by the global variance similarly.

```cpp
for(int stride = blockDim.x / 2; stride > 0; stride /=2){
    if(tidx < stride){
        smem[tidx] += smem[tidx + stride]; // sum ends up in index 0
    }
}
float gmean = smem[0] / n;
```

Once we have our global mean and variance values, we can finally compute the layer norm:

```cpp
// normalize and store outputs
for(int i = tidx; i < n; i += blockDim.x){
    row_out[i] = (row_in[i] - gmean) * stddev;
}
```

Each thread `tidx` normalizes and stores its assigned elements and writes back to global memory `row_out[i]` in a coalesced manner. The output performance of this kernel is:

```
Reduction Kernel Execution time: 0.08168 ms
```

We are already more efficient than PyTorch! But we can still do better.


## Kernel 3: Warp Level Shuffle Functions

For this implementation, let's further optimize by using registers at the warp level instead of shared memory. Accessing registers is faster than accessing shared memory.

In GPU programming, **warps** are a group of (usually 32) threads executed in parallel. In our case, we use 256 threads per block, so number of warps = `blockDim.x / warp_size` = 256/32 = 8 warps per block.

![Warp diagram](/ln_images/image6.png)

Similar to kernel 2, we load the input values from global memory into registers using memory coalescing:

```cpp
for(int i = tidx; i < n; i += blockDim.x){
    float a = row_in[i];
    lmean += a; // lmean is just the sum for now (will divide later)
    lvar += (a*a);
}
__syncthreads();

// store in register instead of smem
float lrmean = lmean;
```

![Register accumulation vs shared memory](/ln_images/image7.png)

The key difference from kernel 2: instead of storing local sums into shared memory (`smem[tidx] = lmean`) ❌, we store it in the thread's own register (`float lrmean = lmean`) ✅ and use shuffle functions to pass values across threads in a warp.

**Warp Level Shuffling**

Similar to how we used strides in kernel 2 to reduce shared memory, we stride across warps in a block to find the sum of each warp in log(n) steps. `__shfl_down_sync` moves values down a warp by an offset. At the end of the loop, the warp's final sum is at its first index. `0xffffffff` sets the mask to all threads in the warp.

![Warp level shuffle diagram](/ln_images/image8.png)

```cpp
// global mean, warp level using shuffling
for(int offset = warp_size/2; offset > 0; offset /= 2){
    lrmean += __shfl_down_sync(0xffffffff, lrmean, offset);
}
// sum of each warp is now stored at index 0 of each warp
```

Next, we save the warp sums into shared memory to reduce them further across the block:

![Block level warp reduction](/ln_images/image9.png)

```cpp
// global mean, block level using shuffling
if (blockDim.x > warp_size){
    if(tidx % warp_size == 0){ // if first index of a warp
        smem[tidx/warp_size] = lrmean; // store sum of each warp into smem
    }
```

**Block Level Shuffling**

Only the first warp loads the warp sums from shared memory. Other warp values are zeroed out. We then perform one final warp reduction using `__shfl_down_sync` — after which `lrmean` holds the total sum of the entire row — and divide by `n` to get the global mean.

```cpp
if(tidx < warp_size){ // only first warp
    lrmean = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : 0.0f;
    for(int offset = warp_size / 2; offset > 0; offset /=2){
        lrmean += __shfl_down_sync(0xffffffff, lrmean, offset);
    }
    if(tidx==0){
        smem[0] = lrmean;
    }
}
float gmean = smem[0] / n; // global mean stored at first index of smem
```

We then repeat the same warp and block reduction pattern to compute global variance:

![Global variance computation](/ln_images/image10.png)

```cpp
float lrvar = lvar;

// warp level reduction
for(int offset = warp_size/2; offset > 0; offset /= 2){
    lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
}

// block level reduction
if (blockDim.x > warp_size){
    if(tidx % warp_size == 0){
        smem[tidx/warp_size] = lrvar; // store local warp variance in smem
    }
    __syncthreads();

    if(tidx < warp_size){
        lrvar = (tidx < (blockDim.x + warp_size - 1) / warp_size) ? smem[tidx] : 0.0f;
        for(int offset = warp_size / 2; offset > 0; offset /=2){
            lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
        }
        if(tidx == 0){
            smem[0] = lrvar;
        }
    }
} else {
    if(tidx == 0){
        smem[0] = lrvar;
    }
}
__syncthreads();

float gvar = (smem[0] / n) - (gmean * gmean); // global variance
```

This kernel outputs:

```
Shuffled Kernel Execution time: 0.07261 ms
```

We are now 10% faster than kernel 2! Let's go a little further.


## Kernel 4: Vectorized Loading

For our final kernel, let's optimize memory access further. So far, each thread loads one element at a time from global memory. Instead, what if we load 4 elements per thread? This is called **vectorized loading**.

![Vectorized loading diagram](/ln_images/image11.png)

We divide the total number of elements in a row by 4 to get the number of vectorized iterations: `vec_iters = n/4`. Here we have `1024/4 = 256 vec_iters`. Since we have 256 threads, each thread loads 4 elements simultaneously using the built-in `float4` struct.

```cpp
int vec_iters = n / 4;
for(int i = tidx; i < vec_iters; i += blockDim.x){
    float4 v = reinterpret_cast<float4 *>(row_in)[i];
    lmean += v.x + v.y + v.z + v.w; // div by n later for mean
    lvar += (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
}
```

`float4` is a built-in struct, so the four loaded values are accessed via `v.x, v.y, v.z, v.w`. In each iteration, one thread loads 4 elements and sums them to compute local mean and variance — much more efficient than scalar coalescing.

![Float4 thread load diagram](/ln_images/image12.png)

The local sums are stored in each thread's register. From there, warp-level reduction using `__shfl_down_sync` finds the global mean and variance — the same reduction code as kernel 3:

```cpp
// reducing sum across warps
for(int offset = WARP_SIZE / 2; offset > 0; offset /=2){
    lmean += __shfl_down_sync(0xffffffff, lmean, offset);
    lvar  += __shfl_down_sync(0xffffffff, lvar,  offset);
}

// global mean and variance
float gmean   = lmean / n;
float gvar    = (lvar / n) - (gmean * gmean);
float std_inv = rsqrtf(gvar + EPSILON);
```

Finally, we compute layer norm and write back to global memory in a vectorized manner:

```cpp
for(int i = tidx; i < vec_iters; i += blockDim.x){
    float4 v = reinterpret_cast<float4 *>(row_in)[i];
    v.x = (v.x - gmean) * std_inv;
    v.y = (v.y - gmean) * std_inv;
    v.z = (v.z - gmean) * std_inv;
    v.w = (v.w - gmean) * std_inv;
    reinterpret_cast<float4 *>(row_out)[i] = v; // write back to global mem
}

// handle remainder elements not covered by float4
for(int i = vec_iters * 4 + tidx; i < n; i += blockDim.x){
    row_out[i] = (row_in[i] - gmean) * std_inv;
}
```

This kernel outputs:

```
Vectorized Kernel Execution time: 0.05632 ms
```

Our vectorized approach is around **0.35 ms faster** than PyTorch — approximately 87% more efficient.

## Conclusion

In this worklog, we saw what layer norm is under the hood, benchmarked PyTorch's implementation, and iteratively wrote optimized kernels from scratch. We started with a naive single-thread-per-row implementation, then moved to memory coalescing with shared memory reductions, then to warp-level register shuffling, and finally to vectorized `float4` loads.

The point was not simply to beat PyTorch, but to understand GPU programming and the optimization techniques that make parallel code efficient. Each kernel improvement exposed a concrete bottleneck: global memory traffic, shared memory latency, and scalar load bandwidth — all of which are real constraints you will encounter in production kernels.

You can find the code implementations for all four kernels in my [GitHub](https://github.com/aryagxr/cuda).

![See you next time](/ln_images/byecat.png)
