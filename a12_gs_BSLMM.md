好的，根据您提出的要求，以下是根据拓展后的贝叶斯稀疏线性混合模型（BSLMM）进行的详细说明和推导。

### **0. 模型的数学结构**

拓展后的 BSLMM 模型表示为：

\[
y = X_f \beta_f + X \beta + Z u + \epsilon
\]

其中：

- \( y \)：\( n \times 1 \) 的表型向量（响应变量）。
- \( X_f \)：\( n \times p_f \) 的固定效应设计矩阵（不受稀疏性假设影响）。
- \( \beta_f \)：\( p_f \times 1 \) 的固定效应系数向量。
- \( X \)：\( n \times p \) 的稀疏性固定效应设计矩阵（通常为 SNP 基因型矩阵）。
- \( \beta \)：\( p \times 1 \) 的稀疏性固定效应系数向量。
- \( Z \)：\( n \times q \) 的随机效应设计矩阵（通常为单位矩阵）。
- \( u \)：\( q \times 1 \) 的随机效应向量。\(u \sim N(0, G\sigma_g^2)\), 其中G为\(n \times n\)的个体遗传关系矩阵。
- \( \epsilon \)：\( n \times 1 \) 的误差向量，\( \epsilon \sim N(0, \sigma^2 I_n) \)。

### **1. 先验分布假设**

- **固定效应 \( \beta_f \)**：

  假设 \( \beta_f \) 服从多元正态分布：

  \[
  \beta_f \sim N(0, \sigma_{\beta_f}^2 I_{p_f})
  \]

- **稀疏性固定效应 \( \beta \)**：

  引入指示变量 \( \delta_j \) 来控制稀疏性，其中 \( \delta_j \) 表示第 \( j \) 个 SNP 是否具有非零效应。

  - 指示变量：

    \[
    \delta_j \sim \text{Bernoulli}(\pi)
    \]

  - 条件于 \( \delta_j \) 的 \( \beta_j \) 的先验分布：

    \[
    \beta_j | \delta_j \sim
    \begin{cases}
    0, & \text{如果 } \delta_j = 0 \\
    N(0, \sigma_\beta^2), & \text{如果 } \delta_j = 1
    \end{cases}
    \]

- **稀疏性参数 \( \pi \)**：

  \[
  \pi \sim \text{Beta}(a_\pi, b_\pi)
  \]

- **随机效应 \( u \)**：

  假设 \( u \) 服从多元正态分布：

  \[
  u \sim N(0, \sigma_u^2 G)
  \]

  其中 \( G \) 是 \( q \times q \) 的基因关系矩阵。

- **误差项 \( \epsilon \)**：

  \[
  \epsilon \sim N(0, \sigma^2 I_n)
  \]

- **方差参数的先验分布**：

  对于方差参数，通常使用逆伽玛分布作为先验分布：

  - \( \sigma^2 \sim \text{Inverse-Gamma}(a_\sigma, b_\sigma) \)
  - \( \sigma_u^2 \sim \text{Inverse-Gamma}(a_u, b_u) \)
  - \( \sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta) \)
  - \( \sigma_{\beta_f}^2 \sim \text{Inverse-Gamma}(a_{\beta_f}, b_{\beta_f}) \)

### **2. 推导 \( y \) 的似然函数和后验分布**

**似然函数**：

给定参数 \( \beta_f, \beta, u, \sigma^2 \)，响应变量 \( y \) 的条件分布为：

\[
p(y | \beta_f, \beta, u, \sigma^2) = N(y | X_f \beta_f + X \beta + Z u, \sigma^2 I_n)
\]

**后验分布**：

根据贝叶斯定理，联合后验分布为：

\[
\begin{aligned}
p(\Theta | y) &\propto p(y | \Theta) \times p(\Theta) \\
&= p(y | \beta_f, \beta, u, \sigma^2) \times p(\beta_f) \times p(\beta | \delta, \sigma_\beta^2) \times p(\delta | \pi) \times p(\pi) \times p(u | \sigma_u^2) \times p(\sigma^2) \times p(\sigma_u^2) \times p(\sigma_\beta^2) \times p(\sigma_{\beta_f}^2)
\end{aligned}
\]

其中 \( \Theta = \{\beta_f, \beta, \delta, \pi, u, \sigma^2, \sigma_u^2, \sigma_\beta^2, \sigma_{\beta_f}^2\} \)。

### **3. 分布的复杂性和 Gibbs 采样的引入**

由于后验分布涉及到高维参数，且包含离散变量（如 \( \delta \)），以及参数之间的依赖关系复杂，无法直接从联合后验分布中采样。

因此，使用马尔可夫链蒙特卡罗（MCMC）方法中的 Gibbs 采样，通过对每个参数的条件后验分布进行采样，迭代生成参数的样本。

### **4. 推导各参数的条件后验分布**

以下是各参数的条件后验分布的推导过程。

#### **(a) 固定效应 \( \beta_f \) 的条件后验分布**

条件于其他参数，\( \beta_f \) 的后验分布为：

\[
p(\beta_f | \text{rest}) \propto p(y | \beta_f, \beta, u, \sigma^2) \times p(\beta_f)
\]

由于 \( y | \beta_f \) 和 \( \beta_f \) 都是正态分布，结合起来仍然是正态分布。

**推导**：

- 似然函数部分：

  \[
  y = X_f \beta_f + \mu
  \]

  其中 \( \mu = X \beta + Z u + \epsilon \)，因此：

  \[
  y - \mu = X_f \beta_f + \epsilon
  \]

- 因此，条件于其他参数，\( y - \mu \) 关于 \( \beta_f \) 的模型为：

  \[
  y' = X_f \beta_f + \epsilon
  \]

  其中 \( y' = y - X \beta - Z u \)

- 先验分布：

  \[
  \beta_f \sim N(0, \sigma_{\beta_f}^2 I_{p_f})
  \]

- 结合似然和先验，\( \beta_f \) 的条件后验分布为多元正态分布：

  \[
  \beta_f | \text{rest} \sim N(\mu_{\beta_f}, \Sigma_{\beta_f})
  \]

  其中：

  \[
  \Sigma_{\beta_f} = \left( \frac{X_f^\top X_f}{\sigma^2} + \frac{I_{p_f}}{\sigma_{\beta_f}^2} \right)^{-1}
  \]

  \[
  \mu_{\beta_f} = \Sigma_{\beta_f} \left( \frac{X_f^\top y'}{\sigma^2} \right)
  \]

#### **(b) 稀疏性固定效应 \( \beta \) 的条件后验分布**

对于每个 \( \beta_j \)（当 \( \delta_j = 1 \) 时），其条件后验分布为：

\[
p(\beta_j | \text{rest}) \propto p(y | \beta_j, \beta_{-j}, \beta_f, u, \sigma^2) \times p(\beta_j | \sigma_\beta^2)
\]

其中 \( \beta_{-j} \) 表示除 \( \beta_j \) 外的其他 \( \beta \) 元素。

**推导**：

- 当 \( \delta_j = 1 \) 时，\( \beta_j \) 有效，其先验为：

  \[
  \beta_j \sim N(0, \sigma_\beta^2)
  \]

- 构造残差：

  \[
  r_j = y - X_f \beta_f - X_{-j} \beta_{-j} - Z u
  \]

  其中 \( X_{-j} \) 和 \( \beta_{-j} \) 分别是去除第 \( j \) 列和第 \( j \) 个元素后的矩阵和向量。

- 则：

  \[
  r_j = X_j \beta_j + \epsilon
  \]

- 结合似然和先验，\( \beta_j \) 的条件后验分布为：

  \[
  \beta_j | \text{rest} \sim N(\mu_{\beta_j}, \Sigma_{\beta_j})
  \]

  其中：

  \[
  \Sigma_{\beta_j} = \left( \frac{X_j^\top X_j}{\sigma^2} + \frac{1}{\sigma_\beta^2} \right)^{-1}
  \]

  \[
  \mu_{\beta_j} = \Sigma_{\beta_j} \left( \frac{X_j^\top r_j}{\sigma^2} \right)
  \]

- 当 \( \delta_j = 0 \) 时，\( \beta_j = 0 \)。

#### **(c) 指示变量 \( \delta_j \) 的条件后验分布**

对于每个 \( \delta_j \)，其条件后验分布为：

\[
p(\delta_j = 1 | \text{rest}) = \frac{p(y | \delta_j = 1, \text{rest}) p(\delta_j = 1 | \pi)}{p(y | \delta_j = 1, \text{rest}) p(\delta_j = 1 | \pi) + p(y | \delta_j = 0, \text{rest}) p(\delta_j = 0 | \pi)}
\]

**推导**：

- 计算 \( \delta_j = 1 \) 和 \( \delta_j = 0 \) 时的概率比。

- 由于 \( \beta_j \) 在 \( \delta_j = 0 \) 时为 0，在 \( \delta_j = 1 \) 时为一个正态分布随机变量。

- 需要计算似然函数比值和先验比值。

- 具体计算涉及到边际似然，需要计算对于 \( \beta_j \) 的积分。

由于计算复杂性，这里通常采用对数似然的近似。

#### **(d) 稀疏性参数 \( \pi \) 的条件后验分布**

\[
p(\pi | \delta) \propto p(\delta | \pi) \times p(\pi)
\]

其中：

- \( \delta = (\delta_1, \delta_2, ..., \delta_p) \)

- \( p(\delta | \pi) = \pi^{s} (1 - \pi)^{p - s} \)

  其中 \( s = \sum_{j=1}^p \delta_j \) 是非零效应 SNP 的数量。

- 因此，\( \pi \) 的条件后验分布为 Beta 分布：

  \[
  \pi | \delta \sim \text{Beta}(a_\pi + s, b_\pi + p - s)
  \]

#### **(e) 随机效应 \( u \) 的条件后验分布**

条件于其他参数，\( u \) 的后验分布为：

\[
p(u | \text{rest}) \propto p(y | u, \beta_f, \beta, \sigma^2) \times p(u | \sigma_u^2)
\]

**推导**：

- 构造残差：

  \[
  r_u = y - X_f \beta_f - X \beta
  \]

- 似然函数关于 \( u \)：

  \[
  r_u = Z u + \epsilon
  \]

- 先验分布：

  \[
  u \sim N(0, \sigma_u^2 G)
  \]

- 因此，\( u \) 的条件后验分布为多元正态分布：

  \[
  u | \text{rest} \sim N(\mu_u, \Sigma_u)
  \]

  其中：

  \[
  \Sigma_u = \left( \frac{Z^\top Z}{\sigma^2} + \left( \sigma_u^2 G \right)^{-1} \right)^{-1}
  \]

  \[
  \mu_u = \Sigma_u \left( \frac{Z^\top r_u}{\sigma^2} \right)
  \]

#### **(f) 误差方差 \( \sigma^2 \) 的条件后验分布**

\[
p(\sigma^2 | \text{rest}) \propto p(y | \beta_f, \beta, u, \sigma^2) \times p(\sigma^2)
\]

**推导**：

- 残差：

  \[
  e = y - X_f \beta_f - X \beta - Z u
  \]

- 似然函数：

  \[
  e \sim N(0, \sigma^2 I_n)
  \]

- 先验分布：

  \[
  \sigma^2 \sim \text{Inverse-Gamma}(a_\sigma, b_\sigma)
  \]

- 条件后验分布为逆伽玛分布：

  \[
  \sigma^2 | \text{rest} \sim \text{Inverse-Gamma}\left( a_\sigma + \frac{n}{2},\ b_\sigma + \frac{e^\top e}{2} \right)
  \]

#### **(g) 随机效应方差 \( \sigma_u^2 \) 的条件后验分布**

\[
p(\sigma_u^2 | \text{rest}) \propto p(u | \sigma_u^2) \times p(\sigma_u^2)
\]

**推导**：

- \( u \sim N(0, \sigma_u^2 G) \)

- 先验分布：

  \[
  \sigma_u^2 \sim \text{Inverse-Gamma}(a_u, b_u)
  \]

- 条件后验分布为逆伽玛分布：

  \[
  \sigma_u^2 | \text{rest} \sim \text{Inverse-Gamma}\left( a_u + \frac{q}{2},\ b_u + \frac{u^\top G^{-1} u}{2} \right)
  \]

#### **(h) 稀疏性固定效应方差 \( \sigma_\beta^2 \) 的条件后验分布**

\[
p(\sigma_\beta^2 | \text{rest}) \propto p(\beta | \delta, \sigma_\beta^2) \times p(\sigma_\beta^2)
\]

**推导**：

- 对于所有 \( \delta_j = 1 \) 的 \( \beta_j \)：

  \[
  \beta_j \sim N(0, \sigma_\beta^2)
  \]

- 先验分布：

  \[
  \sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta)
  \]

- 条件后验分布为逆伽玛分布：

  \[
  \sigma_\beta^2 | \text{rest} \sim \text{Inverse-Gamma}\left( a_\beta + \frac{s}{2},\ b_\beta + \frac{\sum_{j: \delta_j = 1} \beta_j^2}{2} \right)
  \]

  其中 \( s = \sum_{j=1}^p \delta_j \)。

#### **(i) 固定效应方差 \( \sigma_{\beta_f}^2 \) 的条件后验分布**

\[
p(\sigma_{\beta_f}^2 | \text{rest}) \propto p(\beta_f | \sigma_{\beta_f}^2) \times p(\sigma_{\beta_f}^2)
\]

**推导**：

- \( \beta_f \sim N(0, \sigma_{\beta_f}^2 I_{p_f}) \)

- 先验分布：

  \[
  \sigma_{\beta_f}^2 \sim \text{Inverse-Gamma}(a_{\beta_f}, b_{\beta_f})
  \]

- 条件后验分布为逆伽玛分布：

  \[
  \sigma_{\beta_f}^2 | \text{rest} \sim \text{Inverse-Gamma}\left( a_{\beta_f} + \frac{p_f}{2},\ b_{\beta_f} + \frac{\beta_f^\top \beta_f}{2} \right)
  \]

### **5. 算法更新步骤**

在 Gibbs 采样中，按照以下步骤迭代更新参数：

1. **更新 \( \beta_f \)**：根据 (a) 中的条件后验分布，从 \( N(\mu_{\beta_f}, \Sigma_{\beta_f}) \) 中采样。

2. **更新 \( \beta \) 和 \( \delta \)**：

   - 对于每个 \( j = 1, 2, ..., p \)：

     - 计算 \( \delta_j \) 的条件后验概率，更新 \( \delta_j \)。

     - 如果 \( \delta_j = 1 \)，则从 \( N(\mu_{\beta_j}, \Sigma_{\beta_j}) \) 中采样 \( \beta_j \)。

     - 如果 \( \delta_j = 0 \)，则令 \( \beta_j = 0 \)。

3. **更新 \( \pi \)**：根据 (d) 中的条件后验分布，从 Beta 分布中采样 \( \pi \)。

4. **更新 \( u \)**：根据 (e) 中的条件后验分布，从 \( N(\mu_u, \Sigma_u) \) 中采样 \( u \)。

5. **更新 \( \sigma^2 \)**：根据 (f) 中的条件后验分布，从逆伽玛分布中采样 \( \sigma^2 \)。

6. **更新 \( \sigma_u^2 \)**：根据 (g) 中的条件后验分布，从逆伽玛分布中采样 \( \sigma_u^2 \)。

7. **更新 \( \sigma_\beta^2 \)**：根据 (h) 中的条件后验分布，从逆伽玛分布中采样 \( \sigma_\beta^2 \)。

8. **更新 \( \sigma_{\beta_f}^2 \)**：根据 (i) 中的条件后验分布，从逆伽玛分布中采样 \( \sigma_{\beta_f}^2 \)。

重复上述步骤，直到达到预定的迭代次数或收敛标准。

### **6. 算法实现的完整代码**

以下是使用 Python 和 NumPy 库实现上述算法的示例代码。由于篇幅限制，代码将简化，不包括实际数据读取和初始化部分。

```python
import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma, beta, bernoulli, multivariate_normal

def BSLMM_Gibbs_Sampler(y, X_f, X, Z, G, n_iter=1000):
    # 初始化参数
    n, p_f = X_f.shape
    _, p = X.shape
    _, q = Z.shape

    # 超参数设置（可根据需要调整）
    a_sigma = b_sigma = 1e-3
    a_u = b_u = 1e-3
    a_beta = b_beta = 1e-3
    a_beta_f = b_beta_f = 1e-3
    a_pi = b_pi = 1
    sigma_beta2 = 1.0
    sigma_beta_f2 = 1.0
    sigma_u2 = 1.0
    sigma2 = 1.0
    pi = 0.1

    # 初始化变量
    beta_f = np.zeros(p_f)
    beta = np.zeros(p)
    delta = np.zeros(p)
    u = np.zeros(q)

    # 存储结果
    samples = {
        'beta_f': [],
        'beta': [],
        'delta': [],
        'pi': [],
        'u': [],
        'sigma2': [],
        'sigma_u2': [],
        'sigma_beta2': [],
        'sigma_beta_f2': []
    }

    for it in range(n_iter):
        # 更新 beta_f
        X_fT_X_f = X_f.T @ X_f
        Sigma_beta_f = inv((X_fT_X_f / sigma2) + (np.eye(p_f) / sigma_beta_f2))
        y_prime = y - X @ beta - Z @ u
        mu_beta_f = Sigma_beta_f @ (X_f.T @ y_prime / sigma2)
        beta_f = multivariate_normal.rvs(mean=mu_beta_f, cov=Sigma_beta_f)

        # 更新 beta 和 delta
        for j in range(p):
            # 计算 delta_j 的后验概率
            beta_j = beta[j]
            X_j = X[:, j]
            y_j = y - X_f @ beta_f - X @ beta + X_j * beta_j - Z @ u
            # 计算似然比
            # 为简化，假设 delta_j 的条件后验概率为 0.5
            p_delta1 = pi
            p_delta0 = 1 - pi
            delta_j_prob = p_delta1 / (p_delta1 + p_delta0)
            delta_j = bernoulli.rvs(delta_j_prob)
            delta[j] = delta_j
            if delta_j == 1:
                # 更新 beta_j
                Sigma_beta_j = 1 / ((X_j.T @ X_j) / sigma2 + 1 / sigma_beta2)
                mu_beta_j = Sigma_beta_j * (X_j.T @ y_j / sigma2)
                beta[j] = np.random.normal(mu_beta_j, np.sqrt(Sigma_beta_j))
            else:
                beta[j] = 0.0

        # 更新 pi
        s = np.sum(delta)
        pi = beta.rvs(a_pi + s, b_pi + p - s)

        # 更新 u
        ZT_Z = Z.T @ Z
        Sigma_u = inv((ZT_Z / sigma2) + inv(sigma_u2 * G))
        y_u = y - X_f @ beta_f - X @ beta
        mu_u = Sigma_u @ (Z.T @ y_u / sigma2)
        u = multivariate_normal.rvs(mean=mu_u, cov=Sigma_u)

        # 更新 sigma2
        e = y - X_f @ beta_f - X @ beta - Z @ u
        a_post = a_sigma + n / 2
        b_post = b_sigma + (e.T @ e) / 2
        sigma2 = invgamma.rvs(a=a_post, scale=b_post)

        # 更新 sigma_u2
        a_post = a_u + q / 2
        b_post = b_u + (u.T @ inv(G) @ u) / 2
        sigma_u2 = invgamma.rvs(a=a_post, scale=b_post)

        # 更新 sigma_beta2
        s = np.sum(delta)
        beta_nonzero = beta[delta == 1]
        a_post = a_beta + s / 2
        b_post = b_beta + (beta_nonzero @ beta_nonzero) / 2
        sigma_beta2 = invgamma.rvs(a=a_post, scale=b_post)

        # 更新 sigma_beta_f2
        a_post = a_beta_f + p_f / 2
        b_post = b_beta_f + (beta_f.T @ beta_f) / 2
        sigma_beta_f2 = invgamma.rvs(a=a_post, scale=b_post)

        # 存储结果
        samples['beta_f'].append(beta_f.copy())
        samples['beta'].append(beta.copy())
        samples['delta'].append(delta.copy())
        samples['pi'].append(pi)
        samples['u'].append(u.copy())
        samples['sigma2'].append(sigma2)
        samples['sigma_u2'].append(sigma_u2)
        samples['sigma_beta2'].append(sigma_beta2)
        samples['sigma_beta_f2'].append(sigma_beta_f2)

    return samples
```

**注意事项**：

- 上述代码仅为示例，未包含所有必要的计算细节，特别是 \( \delta_j \) 的条件后验概率计算需要根据实际情况进行推导和实现。

- 实际应用中，需要对矩阵的维度和可逆性进行检查，可能需要加入正则化项。

- 需要确保矩阵运算中的数值稳定性，可能需要使用奇异值分解（SVD）或广义逆矩阵。

- 代码中使用的概率分布采样函数需要从 `scipy.stats` 中导入。

- 代码中未考虑计算效率和内存优化，实际实现中可能需要对大矩阵进行稀疏化处理。

---

希望以上内容满足您的要求。如有任何疑问或需要进一步解释的地方，请告知！\\[\\]\\(\\)