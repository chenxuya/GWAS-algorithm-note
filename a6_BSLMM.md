# BSLMM模型简介
**贝叶斯稀疏线性混合模型（Bayesian Sparse Linear Mixed Model, BSLMM）** 是一个结合了**稀疏性**（sparse effect）和**随机效应**的模型，在GWAS中用于捕捉表型与基因型之间的复杂关系。BSLMM通过贝叶斯框架推断基因效应的稀疏性以及个体间的遗传背景差异，兼具稀疏回归和线性混合模型的优点。

### 1. 模型假设

BSLMM模型公式为：
$$
y = X \beta + Z u + \epsilon
$$
- **$ y $**：$ n \times 1 $ 的表型向量。
- **$ X $**：$ n \times p $ 的基因型矩阵，表示 $ n $ 个个体的 $ p $ 个SNP。
- **$ \beta $**：$ p \times 1 $ 的SNP效应大小向量。$ \beta $ 是稀疏的，仅有少部分非零效应。
- **$ Z u $**：随机效应，$ Z $ 是 $ n \times n $ 的设计矩阵，通常为亲缘关系矩阵，$ u $ 是个体的随机遗传效应，假设 $ u \sim N(0, \sigma^2_g G) $。
- **$ \epsilon $**：残差，假设 $ \epsilon \sim N(0, \sigma^2_e I) $。

BSLMM的核心在于对 $ \beta $ 的稀疏性建模，同时引入 $ u $ 用来捕捉个体间的随机效应。

### 2. 先验设定

对参数 $ \beta $ 和 $ u $ 使用贝叶斯先验。

- **SNP效应 $ \beta $ 的稀疏性**：对于 $ \beta $，假设它是稀疏的，即只有一部分SNP位点对表型有显著影响。对每个 $ \beta_j $，其先验分布为：
  $$
  \beta_j \sim \pi \cdot N(0, \sigma^2_{\beta}) + (1 - \pi) \cdot \delta_0
  $$
  这里：
  - $ \pi $ 是SNP效应为非零的概率（稀疏先验）。
  - $ N(0, \sigma^2_{\beta}) $ 是正态分布，表示非零SNP效应的大小。
  - $ \delta_0 $ 是Dirac delta函数，表示 $ \beta_j = 0 $ 时的零效应。

- **随机效应 $ u $**：对于个体间的随机效应 $ u $，假设其先验分布为：
  $$
  u \sim N(0, \sigma^2_g G)
  $$
  其中 $ G $ 是亲缘关系矩阵，用于捕捉个体间的遗传背景。

- **误差项 $ \epsilon $**：假设误差服从正态分布：
  $$
  \epsilon \sim N(0, \sigma^2_e I)
  $$

### 3. 似然函数

基于模型 $ y = X \beta + Z u + \epsilon $，表型 $ y $ 的似然函数为：

$$
p(y | X, \beta, u) = N(X\beta + Z u, \sigma^2_e I)
$$

该似然函数表示，在给定 $ X \beta $ 和 $ Z u $ 的情况下，表型数据 $ y $ 的概率分布是正态分布。

#### 似然函数的表达式展开：
$$
p(y | X, \beta, u) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) \right)
$$

### 4. 后验分布推导

根据贝叶斯定理，后验分布 $ p(\beta, u | y, X) $ 是似然函数和先验分布的乘积：

$$
p(\beta, u | y, X) \propto p(y | X, \beta, u) \cdot p(\beta) \cdot p(u)
$$

#### 展开后验分布：
$$
p(\beta, u | y, X) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) \right) \cdot \prod_{j=1}^p \left[ \pi \cdot N(0, \sigma^2_{\beta}) + (1 - \pi) \cdot \delta_0 \right] \cdot N(0, \sigma^2_g G)
$$

#### 拆分为各部分解释：

1. **似然部分**：描述了模型对表型数据的拟合程度。
2. **先验部分（SNP效应 $ \beta $ 的先验）**：反映了我们对SNP效应稀疏性的先验假设。
3. **先验部分（随机效应 $ u $ 的先验）**：捕捉个体间的随机遗传效应。

由于这个后验分布没有解析解，通常使用**Markov Chain Monte Carlo (MCMC)** 方法进行采样，以近似推断后验分布的样本。

### 5. 参数估计过程

在BSLMM中，主要的参数包括：
- SNP效应 $ \beta $。
- 随机效应 $ u $。
- 方差参数 $ \sigma^2_{\beta} $、$ \sigma^2_g $ 和 $ \sigma^2_e $。

通过**MCMC采样**过程，依次对每个参数进行更新，具体过程如下：

#### 1. **更新SNP效应 $ \beta $**：
- 使用Gibbs采样或Metropolis-Hastings采样来更新每个 $ \beta_j $ 的值。对于每个 $ \beta_j $，其条件分布基于当前的 $ \beta_{-j} $ 和其他参数进行计算。
- 由于 $ \beta_j $ 服从稀疏先验，因此有：
  $$
  p(\beta_j | y, \beta_{-j}, u, \sigma^2_{\beta}, \sigma^2_e) \propto p(y | \beta_j, \beta_{-j}, u) \cdot p(\beta_j)
  $$
  使用类似于我们在稀疏贝叶斯回归中的方法来进行更新。

#### 2. **更新随机效应 $ u $**：
- 根据给定的 $ \beta $ 和 $ \sigma^2_g $，更新 $ u $ 的条件分布。由于 $ u $ 服从正态分布，条件分布的计算相对简单，可以直接使用Gibbs采样。
- 其条件分布为：
  $$
  u | y, \beta, \sigma^2_g, \sigma^2_e \sim N( (Z^T Z + \frac{\sigma^2_e}{\sigma^2_g} G^{-1})^{-1} Z^T (y - X \beta), \sigma^2_g)
  $$

#### 3. **更新方差参数**：
- 对 $ \sigma^2_{\beta} $、$ \sigma^2_g $ 和 $ \sigma^2_e $ 的更新通常使用**逆Gamma分布**或**Gamma分布**作为先验，基于残差来更新。

# 参数估计

## 一. SNP效应 $ \beta $ 的参数估计
好的，让我们详细讨论 **SNP效应 $ \beta $** 的估计过程。BSLMM 的一个重要特点是，它使用贝叶斯方法来估计 SNP 的稀疏效应，这意味着我们需要从后验分布中推断每个 $ \beta_j $ 的值。由于后验分布通常没有解析解，MCMC 采样方法被用来依次更新每个参数。

### 1. 问题回顾

在 BSLMM 模型中，线性模型表示为：
$$
y = X\beta + Zu + \epsilon
$$
其中：
- $ y $：$ n \times 1 $ 的表型向量。
- $ X $：$ n \times p $ 的基因型矩阵。
- $ \beta $：$ p \times 1 $ 的SNP效应大小向量。
- $ Z u $：随机效应，$ u \sim N(0, \sigma^2_g G) $。
- $ \epsilon $：误差，假设 $ \epsilon \sim N(0, \sigma^2_e I) $。

我们的目标是对每个 $ \beta_j $ 进行更新。假设先验 $ \beta_j $ 服从稀疏先验分布：
$$
\beta_j \sim \pi \cdot N(0, \sigma^2_{\beta}) + (1 - \pi) \cdot \delta_0
$$
其中：
- $ \pi $ 是SNP效应为非零的概率。
- $ N(0, \sigma^2_{\beta}) $ 是正态分布，表示非零效应的大小。
- $ \delta_0 $ 是Dirac delta函数，表示效应为零。

### 2. 条件分布的推导

为了使用 MCMC 更新 $ \beta_j $，我们需要推导 $ \beta_j $ 的条件分布，即 $ p(\beta_j | y, \beta_{-j}, u, \sigma^2_{\beta}, \sigma^2_e) $。我们可以通过贝叶斯定理来结合先验分布和似然函数推导这个条件分布。

首先，给定当前的其他参数 $ \beta_{-j} $ 和随机效应 $ u $，模型可以简化为：
$$
r_j = y - X_{-j} \beta_{-j} - Z u
$$
其中：
- $ X_{-j} $ 表示去除第 $ j $ 个 SNP 的基因型矩阵。
- $ \beta_{-j} $ 表示去除第 $ j $ 个 SNP 的效应。

这个残差 $ r_j $ 是去除了其他SNP和随机效应后，依赖于第 $ j $ 个 SNP 效应 $ \beta_j $ 的部分。

因此，模型可以简化为：
$$
r_j = X_j \beta_j + \epsilon
$$
其中 $ X_j $ 是第 $ j $ 列的基因型矩阵。

### 3. 似然函数

在这个简化模型下，给定 $ r_j $，我们可以写出第 $ j $ 个效应 $ \beta_j $ 的似然函数为：
$$
p(r_j | X_j, \beta_j, \sigma^2_e) = N(X_j \beta_j, \sigma^2_e I)
$$
展开为指数形式：
$$
p(r_j | X_j, \beta_j, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (r_j - X_j \beta_j)^T (r_j - X_j \beta_j) \right)
$$

### 4. 先验分布

根据模型假设，$ \beta_j $ 的先验分布为：
$$
p(\beta_j) = \pi \cdot N(0, \sigma^2_{\beta}) + (1 - \pi) \cdot \delta_0
$$
这表示 $ \beta_j $ 以概率 $ \pi $ 服从正态分布 $ N(0, \sigma^2_{\beta}) $，以概率 $ 1 - \pi $ 为零。

### 5. 后验分布的计算

根据贝叶斯定理，后验分布 $ p(\beta_j | r_j, X_j, \sigma^2_e, \sigma^2_{\beta}) $ 是先验分布和似然函数的乘积。我们分两种情况讨论：

#### 情况 1：非零效应 ($ \beta_j \neq 0 $)

在这种情况下，后验分布为正态分布。结合先验和似然函数，后验分布为：
$$
p(\beta_j | r_j, X_j, \sigma^2_e, \sigma^2_{\beta}) \propto \exp\left( -\frac{1}{2} \left( \frac{X_j^T X_j}{\sigma^2_e} + \frac{1}{\sigma^2_{\beta}} \right) \beta_j^2 + \frac{r_j^T X_j}{\sigma^2_e} \beta_j \right)
$$
这是一元正态分布的指数形式。

通过与正态分布的标准形式对比，可以得出：
- **方差**：
  $$
  \text{var}(\beta_j) = \frac{1}{\frac{X_j^T X_j}{\sigma^2_e} + \frac{1}{\sigma^2_{\beta}}}
  $$
- **均值**：
  $$
  \text{mean}(\beta_j) = \text{var}(\beta_j) \cdot \frac{X_j^T r_j}{\sigma^2_e}
  $$

#### 情况 2：零效应 ($ \beta_j = 0 $)

在这种情况下，$ \beta_j $ 的值为零，后验分布是 Dirac delta 函数：
$$
p(\beta_j = 0) = 1 - \pi
$$

### 6. MCMC更新过程

在每次迭代中，对于每个 $ \beta_j $，根据当前的其他参数（即 $ \beta_{-j} $ 和 $ u $）进行条件更新。过程如下：

1. **采样 $ \beta_j $**：
   - 以概率 $ \pi $，根据后验分布的均值和方差采样非零效应：
     $$
     \beta_j \sim N\left( \text{mean}(\beta_j), \text{var}(\beta_j) \right)
     $$
   - 以概率 $ 1 - \pi $，设置 $ \beta_j = 0 $。
   
2. **更新 $ \beta_j $ 后**，使用新的 $ \beta_j $ 更新模型中的残差 $ r_j $，并进入下一个SNP效应的采样。

### 7. 总结

SNP效应 $ \beta_j $ 的更新是基于条件分布的贝叶斯推断过程。我们推导了 $ \beta_j $ 的条件后验分布，并使用 MCMC 方法逐个更新 $ \beta_j $。该过程分为两种情况：非零效应时根据正态分布采样，零效应时直接设为0。

## 二. 随机效应u的估计
接下来我们详细讨论 BSLMM 中 **随机效应 $ u $** 的估计过程。与SNP效应 $ \beta $ 类似，我们需要从随机效应 $ u $ 的条件分布中进行采样更新。随机效应的目的是捕捉个体间的相关性，通常通过亲缘关系矩阵 $ G $ 来表征。

### 1. 问题回顾

BSLMM 的模型公式为：
$$
y = X\beta + Z u + \epsilon
$$
其中：
- $ y $：$ n \times 1 $ 的表型向量。
- $ X $：$ n \times p $ 的基因型矩阵。
- $ \beta $：$ p \times 1 $ 的SNP效应大小向量。
- $ Z u $：随机效应部分，$ Z $ 通常是 $ n \times n $ 的个体设计矩阵（在某些情况下可以取为单位矩阵），$ u $ 是个体的随机效应。
- $ u $：$ n \times 1 $ 的个体随机效应向量，假设 $ u \sim N(0, \sigma^2_g G) $，其中 $ G $ 是 $ n \times n $ 的亲缘关系矩阵。
- $ \epsilon $：误差，假设 $ \epsilon \sim N(0, \sigma^2_e I) $。

我们的目标是推导 $ u $ 的条件分布 $ p(u | y, X, \beta, \sigma^2_g, \sigma^2_e) $，并通过 MCMC 更新 $ u $ 的值。

### 2. 先验分布

随机效应 $ u $ 表示个体间的遗传背景效应，服从如下多元正态分布：
$$
u \sim N(0, \sigma^2_g G)
$$
- $ \sigma^2_g $ 是遗传效应的方差参数。
- $ G $ 是亲缘关系矩阵，用于描述个体间的遗传相似性。

### 3. 似然函数

给定随机效应 $ u $ 和 SNP 效应 $ \beta $，表型 $ y $ 的条件似然函数为：
$$
p(y | X, \beta, u, \sigma^2_e) = N(X\beta + Z u, \sigma^2_e I)
$$
展开为指数形式：
$$
p(y | X, \beta, u, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) \right)
$$

### 4. 后验分布推导

根据贝叶斯定理，随机效应 $ u $ 的条件后验分布为：
$$
p(u | y, X, \beta, \sigma^2_g, \sigma^2_e) \propto p(y | X, \beta, u, \sigma^2_e) \cdot p(u | \sigma^2_g)
$$
解释： 
$$
p(u | y, X, \beta, \sigma^2_g, \sigma^2_e) =\frac{ p(y | X, \beta, u, \sigma^2_e) \cdot p(u | \sigma^2_g)}{p(y|X)}
$$
根据条件概率的链式法则：
$$
p(u | y, X, \beta, \sigma^2_g, \sigma^2_e) \cdot p(y | X) = p(y ,u| X, \beta, \sigma^2_g, \sigma^2_e)
$$
$$
p(y | X, \beta, u, \sigma^2_e) \cdot p(u | \sigma^2_g) = p(y, u | X, \beta, \sigma^2_g, \sigma^2_e)
$$


我们已经有了似然函数 $ p(y | X, \beta, u, \sigma^2_e) $ 和先验分布 $ p(u | \sigma^2_g) $，因此我们可以推导 $ u $ 的后验分布。

#### 1. 展开似然函数：
$$
p(y | X, \beta, u, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) \right)
$$

#### 2. 展开先验分布：
$$
p(u | \sigma^2_g) \propto \exp\left( -\frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$

#### 3. 联合分布：
将似然函数和先验分布相乘，得到 $ u $ 的后验分布（比例项）：
$$
p(u | y, X, \beta, \sigma^2_g, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) - \frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$
我们需要化简这个公式，使其看起来像一个多元正态分布的形式。

### 5. 化简后验分布

为了得到 $ u $ 的后验分布的解析形式，我们将残差 $ r = y - X\beta $ 代入公式，使后验分布变得更清晰：

$$
p(u | r, Z, \sigma^2_g, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (r - Z u)^T (r - Z u) - \frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$

展开平方项后：
$$
p(u | r, Z, \sigma^2_g, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (r^T r - 2 r^T Z u + u^T Z^T Z u) - \frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$

现在，我们将所有关于 $ u $ 的二次型项收集在一起：
$$
p(u | r, Z, \sigma^2_g, \sigma^2_e) \propto \exp\left( -\frac{1}{2} \left( u^T \left( \frac{Z^T Z}{\sigma^2_e} + \frac{G^{-1}}{\sigma^2_g} \right) u - 2 u^T \frac{Z^T r}{\sigma^2_e} \right) \right)
$$

这是一个关于 $ u $ 的多元正态分布的指数形式，因此 $ u $ 的条件后验分布是一个多元正态分布：
$$
u | r, Z, \sigma^2_g, \sigma^2_e \sim N\left( \mu_u, \Sigma_u \right)
$$
其中：
- **均值 $ \mu_u $**：
  $$
  \mu_u = \Sigma_u \cdot \frac{Z^T r}{\sigma^2_e}
  $$
- **协方差矩阵 $ \Sigma_u $**：
  $$
  \Sigma_u = \left( \frac{Z^T Z}{\sigma^2_e} + \frac{G^{-1}}{\sigma^2_g} \right)^{-1}
  $$

### 6. MCMC更新过程

在MCMC采样过程中，$ u $ 的更新步骤为：

1. **计算均值 $ \mu_u $**：
   $$
   \mu_u = \Sigma_u \cdot \frac{Z^T r}{\sigma^2_e}
   $$
   其中 $ r = y - X \beta $ 是当前的残差。

2. **计算协方差矩阵 $ \Sigma_u $**：
   $$
   \Sigma_u = \left( \frac{Z^T Z}{\sigma^2_e} + \frac{G^{-1}}{\sigma^2_g} \right)^{-1}
   $$

3. **根据条件分布采样 $ u $**：
   $ u $ 的条件分布为多元正态分布 $ N(\mu_u, \Sigma_u) $，我们可以根据这个分布采样 $ u $ 的新值。

### 7. 总结

- $ u $ 的条件分布是一个多元正态分布，其均值 $ \mu_u $ 取决于当前的残差 $ r = y - X \beta $ 和设计矩阵 $ Z $。
- 协方差矩阵 $ \Sigma_u $ 是由 $ Z^T Z $ 和 $ G^{-1} $ 共同决定的。
- 在 MCMC 更新过程中，我们通过 Gibbs 采样或 Metropolis-Hastings 采样来从条件分布中更新 $ u $ 的值。

这个更新过程通过捕捉个体间的遗传背景效应来调整模型，使得它更好地解释表型与基因型之间的关系。

## 三. $\sigma^2_g$的估计
接下来，我们详细讨论 **遗传方差 $ \sigma^2_g $** 的估计过程。**$ \sigma^2_g $** 是 BSLMM 模型中用于表示个体间的遗传效应的方差参数。我们将通过贝叶斯推断中的后验分布推导，来确定 $ \sigma^2_g $ 的估计方式。

### 1. 模型回顾

BSLMM 模型的随机效应部分可以写为：
$$
u \sim N(0, \sigma^2_g G)
$$
其中：
- **$ u $** 是个体间的随机效应，表示个体的遗传背景。
- **$ \sigma^2_g $** 是遗传效应的方差。
- **$ G $** 是亲缘关系矩阵，用于捕捉个体之间的遗传相似性。

模型的完整形式为：
$$
y = X\beta + Z u + \epsilon
$$
其中 $ \epsilon \sim N(0, \sigma^2_e I) $ 是残差项。

### 2. 先验分布设定

我们假设 $ \sigma^2_g $ 服从某种先验分布，通常选择逆Gamma分布或Gamma分布作为 $ \sigma^2_g $ 的先验分布。假设 $ \sigma^2_g $ 的先验分布为**逆Gamma分布**：
$$
p(\sigma^2_g) \sim \text{Inverse-Gamma}(\alpha_g, \beta_g)
$$
其中：
- $ \alpha_g $ 和 $ \beta_g $ 是先验分布的超参数，通常通过经验知识或之前的研究设定。

### 3. 后验分布推导

我们的目标是推导 $ \sigma^2_g $ 的条件后验分布 $ p(\sigma^2_g | y, X, \beta, u, \sigma^2_e) $，并通过 MCMC 方法进行更新。根据贝叶斯定理，后验分布与似然函数和先验分布的乘积成正比：
$$
p(\sigma^2_g | y, X, \beta, u, \sigma^2_e) \propto p(y | X, \beta, u, \sigma^2_g, \sigma^2_e) \cdot p(\sigma^2_g)
$$

其中：
- **$ p(y | X, \beta, u, \sigma^2_g, \sigma^2_e) $** 是似然函数。
- **$ p(\sigma^2_g) $** 是 $ \sigma^2_g $ 的先验分布。

#### 1. 似然函数的推导

根据模型 $ y = X \beta + Z u + \epsilon $，给定 $ u $ 的情况下，表型 $ y $ 的似然函数为：
$$
p(y | X, \beta, u, \sigma^2_e) = N(X\beta + Z u, \sigma^2_e I)
$$
展开为指数形式：
$$
p(y | X, \beta, u, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X\beta - Z u)^T (y - X\beta - Z u) \right)
$$

#### 2. $ \sigma^2_g $ 的先验分布

我们假设 $ \sigma^2_g $ 的先验分布为逆Gamma分布：
$$
p(\sigma^2_g) \propto (\sigma^2_g)^{-\alpha_g - 1} \exp\left( -\frac{\beta_g}{\sigma^2_g} \right)
$$
其中 $ \alpha_g $ 和 $ \beta_g $ 是超参数。

#### 3. 后验分布的推导

为了推导 $ \sigma^2_g $ 的后验分布，我们还需要考虑 $ u $ 的先验分布。由于 $ u \sim N(0, \sigma^2_g G) $，即 $ u $ 的方差与 $ \sigma^2_g $ 有关，$ u $ 的似然可以写为：
$$
p(u | \sigma^2_g) = N(0, \sigma^2_g G)
$$
展开为指数形式：
$$
p(u | \sigma^2_g) \propto \exp\left( -\frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$

现在，我们可以写出 $ \sigma^2_g $ 的后验分布：

$$
p(\sigma^2_g | y, X, \beta, u, \sigma^2_e) \propto p(y | X, \beta, u, \sigma^2_e) \cdot p(u | \sigma^2_g) \cdot p(\sigma^2_g)
$$

将似然函数、$ u $ 的先验和 $ \sigma^2_g $ 的先验相乘，得到：
$$
p(\sigma^2_g | y, X, \beta, u, \sigma^2_e) \propto (\sigma^2_g)^{-\alpha_g - 1} \exp\left( -\frac{\beta_g}{\sigma^2_g} \right) \exp\left( -\frac{1}{2\sigma^2_g} u^T G^{-1} u \right)
$$

### 4. 化简后验分布

将所有关于 $ \sigma^2_g $ 的项结合起来，可以将后验分布重新组织为逆Gamma分布的形式：
$$
p(\sigma^2_g | u) \propto (\sigma^2_g)^{-\left(\alpha_g + \frac{n}{2}\right) - 1} \exp\left( -\frac{1}{\sigma^2_g} \left( \beta_g + \frac{1}{2} u^T G^{-1} u \right) \right)
$$
因此，$ \sigma^2_g $ 的条件后验分布仍然是一个**逆Gamma分布**：
$$
\sigma^2_g | u \sim \text{Inverse-Gamma}\left( \alpha_g + \frac{n}{2}, \beta_g + \frac{1}{2} u^T G^{-1} u \right)
$$
其中：
- **形状参数**：$ \alpha_g + \frac{n}{2} $。
- **尺度参数**：$ \beta_g + \frac{1}{2} u^T G^{-1} u $。

### 5. MCMC 更新过程

在 MCMC 采样过程中，$ \sigma^2_g $ 的更新步骤如下：

1. **计算当前的 $ u $**：基于当前的遗传效应 $ u $ 计算残差 $ u^T G^{-1} u $。
2. **根据条件后验分布采样 $ \sigma^2_g $**：使用逆Gamma分布 $ \text{Inverse-Gamma}\left( \alpha_g + \frac{n}{2}, \beta_g + \frac{1}{2} u^T G^{-1} u \right) $ 采样新的 $ \sigma^2_g $。

### 6. 总结

$ \sigma^2_g $ 的估计过程依赖于 $ u $ 的当前值和遗传效应的方差假设。通过推导后验分布，我们得出 $ \sigma^2_g $ 的条件后验分布是一个逆Gamma分布。在 MCMC 采样过程中，我们通过 Gibbs 采样从这个逆Gamma分布中更新 $ \sigma^2_g $ 的值。

## 四. $\sigma^2_e$ 的估计
接下来我们讨论 **误差方差 $ \sigma^2_e $** 的估计过程。**$ \sigma^2_e $** 是表型模型中的残差项的方差，用来表示未解释部分的变异。我们将通过贝叶斯推断推导其后验分布，并解释如何通过 MCMC 进行更新。

### 1. 模型回顾

BSLMM 模型公式为：
$$
y = X\beta + Z u + \epsilon
$$
其中：
- $ y $：$ n \times 1 $ 的表型向量。
- $ X $：$ n \times p $ 的基因型矩阵。
- $ \beta $：$ p \times 1 $ 的SNP效应大小向量。
- $ Z u $：随机效应部分，$ u \sim N(0, \sigma^2_g G) $，其中 $ G $ 是亲缘关系矩阵。
- $ \epsilon \sim N(0, \sigma^2_e I) $：残差项，表示模型中未解释的变异，服从正态分布 $ N(0, \sigma^2_e I) $。

我们的目标是对误差方差 $ \sigma^2_e $ 进行估计。$ \sigma^2_e $ 反映了模型中的噪声水平，它对应的是个体未解释的表型变异。

### 2. 先验分布设定

通常，我们为 $ \sigma^2_e $ 设定一个**逆Gamma分布**作为先验分布：
$$
p(\sigma^2_e) \sim \text{Inverse-Gamma}(\alpha_e, \beta_e)
$$
其中：
- $ \alpha_e $ 和 $ \beta_e $ 是先验的形状和尺度参数。
- 逆Gamma分布常用来建模方差参数的先验，具有良好的数值性质。

### 3. 后验分布推导

为了推导 $ \sigma^2_e $ 的条件后验分布，我们使用贝叶斯定理。根据贝叶斯定理，后验分布与似然函数和先验分布的乘积成正比：
$$
p(\sigma^2_e | y, X, \beta, u, \sigma^2_g) \propto p(y | X, \beta, u, \sigma^2_e) \cdot p(\sigma^2_e)
$$

其中：
- **$ p(y | X, \beta, u, \sigma^2_e) $** 是表型 $ y $ 给定模型参数的似然函数。
- **$ p(\sigma^2_e) $** 是 $ \sigma^2_e $ 的先验分布。

#### 1. 似然函数

给定 $ X \beta + Z u $ 的预测值，表型 $ y $ 的条件分布为：
$$
y | X, \beta, u, \sigma^2_e \sim N(X \beta + Z u, \sigma^2_e I)
$$
根据这一正态分布，似然函数可以写作：
$$
p(y | X, \beta, u, \sigma^2_e) = N(y | X\beta + Z u, \sigma^2_e I)
$$
展开为指数形式：
$$
p(y | X, \beta, u, \sigma^2_e) \propto \exp\left( -\frac{1}{2\sigma^2_e} (y - X \beta - Z u)^T (y - X \beta - Z u) \right)
$$
这是 $ y $ 在给定 $ X\beta $ 和 $ Z u $ 之后的似然函数。

#### 2. $ \sigma^2_e $ 的先验分布

根据设定的先验分布，$ \sigma^2_e $ 服从逆Gamma分布：
$$
p(\sigma^2_e) \propto (\sigma^2_e)^{-\alpha_e - 1} \exp\left( -\frac{\beta_e}{\sigma^2_e} \right)
$$

#### 3. 后验分布的推导

结合似然函数和先验分布，$ \sigma^2_e $ 的后验分布为：
$$
p(\sigma^2_e | y, X, \beta, u, \sigma^2_g) \propto (\sigma^2_e)^{-\alpha_e - 1} \exp\left( -\frac{\beta_e}{\sigma^2_e} \right) \cdot \exp\left( -\frac{1}{2\sigma^2_e} (y - X \beta - Z u)^T (y - X \beta - Z u) \right)
$$

我们将这些项整理在一起：
$$
p(\sigma^2_e | y, X, \beta, u, \sigma^2_g) \propto (\sigma^2_e)^{-\alpha_e - 1} \exp\left( -\frac{1}{\sigma^2_e} \left( \beta_e + \frac{1}{2} (y - X \beta - Z u)^T (y - X \beta - Z u) \right) \right)
$$

### 4. 化简后验分布

从上面的公式可以看出，$ \sigma^2_e $ 的条件后验分布也是一个**逆Gamma分布**：
$$
\sigma^2_e | y, X, \beta, u \sim \text{Inverse-Gamma}\left( \alpha_e + \frac{n}{2}, \beta_e + \frac{1}{2} (y - X \beta - Z u)^T (y - X \beta - Z u) \right)
$$
其中：
- **形状参数**：$ \alpha_e + \frac{n}{2} $。
- **尺度参数**：$ \beta_e + \frac{1}{2} (y - X \beta - Z u)^T (y - X \beta - Z u) $。

这个逆Gamma分布的形状参数和尺度参数分别取决于先验的参数 $ \alpha_e $ 和 $ \beta_e $，以及当前模型的残差平方和 $ (y - X \beta - Z u)^T (y - X \beta - Z u) $。

### 5. MCMC 更新过程

在 MCMC 采样过程中，$ \sigma^2_e $ 的更新步骤如下：

1. **计算当前的残差**：首先计算当前模型的残差 $ r = y - X \beta - Z u $，并计算残差的平方和 $ r^T r $。
2. **根据条件后验分布采样 $ \sigma^2_e $**：使用逆Gamma分布 $ \text{Inverse-Gamma}\left( \alpha_e + \frac{n}{2}, \beta_e + \frac{1}{2} r^T r \right) $ 采样新的 $ \sigma^2_e $。

### 6. 总结

在BSLMM中，误差方差 $ \sigma^2_e $ 的估计过程依赖于模型的残差。通过推导后验分布，我们得出 $ \sigma^2_e $ 的条件后验分布为逆Gamma分布。在 MCMC 采样中，我们通过 Gibbs 采样从这个逆Gamma分布中更新 $ \sigma^2_e $ 的值。

# 代码实现
```python
import numpy as np

# 假设我们有一些数据：n 个样本和 p 个 SNPs
n, p = 200, 10
np.random.seed(42)
X = np.random.normal(0, 1, (n, p))  # 基因型矩阵

# 生成一些 true 的模型参数
true_beta = np.random.normal(0, 1, p)  # 每个 SNP 的效应大小
true_beta[np.random.rand(p) < 0.5] = 0  # 设定部分 SNP 的效应为零（稀疏性）
true_sigma_u = 1.0  # 随机效应的标准差
true_sigma_e = 1.0  # 残差的标准差

# 亲缘关系矩阵 G，可以通过基因型矩阵计算得到
G = np.dot(X, X.T) / p

# 随机效应部分
true_u = np.random.multivariate_normal(mean=np.zeros(n), cov=true_sigma_u**2 * G)

# 生成观测值 y
mu = np.dot(X, true_beta) + true_u
epsilon = np.random.normal(0, true_sigma_e, n)
y = mu + epsilon

# 初始化参数用于估计
beta = np.random.normal(0, 1, p)  # 每个 SNP 的效应大小
sigma_u = 1.0  # 随机效应的标准差
sigma_e = 1.0  # 残差的标准差
pi = 0.3  # 非零效应的比例
# 随机效应部分
u = np.random.multivariate_normal(mean=np.zeros(n), cov=sigma_u**2 * G)

# Gibbs采样迭代过程
def gibbs_sampling(iterations, X, y, beta, sigma_u, sigma_e, G, u, pi):
    beta_samples = np.zeros((iterations, p))
    sigma_u_samples = np.zeros(iterations)
    sigma_e_samples = np.zeros(iterations)
    for it in range(iterations):
        # 更新 beta
        for j in range(p):
            beta_j_mean = np.dot(X[:, j], y - np.dot(X, beta) + X[:, j] * beta[j] - u) / (np.dot(X[:, j], X[:, j]) + sigma_e**2 / 1)
            beta_j_var = sigma_e**2 / (np.dot(X[:, j], X[:, j]) + sigma_e**2 / 1)
            # 基于稀疏先验进行采样（混合正态和Dirac分布）
            if np.random.rand() < pi:  # 非零效应
                beta[j] = np.random.normal(beta_j_mean, np.sqrt(beta_j_var))
            else:  # 零效应
                beta[j] = 0
            beta_samples[it, j] = beta[j]
        
        # 更新 u
        u_mean = np.dot(G, (y - np.dot(X, beta))) / (sigma_e**2 + sigma_u**2) 
        u_cov = sigma_u**2 * G / (sigma_e**2 + sigma_u**2)
        u = np.random.multivariate_normal(mean=u_mean, cov=u_cov)
        
        # 更新 sigma_u 和 sigma_e
        sigma_u = np.sqrt(np.sum(u**2) / np.random.chisquare(n))
        sigma_e = np.sqrt(np.sum((y - np.dot(X, beta) - u)**2) / np.random.chisquare(n))
        sigma_u_samples[it] = sigma_u
        sigma_e_samples[it] = sigma_e
    return beta_samples, sigma_u_samples, sigma_e_samples

# 运行Gibbs采样
beta_samples, sigma_u_samples, sigma_e_samples = gibbs_sampling(1000, X, y, beta, sigma_u, sigma_e, G, u, pi)

# 输出结果
print("True beta:", true_beta.round(2))
print("Estimated beta:", beta_samples[-100:,:].mean(axis=0).round(2))
print("True sigma_u:", true_sigma_u)
print("Estimated sigma_u:", sigma_u_samples[-100:].mean())
print("True sigma_e:", true_sigma_e)
print("Estimated sigma_e:", sigma_e_samples[-100:].mean())
```