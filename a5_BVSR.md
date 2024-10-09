# BVSR 基本框架
 **Bayesian Variable Selection Regression (BVSR)** 是一个相对较简单的贝叶斯模型，它适用于GWAS中的特征选择问题。BVSR模型可以让你更好地理解贝叶斯框架如何处理稀疏效应以及选择与性状相关的重要基因位点。

### Bayesian Variable Selection Regression (BVSR) 原理：
BVSR主要用来解决以下问题：在众多的基因标记中，只有少部分对表型有显著影响，如何选择出这些标记？贝叶斯方法通过引入先验信息，能够灵活地控制对稀疏性的假设。

1. **模型表示**：
   $$
   y = X\beta + \epsilon
   $$
   这里，$ y $ 是目标表型，$ X $ 是基因型矩阵，$ \beta $ 是每个SNP的效应大小，$ \epsilon $ 是误差项。

2. **先验分布**：
   在BVSR中，假设大多数标记的效应为0，只有一小部分标记有非零效应。因此，可以为每个标记的效应系数 $ \beta $ 分配一个稀疏先验分布（通常是双峰分布或混合正态分布）：
   $$
   p(\beta_j) = \pi \cdot N(0, \sigma^2) + (1 - \pi) \cdot \delta_0
   $$
   其中，$\pi$ 是标记效应为非零的概率，$\delta_0$ 表示没有效应（即效应为0），$N(0, \sigma^2)$ 表示非零效应的正态分布。

3. **似然函数**：
   根据标准线性回归模型的假设，表型 $ y $ 的似然函数可以写为：
   $$
   p(y | X, \beta) = N(X\beta, \sigma_y^2)
   $$
   其中，$ \sigma_y^2 $ 是表型误差的方差。

4. **后验分布**：
   基于贝叶斯定理，我们可以通过先验和似然函数来得到后验分布，即：
   $$
   p(\beta | y, X)  = \frac{p(y | X, \beta) \cdot p(\beta)}{p(y | X)} \propto p(y | X, \beta) \cdot p(\beta)
   $$
   其中，**$ p(y | X) $**：**边际似然（证据）**  
   边际似然或称为“证据”，表示给定基因型 $ X $ 时，表型 $ y $ 的总概率。这是通过在所有可能的 $ \beta $ 值上对似然函数和先验分布进行积分得到的：
   $$
   p(y | X) = \int p(y | X, \beta) \cdot p(\beta) \, d\beta
   $$
   边际似然是一个归一化常数，确保后验分布是一个合法的概率分布。
   我们的目标是求出每个标记的后验概率，来评估它是否与表型显著相关。
   
   *注意：在以上的后验分布的公式中，$ p(\beta) $ 实际上是$p(\beta | X)$, 由于$\beta$ 是先验分布，与X 无关，因此$ p(\beta) = p(\beta | X)$。所以以上的公式中，分子部分,根据条件链式法则：
   $$
   p(y | X, \beta) \cdot p(\beta) = p(y | X, \beta) \cdot p(\beta | X) = p(y, \beta | X)
   $$
   分母乘以左侧部分:
   $$
   p(y | X)\cdot p(\beta|y, X) = p(y, \beta | X) 
   $$
   这里做此说明，是方便概率学的不好的同学的。
5. **求解方法**：
   由于直接计算后验分布很复杂，BVSR通常使用Markov Chain Monte Carlo (MCMC)算法进行近似推断。MCMC通过随机抽样来逼近后验分布，进而得到每个标记的显著性估计。

# 吉布斯采样

吉布斯采样是**马尔可夫链蒙特卡洛（MCMC）**方法中的一种，用于从高维联合分布中生成样本，尤其适用于变量之间的条件分布能够方便计算的情况。在Bayesian Variable Selection Regression (BVSR) 模型中，我们使用吉布斯采样来依次更新每个SNP效应 $ \beta_j $，基于它们的条件分布。

## 吉布斯采样的基本原理

在高维参数空间中，我们的目标是从后验分布 $ p(\beta | y, X) $ 中采样，但由于 $ \beta $ 是一个多维向量，直接从其联合分布采样可能很困难。然而，如果我们可以计算每个参数的**条件分布**，即 $ \beta_j $ 在给定其余参数 $ \beta_{-j} $ 的情况下的分布，我们可以使用吉布斯采样来实现逐个参数的更新，从而得到整个参数向量 $ \beta $ 的样本。

## 吉布斯采样的步骤

假设 $ \beta = (\beta_1, \beta_2, ..., \beta_p) $ 是我们想要采样的参数向量，它的联合后验分布是 $ p(\beta | y, X) $。吉布斯采样的过程如下：

1. 初始化 $ \beta $ 的所有元素，例如 $ \beta = 0 $。初始化误差 $\epsilon$
2. **循环迭代**，对于每个 $ j = 1, 2, ..., p $，根据其他参数 $ \beta_{-j} $ 的当前值，从条件分布 $ p(\beta_j | \beta_{-j}, y, X) $ 中采样更新 $ \beta_j $。
3. 更新 $\sigma^2$：在每次更新完所有 $ \beta_j $ 后，计算残差 residual，并利用当前残差的平方和更新 $\sigma^2$。
4. 重复这个过程若干次，直到马尔可夫链收敛，最后的采样点将近似来自后验分布 $ p(\beta | y, X) $ 和 $p(\sigma^2 | y, X, \beta)$。

# 吉布斯采样求解BVSR

## 一. 吉布斯采样$\beta$ 更新
### 1. 问题回顾

我们有一个线性回归模型：
$$
y = X\beta + \epsilon
$$
其中：
- $ y $ 是 $ n \times 1 $ 的表型向量。
- $ X $ 是 $ n \times p $ 的基因型矩阵。
- $ \beta $ 是 $ p \times 1 $ 的SNP效应向量。
- $ \epsilon $ 是误差，假设 $ \epsilon \sim N(0, \sigma^2 I) $。

我们希望对每个 $ \beta_j $ 进行条件采样，基于其他 $ \beta_{-j} $ （即除 $ j $ 以外的所有SNP效应）来更新 $ \beta_j $ 的条件分布。

### 2. 使用贝叶斯定理

我们知道根据贝叶斯定理，后验分布 $ p(\beta_j | y, X, \beta_{-j}) $ 可以写成：
$$
p(\beta_j | y, X, \beta_{-j}) \propto p(y | X, \beta) \cdot p(\beta_j)
$$
这里：
- $ p(y | X, \beta) $ 是似然函数，即给定 $ \beta_j $ 的情况下表型 $ y $ 的可能性。
- $ p(\beta_j) $ 是先验分布，即我们对 $ \beta_j $ 的先验假设，假设 $ \beta_j \sim N(0, \sigma^2_{\beta}) $。

### 3. 条件分布推导的具体步骤

现在我们要推导 $ \beta_j $ 的条件分布 $ p(\beta_j | y, X, \beta_{-j}) $，这涉及将数据分解为与 $ \beta_j $ 相关和不相关的部分。我们可以将回归模型重新写成：

$$
y = X_j \beta_j + X_{-j} \beta_{-j} + \epsilon
$$

这里：
- $ X_j $ 是第 $ j $ 列的基因型矩阵。
- $ X_{-j} $ 是去除第 $ j $ 列的基因型矩阵。
- $ \beta_j $ 是第 $ j $ 个SNP的效应。
- $ \beta_{-j} $ 是去除第 $ j $ 个SNP效应的向量。

为了简化问题，我们将 $ X_{-j} \beta_{-j} $ 部分看作已知量，并称为**残差**：

$$
r = y - X_{-j} \beta_{-j}
$$

因此，模型变为：
$$
r = X_j \beta_j + \epsilon
$$
这里，$ r $ 是已经去除了第 $ j $ 个SNP效应的残差。现在的任务是从这个新的模型中推导出 $ \beta_j $ 的条件分布。

#### 似然函数

在这个简化模型下，残差 $ r $ 是由 $ \beta_j $ 线性回归出来的，因此似然函数是：

$$
p(r | X_j, \beta_j) \propto \exp\left( -\frac{1}{2\sigma^2} (r - X_j \beta_j)^T (r - X_j \beta_j) \right)
$$

这个似然函数是基于高斯分布的标准形式。

#### 先验分布

假设我们对 $ \beta_j $ 施加了正态分布先验，即：

$$
p(\beta_j) \sim N(0, \sigma^2_{\beta})
$$

先验分布为：

$$
p(\beta_j) \propto \exp\left( -\frac{1}{2\sigma^2_{\beta}} \beta_j^2 \right)
$$

#### 后验分布

现在我们结合先验分布和似然函数，通过贝叶斯定理得到后验分布：

$$
p(\beta_j | r, X_j) \propto \exp\left( -\frac{1}{2\sigma^2} (r - X_j \beta_j)^T (r - X_j \beta_j) \right) \cdot \exp\left( -\frac{1}{2\sigma^2_{\beta}} \beta_j^2 \right)
$$

这是我们需要的后验分布，它是一个正态分布的乘积。接下来，我们对这个分布进行展开和化简，得到后验分布的均值和方差。

### 4. 后验分布化简

首先，我们对第一个指数项进行展开：

$$
(r - X_j \beta_j)^T (r - X_j \beta_j) = r^T r - 2 r^T X_j \beta_j + \beta_j^T X_j^T X_j \beta_j
$$

因此，似然函数可以写为：

$$
p(r | X_j, \beta_j) \propto \exp\left( -\frac{1}{2\sigma^2} \left( r^T r - 2 r^T X_j \beta_j + \beta_j^T X_j^T X_j \beta_j \right) \right)
$$

结合先验分布中的项，我们有：

$$
p(\beta_j | r, X_j) \propto \exp\left( -\frac{1}{2\sigma^2} \left( \beta_j^T X_j^T X_j \beta_j - 2 r^T X_j \beta_j \right) - \frac{1}{2\sigma^2_{\beta}} \beta_j^2 \right)
$$

将两个指数项合并后，我们得到：

$$
p(\beta_j | r, X_j) \propto \exp\left( -\frac{1}{2} \left( \frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}} \right) \beta_j^2 + \frac{r^T X_j}{\sigma^2} \beta_j \right)
$$

这是一个关于 $ \beta_j $ 的二次型表达式，它表示的是正态分布的指数形式。根据正态分布的标准形式，我们可以从中提取出均值和方差。

### 5.提取后验分布的均值和方差


#### a. 标准正态分布的形式

标准一维正态分布的概率密度函数形式是：

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

这个公式可以展开为指数的二次项：

$$
p(x) \propto \exp\left( -\frac{1}{2\sigma^2} \left( x^2 - 2\mu x + \mu^2 \right) \right)
$$

从中可以看到：
- **系数 $ \frac{1}{2\sigma^2} $ 前的 $ x^2 $** 项，决定了方差 $ \sigma^2 $。
- **系数 $ -2\mu $** 前的 $ x $ 项，决定了均值 $ \mu $。

因此，正态分布的二次型表达式的一般形式是：
$$
\exp\left( -\frac{1}{2\sigma^2} (x - \mu)^2 \right)
$$
其中，$ x^2 $ 项的系数给出了方差的倒数，$ x $ 项的系数给出了均值。

#### b. 后验分布中的二次型形式

现在让我们回到我们推导出的 $ \beta_j $ 的后验分布。我们得到了如下形式的指数：

$$
p(\beta_j | r, X_j) \propto \exp\left( -\frac{1}{2} \left( \frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}} \right) \beta_j^2 + \frac{r^T X_j}{\sigma^2} \beta_j \right)
$$

这个公式中的二次项与标准正态分布形式相似，现在我们逐项对比，以提取均值和方差。

#### c. 提取方差和均值

##### 方差的提取

在上面的公式中，$ \beta_j^2 $ 前面的系数是：
$$
\frac{1}{2} \left( \frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}} \right)
$$

这与正态分布中 $ \frac{1}{2\sigma^2} $ 的系数对比，因此可以直接得出方差的倒数是：
$$
\frac{1}{\text{var}_j} = \frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}}
$$

从中我们可以解出 $ \beta_j $ 的条件分布的方差 $ \text{var}_j $ 为：
$$
\text{var}_j = \frac{1}{\frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}}}
$$

这就是我们从二次项 $ \beta_j^2 $ 的系数中提取出来的方差。

##### 均值的提取

接下来我们看 $ \beta_j $ 的线性项（$ \beta_j $ 本身）的系数。在后验分布中，线性项的系数是：
$$
\frac{r^T X_j}{\sigma^2}
$$

在标准正态分布中，这个项是 $ -\frac{\mu}{\sigma^2} \beta_j $，因此我们可以对比得出均值 $ \mu_j $ 为：
$$
\text{mean}_j = \text{var}_j \cdot \frac{r^T X_j}{\sigma^2}
$$

这个公式表示，均值是方差 $ \text{var}_j $ 与数据项 $ \frac{r^T X_j}{\sigma^2} $ 的乘积。


#### 6. 后验分布的均值和方差

根据正态分布的二次型表达式，我们可以识别出后验分布的方差和均值。

#### 方差：

系数 $ \beta_j^2 $ 前面的项表示方差的倒数。因此，后验分布的方差 $ \text{var}_j $ 为：

$$
\text{var}_j = \frac{1}{\frac{X_j^T X_j}{\sigma^2} + \frac{1}{\sigma^2_{\beta}}}
$$

#### 均值：

均值可以从 $ \beta_j $ 的线性项中提取出来。后验分布的均值 $ \text{mean}_j $ 为：

$$
\text{mean}_j = \text{var}_j \cdot \frac{X_j^T r}{\sigma^2}
$$

这个推导过程通过贝叶斯推理结合了先验信息和观测数据的信息，生成了 $ \beta_j $ 的条件分布。这为后续的吉布斯采样提供了基础，每次迭代中依次更新每个 $ \beta_j $ 的值，直到模型收敛。

## 二. Gibbs 采样 $\sigma^2$更新
让我们完整推导为什么在 Gibbs 采样中更新误差方差 $\sigma^2$ 时会得到这些特定的后验参数更新。

### 问题背景
我们假设模型如下：
$$
y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$
其中：
- $y$ 是 $n \times 1$ 的响应变量向量，
- $X$ 是 $n \times p$ 的特征矩阵，
- $\beta$ 是 $p \times 1$ 的回归系数向量，
- $\epsilon$ 是误差项，服从均值为 0、方差为 $\sigma^2 I$ 的多元正态分布。

我们的目标是推导出 $\sigma^2$ 的后验分布参数更新。

### Step 1: 定义似然函数

给定 $\beta$ 和 $\sigma^2$，$y$ 的条件分布是多元正态分布：
$$
y | X, \beta, \sigma^2 \sim \mathcal{N}(X \beta, \sigma^2 I)
$$
因此，似然函数为：
$$
p(y | X, \beta, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - X_i \beta)^2\right)
$$
将残差表示为 $r = y - X \beta$，则可以重写为：
$$
p(y | X, \beta, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n r_i^2\right)
$$
其中 $\sum_{i=1}^n r_i^2 = r^T r$ 是残差平方和。

### Step 2: 定义 $\sigma^2$ 的先验分布

我们对 $\sigma^2$ 设定逆 Gamma 分布的先验：
$$
p(\sigma^2) = \text{Inverse-Gamma}(\alpha_0, \beta_0)
$$
其概率密度函数为：
$$
p(\sigma^2) = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)} (\sigma^2)^{-\alpha_0 - 1} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
$$
其中 $\alpha_0$ 和 $\beta_0$ 是超参数。

### Step 3: 后验分布的计算

根据贝叶斯公式，$\sigma^2$ 的后验分布为：
$$
p(\sigma^2 | y, X, \beta) \propto p(y | X, \beta, \sigma^2) \cdot p(\sigma^2)
$$
将似然函数和先验分布的表达式代入，可以得到：
$$
p(\sigma^2 | y, X, \beta) \propto \frac{1}{(\sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} \sum_{i=1}^n r_i^2\right) \cdot (\sigma^2)^{-\alpha_0 - 1} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
$$
我们将相同的 $\sigma^2$ 指数项合并，得到：
$$
p(\sigma^2 | y, X, \beta) \propto (\sigma^2)^{-\left(\frac{n}{2} + \alpha_0 + 1\right)} \exp\left(-\frac{1}{\sigma^2} \left(\frac{1}{2} \sum_{i=1}^n r_i^2 + \beta_0\right)\right)
$$
这与逆 Gamma 分布的形式一致，因此，后验分布为逆 Gamma 分布。

### Step 4: 后验分布参数的更新

由上述形式可以看出，后验分布 $p(\sigma^2 | y, X, \beta)$ 是一个逆 Gamma 分布，其参数更新如下：
1. **形状参数（Shape parameter）**：形状参数 $\alpha_{\text{post}}$ 是先验形状参数 $\alpha_0$ 加上样本数的一半：
   $$
   \alpha_{\text{post}} = \alpha_0 + \frac{n}{2}
   $$

2. **尺度参数（Scale parameter）**：尺度参数 $\beta_{\text{post}}$ 是先验尺度参数 $\beta_0$ 加上残差平方和的一半：
   $$
   \beta_{\text{post}} = \beta_0 + \frac{1}{2} \sum_{i=1}^n (y_i - X_i \beta)^2
   $$
   
### 小结
通过使用似然函数与先验分布相乘的方式，我们得到后验分布也是一个逆 Gamma 分布，其参数更新为：
$$
\alpha_{\text{post}} = \alpha_0 + \frac{n}{2}, \quad \beta_{\text{post}} = \beta_0 + \frac{1}{2} \sum_{i=1}^n (y_i - X_i \beta)^2
$$
这样，我们就完成了对 $\sigma^2$ 的后验参数更新推导。

# 代码实现

```python
import numpy as np

# 模拟数据生成
np.random.seed(42)
n, p = 1000, 10  # 样本数和特征数
X = np.random.randn(n, p)
true_beta = np.array([1.5, -2, 0, 0, 0.5] + [0]*(p-5))
y = X @ true_beta + np.random.randn(n) * 0.5

# 初始参数设置
n_iter = 1000
beta = np.zeros(p)
sigma2 = 1.0  # 初始误差方差
pi = 0.5   # 非零效应的先验概率

# Gibbs采样
def gibbs_sampling(X, y, beta, sigma2, pi, n_iter, alpha_prior=2.0, beta_prior=1.0):
    n, p = X.shape
    beta_samples = np.zeros((n_iter, p))
    sigma2_samples = np.zeros(n_iter)

    for it in range(n_iter):
        # 先更新每个beta_j
        for j in range(p):
            X_j = X[:, j]
            X_rest = X[:, np.arange(p) != j]
            beta_rest = beta[np.arange(p) != j]

            # 根据当前sigma2更新beta_j
            residual = y - X_rest @ beta_rest
            var_j = 1 / (X_j.T @ X_j / sigma2 + 1)
            mean_j = var_j * (X_j.T @ residual / sigma2)
            
            # 基于稀疏先验进行采样（混合正态和Dirac分布）
            if np.random.rand() < pi:  # 非零效应
                beta[j] = np.random.normal(mean_j, np.sqrt(var_j))
            else:  # 零效应
                beta[j] = 0

        # 更新sigma2
        residual = y - X @ beta
        alpha_post = alpha_prior + n / 2
        beta_post = beta_prior + 0.5 * np.sum(residual ** 2)
        sigma2 = 1 / np.random.gamma(alpha_post, 1 / beta_post)
        
        beta_samples[it] = beta
        sigma2_samples[it] = sigma2
    
    return beta_samples, sigma2_samples

# 运行Gibbs采样
beta_samples, sigma2_samples = gibbs_sampling(X, y, beta, sigma2, pi, n_iter)

# 查看结果
beta_mean = np.mean(beta_samples, axis=0)
sigma2_mean = np.mean(sigma2_samples)
print("Estimated beta coefficients:", beta_mean)
print("Estimated sigma^2:", sigma2_mean)

```