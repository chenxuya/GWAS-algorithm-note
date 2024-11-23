## 0. BayesCpi的数学模型结构

**BayesCpi** 是一种贝叶斯变量选择模型，广泛应用于基因组选择和高维回归分析，尤其适用于遗传学中的单核苷酸多态性（SNP）效应估计。与 **BayesC** 模型类似，BayesCpi 假设大部分 SNP 对响应变量的效应为零，仅有少数 SNP 具有非零效应。不同之处在于，BayesCpi 模型中的效应非零比例 \( \Pi \) 不是固定的，而是作为一个待估参数，通过贝叶斯框架进行推断。

### 模型表示

假设我们有一个响应变量向量 \( \mathbf{y} \in \mathbb{R}^n \)，以及 \( p \) 个预测变量（如 SNP）构成的设计矩阵 \( \mathbf{X} \in \mathbb{R}^{n \times p} \)。此外，还有一个固定效应的设计矩阵 \( \mathbf{X_f} \in \mathbb{R}^{n \times q} \)，对应的固定效应回归系数向量 \( \boldsymbol{\beta_f} \in \mathbb{R}^q \)。

模型结构如下：

\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]

其中，

\[
\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma_e^2 \mathbf{I})
\]

- \( \boldsymbol{\beta} = (\beta_1, \beta_2, \dots, \beta_p)^T \) 是 SNP 效应的回归系数向量。
- \( \sigma_e^2 \) 是残差方差。
- \( \boldsymbol{\delta} = (\delta_1, \delta_2, \dots, \delta_p)^T \) 是指示变量向量，其中 \( \delta_j \in \{0, 1\} \)，表示第 \( j \) 个 SNP 是否被选入模型。

在 BayesCpi 模型中，指示变量 \( \delta_j \) 和效应非零比例 \( \Pi \) 的定义如下：

\[
\beta_j | \delta_j, \sigma_\beta^2 \sim \delta_j \cdot \mathcal{N}(0, \sigma_\beta^2) + (1 - \delta_j) \cdot \delta_0
\]

\[
\delta_j \sim \text{Bernoulli}(\Pi) \quad \forall j = 1, 2, \dots, p
\]

\[
\Pi \sim \text{Beta}(a, b)
\]

其中：

- \( \sigma_\beta^2 \) 是非零效应的方差。
- \( \Pi \) 是所有 SNP 被选入模型的共同先验概率，遵循 Beta 分布。
- \( a \) 和 \( b \) 是 Beta 分布的超参数。

此外，残差方差 \( \sigma_e^2 \) 和非零效应方差 \( \sigma_\beta^2 \) 也有各自的先验分布。

## 1. 先验分布假设

在 BayesCpi 模型中，所有参数的先验分布假设如下：

1. **回归系数 \( \beta_j \) 的先验分布**：

   \[
   \beta_j | \delta_j, \sigma_\beta^2 \sim \delta_j \cdot \mathcal{N}(0, \sigma_\beta^2) + (1 - \delta_j) \cdot \delta_0
   \]

   - 当 \( \delta_j = 1 \) 时，\( \beta_j \) 来自均值为 0、方差为 \( \sigma_\beta^2 \) 的正态分布。
   - 当 \( \delta_j = 0 \) 时，\( \beta_j = 0 \)。

2. **指示变量 \( \delta_j \) 的先验分布**：

   \[
   \delta_j \sim \text{Bernoulli}(\Pi) \quad \forall j = 1, 2, \dots, p
   \]

   - 每个 \( \delta_j \) 独立地以概率 \( \Pi \) 被选入模型。

3. **稀疏性参数 \( \Pi \) 的先验分布**：

   \[
   \Pi \sim \text{Beta}(a, b)
   \]

   - \( a \) 和 \( b \) 是 Beta 分布的超参数，通常反映对 \( \Pi \) 的先验信念。

4. **残差方差 \( \sigma_e^2 \) 的先验分布**：

   \[
   \sigma_e^2 \sim \text{Inverse-Gamma}(a_e, b_e)
   \]

   - \( a_e \) 和 \( b_e \) 是逆伽玛分布的超参数。

5. **非零效应方差 \( \sigma_\beta^2 \) 的先验分布**：

   \[
   \sigma_\beta^2 \sim \text{Inverse-Gamma}(a_\beta, b_\beta)
   \]

   - \( a_\beta \) 和 \( b_\beta \) 是逆伽玛分布的超参数。

6. **固定效应回归系数 \( \boldsymbol{\beta_f} \) 的先验分布**：

   \[
   \boldsymbol{\beta_f} \sim \mathcal{N}(\mathbf{0}, \tau_f^2 \mathbf{I})
   \]

   - \( \tau_f^2 \) 是固定效应回归系数的先验方差，通常设定为一个较大的值以表示弱先验。

## 2. 推导 \( \mathbf{y} \) 的似然函数，并根据贝叶斯定理列出后验分布

### 2.1 似然函数 \( p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \)

根据模型假设：

\[
\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2 \sim \mathcal{N}(\mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta}, \sigma_e^2 \mathbf{I})
\]

因此，似然函数为：

\[
p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma_e^2}} \exp\left( -\frac{(y_i - \mathbf{x_{f,i}}^T \boldsymbol{\beta_f} - \mathbf{x_i}^T \boldsymbol{\beta})^2}{2\sigma_e^2} \right)
\]

其中，\( \mathbf{x_{f,i}} \) 和 \( \mathbf{x_i} \) 分别是设计矩阵 \( \mathbf{X_f} \) 和 \( \mathbf{X} \) 的第 \( i \) 行。

### 2.2 后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}, \Pi, \sigma_e^2, \sigma_\beta^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \)

根据贝叶斯定理：

\[
p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}, \Pi, \sigma_e^2, \sigma_\beta^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f}) \cdot p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2) \cdot p(\boldsymbol{\delta} | \Pi) \cdot p(\Pi) \cdot p(\sigma_e^2) \cdot p(\sigma_\beta^2)
\]

具体来说，各个部分的含义如下：

- \( p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \) 是似然函数。
- \( p(\boldsymbol{\beta_f}) \) 是固定效应回归系数的先验分布。
- \( p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2) \) 是 SNP 效应回归系数的先验分布。
- \( p(\boldsymbol{\delta} | \Pi) \) 是指示变量的先验分布。
- \( p(\Pi) \) 是稀疏性参数的先验分布。
- \( p(\sigma_e^2) \) 和 \( p(\sigma_\beta^2) \) 分别是残差方差和非零效应方差的先验分布。

由于模型中的参数高度耦合，联合后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}, \Pi, \sigma_e^2, \sigma_\beta^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \) 是一个高维复杂分布，难以直接从中采样或进行解析求解。因此，需要采用数值方法，如马尔可夫链蒙特卡洛（MCMC）方法中的 Gibbs 采样进行参数的迭代更新。

## 3. 分布的复杂性与 Gibbs 采样的引入

后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}, \Pi, \sigma_e^2, \sigma_\beta^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \) 是一个高维、多峰的复杂分布，难以直接从中采样。因此，采用 MCMC 方法中的 **Gibbs 采样** 进行参数的迭代更新。

**Gibbs 采样** 的基本思想是通过依次从每个参数的条件后验分布中采样，逐步逼近联合后验分布。具体步骤如下：

1. **初始化**：为所有参数赋予初始值。
2. **迭代更新**：
   - 从 \( p(\boldsymbol{\beta_f} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta}, \boldsymbol{\delta}, \sigma_e^2) \) 中采样 \( \boldsymbol{\beta_f} \)。
   - 从 \( p(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\delta}, \sigma_e^2, \sigma_\beta^2) \) 中采样 \( \boldsymbol{\beta} \)。
   - 从 \( p(\boldsymbol{\delta} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \Pi, \sigma_e^2, \sigma_\beta^2) \) 中采样 \( \boldsymbol{\delta} \)。
   - 从 \( p(\Pi | \boldsymbol{\delta}, a, b) \) 中采样 \( \Pi \)。
   - 从 \( p(\sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}) \) 中采样 \( \sigma_e^2 \)。
   - 从 \( p(\sigma_\beta^2 | \boldsymbol{\beta}, \boldsymbol{\delta}, a_\beta, b_\beta) \) 中采样 \( \sigma_\beta^2 \)。
3. **重复** 迭代过程，直到收敛。
4. **汇总**：根据采样结果进行后验估计和变量选择。

由于每个条件后验分布通常属于已知的分布族（如正态分布、Beta 分布、逆伽玛分布），Gibbs 采样能够高效地执行参数更新。

## 4. 各参数的条件后验分布推导

以下详细推导 BayesCpi 模型中各参数的条件后验分布，包括固定效应回归系数 \( \boldsymbol{\beta_f} \)、SNP 效应回归系数 \( \boldsymbol{\beta} \)、指示变量 \( \boldsymbol{\delta} \)、稀疏性参数 \( \Pi \)、残差方差 \( \sigma_e^2 \)、和非零效应方差 \( \sigma_\beta^2 \)。

### 4.1 固定效应回归系数 \( \boldsymbol{\beta_f} \) 的条件后验分布

\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta}, \boldsymbol{\delta}, \sigma_e^2, \sigma_\beta^2) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f})
\]

根据模型假设：

\[
\boldsymbol{\beta_f} \sim \mathcal{N}(\mathbf{0}, \tau_f^2 \mathbf{I})
\]

因此，后验分布为：

\[
\boldsymbol{\beta_f} | \cdot \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \mathbf{\Sigma_{\beta_f}})
\]

其中，

\[
\mathbf{\Sigma_{\beta_f}} = \left( \frac{\mathbf{X_f}^T \mathbf{X_f}}{\sigma_e^2} + \frac{1}{\tau_f^2} \mathbf{I} \right)^{-1}
\]

\[
\boldsymbol{\mu_{\beta_f}} = \mathbf{\Sigma_{\beta_f}} \cdot \frac{\mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})}{\sigma_e^2}
\]

### 4.2 SNP 效应回归系数 \( \boldsymbol{\beta} \) 的条件后验分布

\[
p(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\delta}, \sigma_e^2, \sigma_\beta^2) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2)
\]

根据模型假设：

\[
p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2) = \prod_{j=1}^p \left[ \delta_j \cdot \mathcal{N}(\beta_j | 0, \sigma_\beta^2) + (1 - \delta_j) \cdot \delta_0(\beta_j) \right]
\]

因此，针对每个 \( \beta_j \)，其条件后验分布为：

\[
p(\beta_j | \cdot) = 
\begin{cases}
\mathcal{N}(\mu_j, \tau_j^2) & \text{如果 } \delta_j = 1 \\
\delta_0(\beta_j) & \text{如果 } \delta_j = 0
\end{cases}
\]

其中，

\[
\tau_j^2 = \frac{\sigma_e^2}{\mathbf{x_j}^T \mathbf{x_j} + \frac{\sigma_e^2}{\sigma_\beta^2}}
\]

\[
\mu_j = \tau_j^2 \cdot \frac{\mathbf{x_j}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j})}{\sigma_e^2}
\]

这里，\( \mathbf{x_j} \) 是设计矩阵 \( \mathbf{X} \) 的第 \( j \) 列，\( \mathbf{X}_{-j} \) 和 \( \boldsymbol{\beta}_{-j} \) 分别表示除第 \( j \) 个 SNP 外的其他 SNP 和对应的回归系数。

因此，SNP 效应回归系数 \( \beta_j \) 的更新规则为：

- 如果 \( \delta_j = 1 \)，则从正态分布 \( \mathcal{N}(\mu_j, \tau_j^2) \) 中采样。
- 如果 \( \delta_j = 0 \)，则 \( \beta_j = 0 \)。

### 4.3 指示变量 \( \delta_j \) 的条件后验分布

\[
p(\delta_j | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \Pi, \sigma_e^2, \sigma_\beta^2) \propto p(\beta_j | \delta_j, \sigma_\beta^2) \cdot p(\delta_j | \Pi)
\]

根据模型假设：

\[
p(\delta_j = 1 | \Pi) = \Pi
\]

\[
p(\delta_j = 0 | \Pi) = 1 - \Pi
\]

\[
p(\beta_j | \delta_j = 1, \sigma_\beta^2) = \mathcal{N}(\beta_j | 0, \sigma_\beta^2)
\]

\[
p(\beta_j | \delta_j = 0, \sigma_\beta^2) = \delta_0(\beta_j)
\]

因此，条件后验概率为：

\[
p(\delta_j = 1 | \cdot) = \frac{\Pi \cdot \mathcal{N}(\beta_j | 0, \sigma_\beta^2)}{\Pi \cdot \mathcal{N}(\beta_j | 0, \sigma_\beta^2) + (1 - \Pi) \cdot \delta_0(\beta_j)}
\]

由于 \( \delta_j = 0 \) 时 \( \beta_j = 0 \)，即：

- 如果 \( \beta_j \neq 0 \)，则 \( \delta_j = 1 \) 的概率为 1。
- 如果 \( \beta_j = 0 \)，则：

\[
p(\delta_j = 1 | \cdot) = \frac{\Pi \cdot \mathcal{N}(0 | 0, \sigma_\beta^2)}{\Pi \cdot \mathcal{N}(0 | 0, \sigma_\beta^2) + (1 - \Pi)}
\]

其中，

\[
\mathcal{N}(0 | 0, \sigma_\beta^2) = \frac{1}{\sqrt{2\pi \sigma_\beta^2}}
\]

因此，当 \( \beta_j = 0 \) 时，指示变量 \( \delta_j \) 的更新概率为：

\[
p(\delta_j = 1 | \cdot) = \frac{\Pi / \sqrt{2\pi \sigma_\beta^2}}{\Pi / \sqrt{2\pi \sigma_\beta^2} + (1 - \Pi)}
\]

为了提高数值稳定性，通常在计算时采用对数概率进行处理。

### 4.4 稀疏性参数 \( \Pi \) 的条件后验分布

\[
p(\Pi | \boldsymbol{\delta}, a, b) \propto p(\boldsymbol{\delta} | \Pi) \cdot p(\Pi)
\]

根据模型假设：

\[
p(\boldsymbol{\delta} | \Pi) = \prod_{j=1}^p \Pi^{\delta_j} (1 - \Pi)^{1 - \delta_j}
\]

\[
p(\Pi) = \text{Beta}(\Pi | a, b) = \frac{\Pi^{a-1} (1 - \Pi)^{b-1}}{B(a, b)}
\]

因此，后验分布为：

\[
p(\Pi | \boldsymbol{\delta}, a, b) \propto \Pi^{\sum_{j=1}^p \delta_j + a - 1} (1 - \Pi)^{\sum_{j=1}^p (1 - \delta_j) + b - 1}
\]

即，

\[
\Pi | \boldsymbol{\delta}, a, b \sim \text{Beta}\left(a + \sum_{j=1}^p \delta_j, \ b + p - \sum_{j=1}^p \delta_j\right) = \text{Beta}(a + k, \ b + p - k)
\]

其中，\( k = \sum_{j=1}^p \delta_j \) 是被选中的变量数量。

### 4.5 残差方差 \( \sigma_e^2 \) 的条件后验分布

\[
p(\sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\delta}, \sigma_\beta^2) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\sigma_e^2)
\]

根据模型假设：

\[
p(\mathbf{y} | \cdot) = \mathcal{N}(\mathbf{y} | \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta}, \sigma_e^2 \mathbf{I})
\]

\[
p(\sigma_e^2) = \text{Inverse-Gamma}(a_e, b_e)
\]

因此，后验分布为：

\[
p(\sigma_e^2 | \cdot) \propto (\sigma_e^2)^{-\frac{n}{2}} \exp\left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot (\sigma_e^2)^{-a_e - 1} \exp\left( -\frac{b_e}{\sigma_e^2} \right)
\]

整理得：

\[
\sigma_e^2 | \cdot \sim \text{Inverse-Gamma}\left(a_e + \frac{n}{2}, \ b_e + \frac{1}{2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right)
\]

### 4.6 非零效应方差 \( \sigma_\beta^2 \) 的条件后验分布

\[
p(\sigma_\beta^2 | \boldsymbol{\beta}, \boldsymbol{\delta}, a_\beta, b_\beta) \propto p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2) \cdot p(\sigma_\beta^2)
\]

根据模型假设：

\[
p(\boldsymbol{\beta} | \boldsymbol{\delta}, \sigma_\beta^2) = \prod_{j=1}^p \left[ \delta_j \cdot \mathcal{N}(\beta_j | 0, \sigma_\beta^2) + (1 - \delta_j) \cdot \delta_0(\beta_j) \right]
\]

\[
p(\sigma_\beta^2) = \text{Inverse-Gamma}(a_\beta, b_\beta)
\]

因此，后验分布为：

\[
p(\sigma_\beta^2 | \cdot) \propto \prod_{j=1}^p \left[ \delta_j \cdot \mathcal{N}(\beta_j | 0, \sigma_\beta^2) + (1 - \delta_j) \cdot \delta_0(\beta_j) \right] \cdot (\sigma_\beta^2)^{-a_\beta - 1} \exp\left( -\frac{b_\beta}{\sigma_\beta^2} \right)
\]

由于 \( \beta_j = 0 \) 当 \( \delta_j = 0 \)，因此可以简化为：

\[
p(\sigma_\beta^2 | \cdot) \propto \prod_{j: \delta_j = 1} \mathcal{N}(\beta_j | 0, \sigma_\beta^2) \cdot (\sigma_\beta^2)^{-a_\beta - 1} \exp\left( -\frac{b_\beta}{\sigma_\beta^2} \right)
\]

进一步展开：

\[
p(\sigma_\beta^2 | \cdot) \propto (\sigma_\beta^2)^{-\frac{k}{2}} \exp\left( -\frac{1}{2\sigma_\beta^2} \sum_{j: \delta_j = 1} \beta_j^2 \right) \cdot (\sigma_\beta^2)^{-a_\beta - 1} \exp\left( -\frac{b_\beta}{\sigma_\beta^2} \right)
\]

整理得：

\[
\sigma_\beta^2 | \cdot \sim \text{Inverse-Gamma}\left(a_\beta + \frac{k}{2}, \ b_\beta + \frac{1}{2} \sum_{j=1}^p \delta_j \beta_j^2 \right)
\]

其中，\( k = \sum_{j=1}^p \delta_j \) 是被选中的变量数量。

## 5. 算法更新步骤

基于上述条件后验分布，BayesCpi 模型的 Gibbs 采样算法更新步骤如下：

### 5.1 初始化

为所有参数赋予初始值：

- \( \boldsymbol{\beta_f}^{(0)} = \mathbf{0} \)
- \( \boldsymbol{\beta}^{(0)} = \mathbf{0} \)
- \( \boldsymbol{\delta}^{(0)} = \mathbf{0} \)
- \( \Pi^{(0)} = \frac{1}{p} \) 或其他合理值。
- \( \sigma_e^{2(0)} = 1 \)
- \( \sigma_\beta^{2(0)} = 1 \)

### 5.2 迭代更新

对每一次迭代 \( t = 1, 2, \dots, T \) 进行以下步骤：

#### 步骤1：更新固定效应回归系数 \( \boldsymbol{\beta_f} \)

根据条件后验分布：

\[
\boldsymbol{\beta_f} | \cdot \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \mathbf{\Sigma_{\beta_f}})
\]

其中，

\[
\mathbf{\Sigma_{\beta_f}} = \left( \frac{\mathbf{X_f}^T \mathbf{X_f}}{\sigma_e^{2(t-1)}} + \frac{1}{\tau_f^2} \mathbf{I} \right)^{-1}
\]

\[
\boldsymbol{\mu_{\beta_f}} = \mathbf{\Sigma_{\beta_f}} \cdot \frac{\mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}^{(t-1)})}{\sigma_e^{2(t-1)}}
\]

使用多元正态分布采样得到 \( \boldsymbol{\beta_f}^{(t)} \)。

#### 步骤2：更新 SNP 效应回归系数 \( \boldsymbol{\beta} \)

对于每个 SNP \( j = 1, 2, \dots, p \)：

\[
\beta_j^{(t)} | \cdot \sim 
\begin{cases}
\mathcal{N}(\mu_j, \tau_j^2) & \text{如果 } \delta_j^{(t-1)} = 1 \\
0 & \text{如果 } \delta_j^{(t-1)} = 0
\end{cases}
\]

其中，

\[
\tau_j^2 = \frac{\sigma_e^{2(t-1)}}{\mathbf{x_j}^T \mathbf{x_j} + \frac{\sigma_e^{2(t-1)}}{\sigma_\beta^{2(t-1)}}}
\]

\[
\mu_j = \tau_j^2 \cdot \frac{\mathbf{x_j}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}^{(t-1)} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j}^{(t-1)})}{\sigma_e^{2(t-1)}}
\]

- 如果 \( \delta_j = 1 \)，则从 \( \mathcal{N}(\mu_j, \tau_j^2) \) 中采样 \( \beta_j^{(t)} \)。
- 如果 \( \delta_j = 0 \)，则 \( \beta_j^{(t)} = 0 \)。

#### 步骤3：更新指示变量 \( \delta_j \)

对于每个 SNP \( j = 1, 2, \dots, p \)：

根据条件后验概率：

\[
p(\delta_j = 1 \mid \cdot) = 
\begin{cases} 
1 & \text{如果 } \beta_j^{(t)} \neq 0 \\
\frac{\Pi^{(t-1)} \cdot \mathcal{N}(0 \mid 0, \sigma_\beta^{2(t-1)})}{\Pi^{(t-1)} \cdot \mathcal{N}(0 \mid 0, \sigma_\beta^{2(t-1)}) + (1 - \Pi^{(t-1)})} & \text{如果 } \beta_j^{(t)} = 0
\end{cases}
\]



具体步骤如下：

1. **计算 \( p(\delta_j = 1 | \cdot) \)**：

   - 如果 \( \beta_j^{(t)} \neq 0 \)，则 \( p(\delta_j = 1 | \cdot) = 1 \)。
   - 如果 \( \beta_j^{(t)} = 0 \)，则：
     \[
     p(\delta_j = 1 | \cdot) = \frac{\Pi^{(t-1)} \cdot \frac{1}{\sqrt{2\pi \sigma_\beta^{2(t-1)}}}}{\Pi^{(t-1)} \cdot \frac{1}{\sqrt{2\pi \sigma_\beta^{2(t-1)}}} + (1 - \Pi^{(t-1)})}
     \]

2. **从 Bernoulli 分布中采样 \( \delta_j^{(t)} \)**：

   \[
   \delta_j^{(t)} \sim \text{Bernoulli}\left( p(\delta_j = 1 | \cdot) \right)
   \]

3. **同步更新**：

   - 如果 \( \delta_j^{(t)} = 0 \)，则设定 \( \beta_j^{(t)} = 0 \)。

**数值稳定性处理**：

为了避免在计算 \( p(\delta_j = 1 | \cdot) \) 时出现数值下溢或上溢，通常采用对数概率进行计算，并使用数值技巧进行归一化。例如：

\[
\log p(\delta_j = 1 \mid \cdot) = \log \Pi^{(t-1)} + \log \mathcal{N}(0 \mid 0, \sigma_\beta^{2(t-1)})
\]


\[
\log p(\delta_j = 0 | \cdot) = \log (1 - \Pi^{(t-1)})
\]

然后通过最大化 \( \log p(\delta_j = 1 | \cdot) \) 和 \( \log p(\delta_j = 0 | \cdot) \) 进行归一化，避免直接计算概率比率导致的数值问题。

#### 步骤4：更新稀疏性参数 \( \Pi \)

根据条件后验分布：

\[
\Pi^{(t)} \sim \text{Beta}\left(a + \sum_{j=1}^p \delta_j^{(t)}, \ b + p - \sum_{j=1}^p \delta_j^{(t)}\right)
\]

其中，\( k = \sum_{j=1}^p \delta_j^{(t)} \) 是被选中的变量数量。

#### 步骤5：更新残差方差 \( \sigma_e^2 \)

根据条件后验分布：

\[
\sigma_e^{2(t)} \sim \text{Inverse-Gamma}\left(a_e + \frac{n}{2}, \ b_e + \frac{1}{2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}^{(t)} - \mathbf{X} \boldsymbol{\beta}^{(t)})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}^{(t)} - \mathbf{X} \boldsymbol{\beta}^{(t)}) \right)
\]

其中，\( \mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}^{(t)} - \mathbf{X} \boldsymbol{\beta}^{(t)} \) 是残差向量。

#### 步骤6：更新非零效应方差 \( \sigma_\beta^2 \)

根据条件后验分布：

\[
\sigma_\beta^{2(t)} \sim \text{Inverse-Gamma}\left(a_\beta + \frac{k}{2}, \ b_\beta + \frac{1}{2} \sum_{j=1}^p \delta_j^{(t)} \beta_j^{2(t)} \right)
\]

其中，\( k = \sum_{j=1}^p \delta_j^{(t)} \) 是被选中的变量数量。

### 5.3 算法流程总结

1. **初始化** 参数 \( \boldsymbol{\beta_f}^{(0)} \), \( \boldsymbol{\beta}^{(0)} \), \( \boldsymbol{\delta}^{(0)} \), \( \Pi^{(0)} \), \( \sigma_e^{2(0)} \), \( \sigma_\beta^{2(0)} \)。
2. **对于每一次迭代 \( t \)**，执行以下步骤：
   - **步骤1**：更新固定效应回归系数 \( \boldsymbol{\beta_f} \)。
   - **步骤2**：更新 SNP 效应回归系数 \( \boldsymbol{\beta} \)。
   - **步骤3**：更新指示变量 \( \boldsymbol{\delta} \)。
   - **步骤4**：更新稀疏性参数 \( \Pi \)。
   - **步骤5**：更新残差方差 \( \sigma_e^2 \)。
   - **步骤6**：更新非零效应方差 \( \sigma_\beta^2 \)。
3. **重复** 迭代步骤，直到达到预设的迭代次数或满足收敛条件。
4. **结果汇总**：根据采样结果进行后验估计和变量选择。

### 5.4 算法实现示例

以下是一个基于上述推导的 BayesCpi 模型的 Gibbs 采样算法的 Python 实现示例：

```python
import numpy as np
from scipy.stats import norm, beta as beta_dist, invgamma

def bayesCpi_gibbs_sampler(X_f, X, y, num_iterations, 
                           a=1.0, b=1.0, 
                           a_e=2.0, b_e=2.0, 
                           a_beta=2.0, b_beta=2.0, 
                           tau_f2=1e6):
    """
    BayesCpi 模型的 Gibbs 采样算法

    参数：
    X_f: 固定效应的设计矩阵 (n x q)
    X: SNP 效应的设计矩阵 (n x p)
    y: 响应变量向量 (n,)
    num_iterations: 迭代次数
    a, b: Pi 的 Beta 分布超参数
    a_e, b_e: 残差方差的逆伽玛分布超参数
    a_beta, b_beta: 非零效应方差的逆伽玛分布超参数
    tau_f2: 固定效应回归系数的先验方差

    返回：
    beta_f_samples: 固定效应回归系数的采样 (num_iterations x q)
    beta_samples: SNP 效应回归系数的采样 (num_iterations x p)
    delta_samples: 指示变量的采样 (num_iterations x p)
    Pi_samples: Pi 的采样 (num_iterations,)
    sigma_e2_samples: 残差方差的采样 (num_iterations,)
    sigma_beta2_samples: 非零效应方差的采样 (num_iterations,)
    """
    n, p = X.shape
    q = X_f.shape[1]
    
    # 初始化参数
    beta_f = np.zeros(q)
    beta = np.zeros(p)
    delta = np.zeros(p, dtype=int)
    Pi = 0.5
    sigma_e2 = 1.0
    sigma_beta2 = 1.0
    
    # 存储采样结果
    beta_f_samples = np.zeros((num_iterations, q))
    beta_samples = np.zeros((num_iterations, p))
    delta_samples = np.zeros((num_iterations, p), dtype=int)
    Pi_samples = np.zeros(num_iterations)
    sigma_e2_samples = np.zeros(num_iterations)
    sigma_beta2_samples = np.zeros(num_iterations)
    
    # 预计算 X_f^T X_f
    X_fTX_f = X_f.T @ X_f
    
    for t in range(num_iterations):
        # 步骤1：更新固定效应回归系数 beta_f
        Sigma_beta_f_inv = (X_fTX_f / sigma_e2) + (1 / tau_f2) * np.eye(q)
        Sigma_beta_f = np.linalg.inv(Sigma_beta_f_inv)
        mu_beta_f = Sigma_beta_f @ (X_f.T @ (y - X @ beta)) / sigma_e2
        beta_f = np.random.multivariate_normal(mu_beta_f, Sigma_beta_f)
        
        # 步骤2：更新 SNP 效应回归系数 beta
        for j in range(p):
            if delta[j] == 1:
                X_j = X[:, j]
                # 计算残差
                residual = y - X_f @ beta_f - X @ beta + X_j * beta[j]
                tau_j2 = sigma_e2 / (X_j.T @ X_j + sigma_e2 / sigma_beta2)
                mu_j = tau_j2 * (X_j.T @ residual) / sigma_e2
                beta[j] = np.random.normal(mu_j, np.sqrt(tau_j2))
            else:
                beta[j] = 0.0
        
        # 步骤3：更新指示变量 delta
        for j in range(p):
            if beta[j] != 0:
                delta[j] = 1
            else:
                # 计算 p(delta_j = 1 | ...)
                prob_1 = Pi * norm.pdf(0, loc=0, scale=np.sqrt(sigma_beta2))
                prob_0 = 1 - Pi
                prob_delta_1 = prob_1 / (prob_1 + prob_0 + 1e-10)  # 加小数防止除零
                delta[j] = np.random.binomial(1, prob_delta_1)
                if delta[j] == 0:
                    beta[j] = 0.0
        
        # 步骤4：更新 Pi
        k = np.sum(delta)
        Pi = beta_dist.rvs(a + k, b + p - k)
        
        # 步骤5：更新残差方差 sigma_e2
        residual = y - X_f @ beta_f - X @ beta
        shape_e = a_e + n / 2
        scale_e = b_e + 0.5 * np.sum(residual**2)
        sigma_e2 = invgamma.rvs(a=shape_e, scale=scale_e)
        
        # 步骤6：更新非零效应方差 sigma_beta2
        sum_beta_sq = np.sum(delta * beta**2)
        shape_beta = a_beta + np.sum(delta) / 2
        scale_beta = b_beta + 0.5 * sum_beta_sq
        sigma_beta2 = invgamma.rvs(a=shape_beta, scale=scale_beta)
        
        # 存储采样结果
        beta_f_samples[t, :] = beta_f
        beta_samples[t, :] = beta
        delta_samples[t, :] = delta
        Pi_samples[t] = Pi
        sigma_e2_samples[t] = sigma_e2
        sigma_beta2_samples[t] = sigma_beta2
        
        # 可选：打印迭代进度
        if (t+1) % 1000 == 0 or t == 0:
            print(f"Iteration {t+1}/{num_iterations} completed.")
    
    return beta_f_samples, beta_samples, delta_samples, Pi_samples, sigma_e2_samples, sigma_beta2_samples

# 使用示例

# 假设数据已经准备好
# X_f: 固定效应的设计矩阵 (n x q)
# X: SNP 效应的设计矩阵 (n x p)
# y: 响应变量向量 (n,)

# 设置先验参数
a, b = 1.0, 1.0
a_e, b_e = 2.0, 2.0
a_beta, b_beta = 2.0, 2.0
num_iterations = 10000

# 运行 Gibbs 采样
# beta_f_samples, beta_samples, delta_samples, Pi_samples, sigma_e2_samples, sigma_beta2_samples = bayesCpi_gibbs_sampler(X_f, X, y, num_iterations, a, b, a_e, b_e, a_beta, b_beta)

# 后续分析，如绘制参数的收敛诊断图
# import matplotlib.pyplot as plt
# plt.plot(beta_f_samples[:, 0])
# plt.title('Trace plot for beta_f[0]')
# plt.show()
```

**注意事项**：

1. **数值稳定性**：在计算 \( p(\delta_j = 1 | \cdot) \) 时，加入一个非常小的常数（如 \( 1e-10 \)）以避免除零错误。同时，建议在计算对数概率时采用数值稳定的技巧，如对数空间运算，以防止概率下溢或上溢。

2. **向量化优化**：当前实现中，对 \( \boldsymbol{\beta} \) 和 \( \boldsymbol{\delta} \) 的更新使用了显式循环，这在大规模数据中可能导致效率较低。可以进一步向量化这些步骤以提高计算效率。例如，可以一次性处理所有 \( \delta_j \) 的更新。

3. **收敛诊断**：在实际应用中，应对 Gibbs 采样的结果进行收敛性诊断，如使用 Gelman-Rubin 诊断、观察参数的轨迹图或计算自相关系数，以确保采样过程已经收敛。

4. **初始化**：参数的初始化选择可能会影响采样的收敛速度和结果质量。可以尝试不同的初始化策略，如随机初始化或基于某些预处理结果的初始化，以提高算法的性能。

5. **后验样本处理**：通常需要在采样过程中舍弃前面的“burn-in”期的样本，并对采样结果进行后处理，如计算后验均值、标准差和可信区间。此外，可以使用抽样间隔（thinning）来减少样本之间的自相关，提高样本的独立性。


## 6. 总结

**BayesCpi** 模型通过引入可变的稀疏性参数 \( \Pi \) 和指示变量 \( \boldsymbol{\delta} \)，在贝叶斯框架下实现了高维数据中的变量选择和效应估计。通过 Gibbs 采样方法，能够有效地从复杂的后验分布中采样，进而进行参数的推断和模型的学习。上述详细的数学推导和算法步骤为实现和理解 BayesCpi 模型提供了坚实的理论基础。

在实际应用中，BayesCpi 模型因其灵活性和高效性，成为遗传学和其他高维数据分析领域中的重要工具。通过适当的参数设置和算法优化，BayesCpi 能够在大规模数据中实现有效的变量选择和精确的效应估计。\\[\\]\\(\\)