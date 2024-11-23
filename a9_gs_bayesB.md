# 1. 数学原理
BayesB 是一种用于基因组选择的贝叶斯方法，与 BayesA 相比，它引入了一个重要的假设，即并不是所有的标记位点（SNP）都对性状有贡献。BayesB 假设大多数 SNP 的效应为 0，而只有一部分 SNP 具有显著效应。这种稀疏性假设使得 BayesB 更适合处理高维数据（即很多 SNP）时，尤其是当我们预期大多数 SNP 标记效应较小或为零。
## 1.1 模型表达式
在 **BayesB（pi）** 中，线性模型用于描述表型 \( \mathbf{y} \) 与固定效应和 SNP 效应之间的关系：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
或
\[
\begin{aligned}
y | X, \beta, \sigma^2 &\sim \mathcal{N}(X \beta + X_f \beta_f, \sigma^2 I) \\
\beta_j | \delta_j, \sigma_j^2 &\sim \delta_j \cdot \mathcal{N}(0, \sigma_j^2) + (1 - \delta_j) \cdot \delta_0 \\
\delta_j &\sim \text{Bernoulli}(\pi) \quad \forall j = 1, 2, \dots, p \\
\pi &\sim \text{Beta}(a, b) \\
\sigma^2 &\sim \text{Inverse-Gamma}(a_e, b_e) \\
\sigma_j^2 &\sim \text{Inverse-Gamma}(a_j, b_j) \quad \forall j
\end{aligned}
\]

其中：
- \( y \) 是响应变量。
- \( X \) 是设计矩阵，包含 \( p \) 个预测变量（如 SNP）。
- \( X_f \) 是固定效应的设计矩阵。
- \( \beta \) 和 \( \beta_f \) 是对应的回归系数。
- \( \sigma^2 \) 是残差方差。
- \( \sigma_j^2 \) 是第 \( j \) 个变量的效应方差。
- \( \delta_j \) 是指示变量，表示第 \( j \) 个变量是否被选入模型。
- \( \pi \) 是所有 \( \delta_j \) 的共同先验概率。
- \( a, b \) 是 Beta 分布的超参数。


## 1.2 稀疏性假设
BayesB 的核心假设之一是 **SNP 标记效应的稀疏性**，即大多数 SNP 的效应 \( \beta_j \) 为 0，只有一小部分 SNP 标记对表型具有显著影响。假设每个 SNP 标记效应 \( \beta_j \) 服从以下分布：
\[
\beta_j \sim \begin{cases} 
0, & \text{with probability } 1 - \pi \\
\mathcal{N}(0, \sigma_j^2), & \text{with probability } \pi
\end{cases}
\]
其中 \( \pi \) 是控制 SNP 标记效应非零概率的超参数。通过该稀疏性假设，BayesB 可以有效处理大量无关 SNP 数据。

## 1.3 先验分布
在 BayesB 中，假设 SNP 效应 \( \beta_j \) 和它的方差 \( \sigma_j^2 \) 具有以下先验分布：
1. **SNP 效应方差** \( \sigma_j^2 \) 服从逆卡方分布：
\[
\sigma_j^2 \sim \text{Inv-}\chi^2(\nu_j, S_j)
\]
2. **残差方差** \( \sigma_e^2 \) 也服从逆卡方分布：
\[
\sigma_e^2 \sim \text{Inv-}\chi^2(\nu_e, S_e)
\]

# 2. 参数的后验分布

## 2.1 y 的似然函数
在给定参数 \( \boldsymbol{\beta_f} \)、\( \boldsymbol{\beta} \) 和残差方差 \( \sigma_e^2 \)、SNP效应方差 \( \sigma_j^2 \) 的情况下，表型 \( \mathbf{y} \) 的似然函数为：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2, D) \propto (\sigma_e^2)^{-n/2} \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right)
\]
这个似然函数表示给定参数下观测到数据 \( \mathbf{y} \) 的概率。其中\(D\)是SNP效应方差矩阵。

## 2.2 Bayes 基础原理
根据贝叶斯定理，后验分布与似然函数和先验分布成正比：
\[
p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2, \mathbf{D} | \mathbf{y}) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f}) \cdot p(\boldsymbol{\beta} | \mathbf{D}) \cdot p(\mathbf{D}) \cdot p(\sigma_e^2)
\]
其中，\( p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \) 是似然函数，\( p(\boldsymbol{\beta} | \mathbf{D}) \) 是 SNP 效应的先验分布，\( p(\mathbf{D}) \) 是 SNP 方差的先验分布，\( p(\sigma_e^2) \) 是残差方差的先验分布。
由于此后验分布的非常复杂，通常需要使用 MCMC 方法进行采样求解。

# 3. 求解方式

在 **Gibbs 采样** 过程中，我们通过逐步从各个参数的 **条件后验分布** 中抽样，以更新模型中的参数。对于 BayesB 模型，通常需要以下几个参数的条件后验分布：

## 3.1. **SNP 标记效应 \( \beta_j \) 的条件后验分布**
SNP 效应 \( \beta_j \) 是最核心的参数之一。给定当前的其他参数值，包括固定效应 \( \boldsymbol{\beta_f} \)、SNP 方差 \( \sigma_j^2 \)，以及残差方差 \( \sigma_e^2 \)，\( \beta_j \) 的条件后验分布为：
\[
p(\beta_j | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2, \sigma_j^2) \sim \mathcal{N}(\mu_{\beta_j}, \sigma_{\beta_j}^2)
\]
其中：
- 均值 \( \mu_{\beta_j} \) 和方差 \( \sigma_{\beta_j}^2 \) 依赖于当前参数 \( \sigma_e^2 \) 和其他 SNP 标记效应。

具体而言，\( \beta_j \) 的更新取决于它是被稀疏假设（即 \( \beta_j = 0 \) 的假设）排除还是被认为具有显著效应。

## 3. 2. **SNP 标记效应方差 \( \sigma_j^2 \) 的条件后验分布**
对于每个非零的 \( \beta_j \)，其方差 \( \sigma_j^2 \) 的条件后验分布服从 **逆卡方分布**：
\[
p(\sigma_j^2 | \beta_j) \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]
这个分布反映了标记效应方差的更新，基于当前 \( \beta_j^2 \) 和先验参数 \( \nu_j \) 及 \( S_j \)。

## 3. 3. **残差方差 \( \sigma_e^2 \) 的条件后验分布**
残差方差 \( \sigma_e^2 \) 的更新基于表型数据 \( \mathbf{y} \)、固定效应 \( \boldsymbol{\beta_f} \) 和 SNP 效应 \( \boldsymbol{\beta} \)。它的条件后验分布也服从 **逆卡方分布**：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]
其中，\( SS_{\text{residual}} \) 是残差平方和。

## 3. 4. **固定效应 \( \boldsymbol{\beta_f} \) 的条件后验分布**
固定效应 \( \boldsymbol{\beta_f} \) 的条件后验分布通常服从正态分布，基于当前 SNP 效应 \( \boldsymbol{\beta} \) 和残差方差 \( \sigma_e^2 \) 的更新：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \boldsymbol{\Sigma_{\beta_f}})
\]
这里的均值和协方差矩阵通过固定效应矩阵 \( \mathbf{X_f} \) 和表型数据计算得出。

## 3. 5. **稀疏性参数 \( \pi \) 的更新（可选）**
在某些 BayesB 实现中，稀疏性参数 \( \pi \)（表示 SNP 标记效应为非零的概率）也可以进行更新。如果 \( \pi \) 被视为未知超参数，则可以通过贝叶斯推断对其进行采样。通常 \( \pi \) 服从 Beta 分布：
\[
\pi | \mathbf{\beta} \sim \text{Beta}(a + k, b + p - k)
\]
其中 \( k \) 是非零效应的 SNP 个数，\( p \) 是总的 SNP 数量，\( a \) 和 \( b \) 是 Beta 分布的超参数。

## 3.6 总结

在 **Gibbs 采样** 中，需要知道以下参数的条件后验分布：
1. **SNP 效应** \( \beta_j \) 的条件后验分布（正态分布）。
2. **SNP 效应方差** \( \sigma_j^2 \) 的条件后验分布（逆卡方分布）。
3. **残差方差** \( \sigma_e^2 \) 的条件后验分布（逆卡方分布）。
4. **固定效应** \( \boldsymbol{\beta_f} \) 的条件后验分布（正态分布）。
5. （可选）**稀疏性参数** \( \pi \) 的条件后验分布（Beta 分布）。

这些条件后验分布允许我们使用 **Gibbs 采样** 来逐步更新模型中的每个参数。

# 4. 各个参数条件后验分布推导
## 4.1 **SNP 效应 \( \beta_j \) 的条件后验分布**
### 4.1.1. 问题背景

在 **BayesB** 模型中，我们希望推导出每个 SNP 效应 \( \beta_j \) 的后验分布。对于每个 \( \beta_j \)，它可能为零（即该标记没有显著效应），或者为非零且服从正态分布。稀疏先验假设 \( \beta_j \) 有概率 \( \pi \) 是非零，概率 \( 1 - \pi \) 是零。因此我们要基于表型数据 \( \mathbf{y} \) 和其他参数，推导出 \( \beta_j \) 的条件后验分布。

### 4.1.2. 先验分布

对于每个 SNP 标记效应 \( \beta_j \)，其先验分布包含两部分：
\[
\beta_j \sim \begin{cases} 
0, & \text{with probability } 1 - \pi \\
\mathcal{N}(0, \sigma_j^2), & \text{with probability } \pi
\end{cases}
\]
这个分布表示稀疏性假设：大部分 \( \beta_j \) 为零，只有一部分标记具有显著效应。

因此我们引入一个二值指示变量 \( \delta_j \)，其服从 **Bernoulli 分布**：
\[
\delta_j \sim \text{Bernoulli}(\pi)
\]
当 \( \delta_j = 0 \) 时，\( \beta_j = 0 \)；当 \( \delta_j = 1 \) 时，\( \beta_j \) 服从正态分布 \( \mathcal{N}(0, \sigma_j^2) \)。

### 4.1.3. 似然函数

我们从 **似然函数** 出发，模型为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
给定误差 \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \)，我们可以写出表型 \( \mathbf{y} \) 的似然函数：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right)
\]
我们感兴趣的是单个 \( \beta_j \) 的后验分布，因此我们将模型重写为：
\[
\mathbf{y} = \mathbf{X}_{-j} \boldsymbol{\beta}_{-j} + \mathbf{X_j} \beta_j + \mathbf{e}
\]
其中 \( \mathbf{X_j} \) 是与 \( \beta_j \) 对应的 SNP 列向量，\( \mathbf{X}_{-j} \) 是除 \( \mathbf{X_j} \) 之外的 SNP 矩阵。

调整残差为 \( \mathbf{y}_j = \mathbf{y} - \mathbf{X}_{-j} \boldsymbol{\beta}_{-j} \)，则似然函数可以简化为：
\[
p(\mathbf{y} | \beta_j, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y}_j - \mathbf{X_j} \beta_j)^\top (\mathbf{y}_j - \mathbf{X_j} \beta_j) \right)
\]

### 4.1.4. 后验分布

根据 **贝叶斯定理**，我们有：
\[
p(\beta_j | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2, \sigma_j^2) \propto p(\mathbf{y} | \beta_j, \sigma_e^2) \cdot p(\beta_j | \sigma_j^2, \delta_j)
\]
对于 \( \delta_j = 1 \) 的情况，先验分布为：
\[
p(\beta_j | \sigma_j^2, \delta_j = 1) \propto \exp \left( -\frac{\beta_j^2}{2\sigma_j^2} \right)
\]
将似然函数和先验分布结合，后验分布为：
\[
p(\beta_j | \mathbf{y}, \sigma_j^2, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y}_j - \mathbf{X_j} \beta_j)^\top (\mathbf{y}_j - \mathbf{X_j} \beta_j) - \frac{\beta_j^2}{2\sigma_j^2} \right)
\]

### 4.1.5. 完整后验分布推导

展开并整理上述式子：
\[
p(\beta_j | \mathbf{y}, \sigma_j^2, \sigma_e^2) \propto \exp \left( -\frac{1}{2} \left( \beta_j^2 \left( \frac{\mathbf{X_j}^\top \mathbf{X_j}}{\sigma_e^2} + \frac{1}{\sigma_j^2} \right) - 2 \beta_j \frac{\mathbf{y}_j^\top \mathbf{X_j}}{\sigma_e^2} \right) \right)
\]
这是一个关于 \( \beta_j \) 的二次型。通过配平方，可以识别出后验分布为正态分布：
\[
p(\beta_j | \mathbf{y}, \sigma_j^2, \sigma_e^2) \sim \mathcal{N}(\mu_{\beta_j}, \sigma_{\beta_j}^2)
\]
其中：
- **条件均值**：
\[
\mu_{\beta_j} = \frac{\mathbf{y}_j^\top \mathbf{X_j}}{\sigma_e^2 \left( \frac{\mathbf{X_j}^\top \mathbf{X_j}}{\sigma_e^2} + \frac{1}{\sigma_j^2} \right)}
\]
- **条件方差**：
\[
\sigma_{\beta_j}^2 = \left( \frac{\mathbf{X_j}^\top \mathbf{X_j}}{\sigma_e^2} + \frac{1}{\sigma_j^2} \right)^{-1}
\]

### 4.1.6. 稀疏先验的影响

稀疏先验通过引入 \( \delta_j \) 来决定 \( \beta_j \) 是否为零。完整的后验分布可以写为：
\[
p(\beta_j | \mathbf{y}, \sigma_j^2, \sigma_e^2, \pi) = (1 - \pi) \delta(\beta_j = 0) + \pi \mathcal{N}(\mu_{\beta_j}, \sigma_{\beta_j}^2)
\]

### 4.1.7. 总结

我们通过似然函数和先验分布，推导出了 \( \beta_j \) 的后验分布。当 \( \delta_j = 1 \) 时，\( \beta_j \) 服从正态分布，其条件均值和方差如上所示；当 \( \delta_j = 0 \) 时，\( \beta_j = 0 \)。

## 4.2 **SNP 效应方差 \( \sigma_j^2 \) 的条件后验分布**
在 **BayesB** 模型中，每个 SNP 效应 \( \beta_j \) 对应一个方差 \( \sigma_j^2 \)，表示该 SNP 效应的变异性。假设 \( \sigma_j^2 \) 服从逆卡方分布，我们将推导其条件后验分布。
**\( \sigma_j^2 \)** 的条件后验分布 **不需要** 考虑稀疏先验。稀疏先验 \( \delta_j \) 只影响 **SNP 效应 \( \beta_j \)** 的更新过程，而不会影响 \( \sigma_j^2 \) 的更新。原因是稀疏先验的作用是在于决定 \( \beta_j \) 是否为零，而 \( \sigma_j^2 \) 的后验分布仅仅依赖于非零的 \( \beta_j \) 和相应的先验信息。因此，**\( \sigma_j^2 \)** 的后验分布如前面所推导的那样，不需要引入稀疏性假设。

### 4.2.1. 模型和先验假设

#### 4.2.1.1 模型背景

我们在 **BayesB** 中假设每个 SNP 效应 \( \beta_j \) 服从正态分布（当 \( \delta_j = 1 \)）：
\[
\beta_j | \sigma_j^2 \sim \mathcal{N}(0, \sigma_j^2)
\]
方差 \( \sigma_j^2 \) 则反映了 SNP 效应的变异性。

#### 4.2.1.2 先验分布

在贝叶斯推断中，\( \sigma_j^2 \) 通常被认为服从 **逆卡方分布**：
\[
\sigma_j^2 \sim \text{Inv-}\chi^2(\nu_j, S_j)
\]
其中 \( \nu_j \) 是自由度，\( S_j \) 是比例参数。这个先验反映了我们对 SNP 方差的先验信念。

### 4.2.2. 条件后验分布的推导

#### 4.2.2.1 目标

我们需要推导 \( \sigma_j^2 \) 的条件后验分布，即：
\[
p(\sigma_j^2 | \beta_j)
\]
这是在已知当前的 \( \beta_j \) 值后，更新 SNP 方差 \( \sigma_j^2 \) 的分布。

#### 4.2.2.2 使用贝叶斯定理

根据贝叶斯定理：
\[
p(\sigma_j^2 | \beta_j) \propto p(\beta_j | \sigma_j^2) \cdot p(\sigma_j^2)
\]
我们知道：
1. **似然函数** \( p(\beta_j | \sigma_j^2) \) 表示 \( \beta_j \) 在给定 \( \sigma_j^2 \) 下的概率分布。由于 \( \beta_j | \sigma_j^2 \sim \mathcal{N}(0, \sigma_j^2) \)，其似然函数为：
   \[
   p(\beta_j | \sigma_j^2) \propto \frac{1}{\sqrt{\sigma_j^2}} \exp \left( -\frac{\beta_j^2}{2\sigma_j^2} \right)
   \]
2. **先验分布** \( p(\sigma_j^2) \) 服从逆卡方分布：
   \[
   p(\sigma_j^2) \propto (\sigma_j^2)^{-(\nu_j/2 + 1)} \exp \left( -\frac{S_j}{2\sigma_j^2} \right)
   \]

#### 4.2.2.3 结合似然函数和先验分布

将似然函数和先验分布结合，得到：
\[
p(\sigma_j^2 | \beta_j) \propto (\sigma_j^2)^{-(\nu_j/2 + 1 + 1/2)} \exp \left( -\frac{1}{2\sigma_j^2} (\beta_j^2 + S_j) \right)
\]
其中，\( (\nu_j/2 + 1 + 1/2) \) 是结合了先验自由度和似然的参数。

#### 4.2.2.4 确定后验分布形式

我们看到这个表达式与 **逆卡方分布（Inverse-Chi-Squared Distribution）** 的形式一致。逆卡方分布的概率密度函数通常为：
\[
p(\sigma_j^2) \propto (\sigma_j^2)^{-(\nu'/2 + 1)} \exp \left( -\frac{S'}{2\sigma_j^2} \right)
\]
与标准形式对比，我们得出：

- 后验自由度为： \( \nu_j' = \nu_j + 1 \)
- 后验比例参数为： \( S_j' = S_j + \beta_j^2 \)

因此，\( \sigma_j^2 \) 的条件后验分布为：
\[
\sigma_j^2 | \beta_j \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]

### 4.2.3. 结果总结

我们推导出了 **SNP 效应方差 \( \sigma_j^2 \)** 的条件后验分布，其形式是逆卡方分布：
\[
\sigma_j^2 | \beta_j \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]
这个分布结合了先验信息（\( \nu_j \) 和 \( S_j \)）与当前 \( \beta_j \) 的值来更新方差 \( \sigma_j^2 \) 的分布。

## 4.3 **残差方差** \( \sigma_e^2 \) 的条件后验分布
在推导 **残差方差 \( \sigma_e^2 \)** 的条件后验分布时，稀疏先验并不会影响残差方差的更新，因此我们不需要考虑稀疏性假设。这是因为残差方差 \( \sigma_e^2 \) 的后验分布主要依赖于整个模型的残差平方和，而不是单个 SNP 效应是否为零。因此，我们按照标准的方式推导 \( \sigma_e^2 \) 的条件后验分布。

### 4.3.1. 模型背景

我们从 **BayesB** 模型出发，假设表型 \( \mathbf{y} \) 的生成过程为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
其中：
- \( \mathbf{X_f} \boldsymbol{\beta_f} \) 是固定效应部分。
- \( \mathbf{X} \boldsymbol{\beta} \) 是 SNP 效应部分。
- \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \) 是误差项，假设服从正态分布，且方差为 \( \sigma_e^2 \)。

我们希望推导 \( \sigma_e^2 \) 的条件后验分布。

### 4.3.2. 先验分布

假设 \( \sigma_e^2 \) 服从 **逆卡方分布**，其先验分布为：
\[
\sigma_e^2 \sim \text{Inv-}\chi^2(\nu_e, S_e)
\]
其中：
- \( \nu_e \) 是先验自由度。
- \( S_e \) 是先验比例参数。

### 4.3.3. 似然函数

我们知道残差项 \( \mathbf{e} = \mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta} \)，其分布为：
\[
\mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I})
\]
因此，给定 \( \boldsymbol{\beta_f} \)、\( \boldsymbol{\beta} \) 和 \( \sigma_e^2 \)，表型 \( \mathbf{y} \) 的似然函数为：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto (\sigma_e^2)^{-n/2} \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right)
\]

### 4.3.4. 条件后验分布的推导

根据贝叶斯定理，残差方差 \( \sigma_e^2 \) 的条件后验分布与似然函数和先验分布成正比：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\sigma_e^2)
\]
我们将似然函数和先验分布结合起来，得到：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto (\sigma_e^2)^{-n/2} \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot (\sigma_e^2)^{-(\nu_e/2 + 1)} \exp \left( -\frac{S_e}{2\sigma_e^2} \right)
\]

### 4.3.5. 残差平方和的简化

我们将残差 \( \mathbf{e} = \mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta} \) 的平方和表示为：
\[
SS_{\text{residual}} = (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})
\]
这样，条件后验分布可以写为：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto (\sigma_e^2)^{-(n/2 + \nu_e/2 + 1)} \exp \left( -\frac{1}{2\sigma_e^2} (SS_{\text{residual}} + S_e) \right)
\]

### 4.3.6. 识别后验分布形式

这是 **逆卡方分布** 的标准形式。根据逆卡方分布的形式：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]

### 4.3.7. 结果总结

我们推导出残差方差 \( \sigma_e^2 \) 的条件后验分布为：
\[
\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta} \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]
这个分布结合了残差平方和 \( SS_{\text{residual}} \) 和先验参数 \( S_e \) 及自由度 \( \nu_e \)，从而更新 \( \sigma_e^2 \) 的后验分布。

稀疏先验不影响残差方差的更新，因为它仅决定 SNP 效应 \( \beta_j \) 是否为零，而不影响整个模型的残差平方和。
## 4.4 **固定效应** \( \boldsymbol{\beta_f} \) 的条件后验分布
在推导 **固定效应 \( \beta_f \)** 的条件后验分布时，不需要考虑稀疏先验，因为稀疏先验主要针对 **SNP 效应 \( \beta_j \)**，用于表示标记效应的稀疏性。而固定效应通常与个体的共变量（如年龄、性别等）相关，不具备稀疏性假设。因此，我们将按照标准的贝叶斯推导过程，推导 \( \beta_f \) 的条件后验分布。

### 4.4.1. 模型背景

我们使用的 **BayesB** 模型可以写为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
其中：
- \( \mathbf{X_f} \) 是固定效应矩阵（大小为 \( n \times q \)），对应的固定效应参数为 \( \boldsymbol{\beta_f} \)。
- \( \mathbf{X} \) 是 SNP 矩阵，\( \boldsymbol{\beta} \) 是 SNP 效应。
- \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \) 是残差项，方差为 \( \sigma_e^2 \)。

我们希望推导固定效应 \( \boldsymbol{\beta_f} \) 的条件后验分布。

### 4.4.2. 先验分布

假设固定效应 \( \boldsymbol{\beta_f} \) 服从正态先验分布：
\[
\boldsymbol{\beta_f} \sim \mathcal{N}(\mathbf{0}, \sigma_f^2 \mathbf{I})
\]
其中 \( \sigma_f^2 \) 是固定效应的先验方差。

### 4.4.3. 似然函数

表型 \( \mathbf{y} \) 的似然函数为：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right)
\]
将 \( \mathbf{y} \) 写成关于 \( \boldsymbol{\beta_f} \) 的形式，定义调整后的残差 \( \mathbf{r} = \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \)，则似然函数可以简化为：
\[
p(\mathbf{r} | \boldsymbol{\beta_f}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{r} - \mathbf{X_f} \boldsymbol{\beta_f})^\top (\mathbf{r} - \mathbf{X_f} \boldsymbol{\beta_f}) \right)
\]

### 4.4.4. 条件后验分布的推导

根据贝叶斯定理，固定效应 \( \boldsymbol{\beta_f} \) 的条件后验分布与似然函数和先验分布的乘积成正比：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto p(\mathbf{r} | \boldsymbol{\beta_f}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f})
\]
将似然函数和先验分布代入后：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{r} - \mathbf{X_f} \boldsymbol{\beta_f})^\top (\mathbf{r} - \mathbf{X_f} \boldsymbol{\beta_f}) - \frac{1}{2\sigma_f^2} \boldsymbol{\beta_f}^\top \boldsymbol{\beta_f} \right)
\]

### 4.4.5. 完整后验分布推导

我们将这个式子展开并整理：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2} \left[ \boldsymbol{\beta_f}^\top \left( \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I} \right) \boldsymbol{\beta_f} - 2 \frac{1}{\sigma_e^2} \mathbf{r}^\top \mathbf{X_f} \boldsymbol{\beta_f} \right] \right)
\]

通过配平方和整理后，这个形式可以识别为正态分布：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \boldsymbol{\Sigma_{\beta_f}})
\]
其中：
- **均值**：
\[
\boldsymbol{\mu_{\beta_f}} = \boldsymbol{\Sigma_{\beta_f}} \cdot \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{r}
\]
- **协方差矩阵**：
\[
\boldsymbol{\Sigma_{\beta_f}} = \left( \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I} \right)^{-1}
\]

### 4.4.6. 结果总结

我们推导得出固定效应 \( \boldsymbol{\beta_f} \) 的条件后验分布为：
\[
\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2 \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \boldsymbol{\Sigma_{\beta_f}})
\]
其中，均值 \( \boldsymbol{\mu_{\beta_f}} \) 由调整后的残差和固定效应矩阵计算，协方差矩阵 \( \boldsymbol{\Sigma_{\beta_f}} \) 则考虑了观测误差和先验方差信息。

**结论**：稀疏先验不影响固定效应 \( \boldsymbol{\beta_f} \) 的更新，因为它主要适用于 **SNP 效应 \( \beta_j \)**，而不是固定效应。

## 4.5 **稀疏性参数** \( \pi \) 的条件后验分布
在 **BayesB** 模型中，**稀疏性参数 \( \pi \)** 表示每个 SNP 标记效应 \( \beta_j \) 非零的概率。我们希望推导 \( \pi \) 的条件后验分布，以更新稀疏性参数。由于 \( \pi \) 控制的是 SNP 效应的稀疏性，因此它直接与稀疏先验相关，我们需要考虑稀疏性假设。

### 4.5.1. 模型和先验假设

在 **BayesB** 中，假设每个 SNP 效应 \( \beta_j \) 服从如下的混合分布：
\[
\beta_j \sim \begin{cases} 
0, & \text{with probability } 1 - \pi \\
\mathcal{N}(0, \sigma_j^2), & \text{with probability } \pi
\end{cases}
\]
这里的 \( \pi \) 是一个超参数，表示 SNP 效应 \( \beta_j \) 非零的概率。

#### 4.5.1.1 先验分布

我们通常假设 \( \pi \) 服从 **Beta 分布** 作为先验：
\[
\pi \sim \text{Beta}(a, b)
\]
其中 \( a \) 和 \( b \) 是超参数，表示我们对稀疏性先验的初始信念。

### 4.5.2. 条件后验分布的推导

为了推导 \( \pi \) 的条件后验分布，我们需要利用贝叶斯定理，将似然函数和先验分布结合起来：
\[
p(\pi | \boldsymbol{\delta}) \propto p(\boldsymbol{\delta} | \pi) \cdot p(\pi)
\]
其中：
- \( \boldsymbol{\delta} = (\delta_1, \delta_2, \dots, \delta_p) \) 是指示变量的向量，表示每个 \( \beta_j \) 是否为零。
- \( p(\boldsymbol{\delta} | \pi) \) 是似然函数，描述给定稀疏性参数 \( \pi \) 时，指示变量 \( \delta_j \) 的分布。
- \( p(\pi) \) 是先验分布（Beta 分布）。

#### 4.5.2.1. 似然函数

每个 \( \delta_j \) 是一个 Bernoulli 变量，表示 \( \beta_j \) 是否非零，因此似然函数为：
\[
p(\boldsymbol{\delta} | \pi) = \prod_{j=1}^{p} \pi^{\delta_j} (1 - \pi)^{1 - \delta_j}
\]
这表示给定 \( \pi \) 后，\( \delta_j = 1 \) 的概率为 \( \pi \)，而 \( \delta_j = 0 \) 的概率为 \( 1 - \pi \)。

#### 4.5.2.2. 先验分布

稀疏性参数 \( \pi \) 的先验分布是 Beta 分布：
\[
p(\pi) = \frac{\pi^{a-1} (1 - \pi)^{b-1}}{B(a, b)}
\]
其中 \( B(a, b) \) 是 Beta 函数，\( a \) 和 \( b \) 是 Beta 分布的参数。

### 4.5.3. 结合似然函数和先验分布

我们将似然函数和先验分布结合，得到 \( \pi \) 的后验分布：
\[
p(\pi | \boldsymbol{\delta}) \propto \left( \prod_{j=1}^{p} \pi^{\delta_j} (1 - \pi)^{1 - \delta_j} \right) \cdot \pi^{a-1} (1 - \pi)^{b-1}
\]

展开并整理后：
\[
p(\pi | \boldsymbol{\delta}) \propto \pi^{\sum_{j=1}^{p} \delta_j + a - 1} (1 - \pi)^{p - \sum_{j=1}^{p} \delta_j + b - 1}
\]

### 4.5.4. 识别后验分布形式

通过整理我们发现，后验分布的形式仍然是一个 Beta 分布：
\[
p(\pi | \boldsymbol{\delta}) \sim \text{Beta}(a + k, b + p - k)
\]
其中：
- \( k = \sum_{j=1}^{p} \delta_j \)，即非零的 \( \beta_j \) 的个数。
- \( p \) 是 SNP 标记的总个数。

### 4.5.5. 结果总结

我们得到了稀疏性参数 \( \pi \) 的条件后验分布：
\[
p(\pi | \boldsymbol{\delta}) \sim \text{Beta}(a + k, b + p - k)
\]
其中 \( k \) 是非零的 SNP 效应数量，\( p - k \) 是零效应的 SNP 数量。

**结论**：由于 \( \pi \) 本身就是稀疏性参数的超参数，因此我们必须考虑稀疏先验。通过 Beta 分布作为先验，结合似然函数后，稀疏性参数 \( \pi \) 的条件后验分布仍然是一个 Beta 分布。


## 4.6. 条件后验分布 \( p(\delta_j = 1 | \cdot) \)

对于每个指示变量 \( \delta_j \)，其条件后验分布取决于其是否被选入模型：

\[
p(\delta_j = 1 | \beta_j, \pi, \sigma_j^2) \propto p(\beta_j | \delta_j = 1, \sigma_j^2) p(\delta_j = 1 | \pi)
\]

\[
p(\delta_j = 0 | \beta_j, \pi, \sigma_j^2) \propto p(\beta_j | \delta_j = 0) p(\delta_j = 0 | \pi)
\]

由于 \( p(\delta_j = 0 | \pi) = 1 - \pi \) 且 \( p(\delta_j = 1 | \pi) = \pi \)，并且：

\[
p(\beta_j | \delta_j = 1, \sigma_j^2) = \mathcal{N}(\beta_j | 0, \sigma_j^2)
\]

\[
p(\beta_j | \delta_j = 0) = \delta_0(\beta_j) \quad (\text{Dirac delta function at } \beta_j = 0)
\]

综合上述：

\[
p(\delta_j = 1 | \cdot) = \frac{p(\beta_j | \delta_j = 1, \sigma_j^2) p(\delta_j = 1 | \pi)}{p(\beta_j | \delta_j = 1, \sigma_j^2) p(\delta_j = 1 | \pi) + p(\beta_j | \delta_j = 0) p(\delta_j = 0 | \pi)}
\]
从 Bernoulli 分布中采样 \( \delta_j \)：
     \[
     \delta_j \sim \text{Bernoulli}\left(p(\delta_j = 1 | \cdot)\right)
     \]\\[\\]\\(\\)