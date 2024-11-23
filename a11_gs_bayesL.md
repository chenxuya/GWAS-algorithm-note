## 0. BayesL的数学模型结构

**BayesL**（Bayesian LASSO）是一种贝叶斯变量选择与估计方法，广泛应用于高维回归分析，尤其适用于遗传学中的单核苷酸多态性（SNP）效应估计。与传统的LASSO不同，BayesL通过贝叶斯框架引入先验分布，实现参数的不确定性量化和变量选择。

在本模型中，**所有SNPs均被假设具有非零效应**，但每个SNP效应的方差遵循指数分布，从而实现对效应大小的控制和稀疏性。这种方法不仅保留了LASSO的稀疏性优势，还提供了参数不确定性的全面描述。

此外，模型中还考虑了**固定效应**，即额外的协变量 \( \mathbf{X_f} \)，以控制可能的混杂因素或其他已知影响因子。

### 模型表示

假设我们有：

- 响应变量向量 \( \mathbf{y} \in \mathbb{R}^n \)
- SNPs的设计矩阵 \( \mathbf{X} \in \mathbb{R}^{n \times p} \)
- 固定效应的设计矩阵 \( \mathbf{X_f} \in \mathbb{R}^{n \times q} \)
- 回归系数向量 \( \boldsymbol{\beta_f} \in \mathbb{R}^q \)（对应固定效应）
- SNP效应回归系数向量 \( \boldsymbol{\beta} \in \mathbb{R}^p \)
- 残差向量 \( \mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma_e^2 \mathbf{I}) \)

模型结构如下：

\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]

### 参数的先验分布

1. **固定效应回归系数 \( \boldsymbol{\beta_f} \) 的先验分布**：
   \[
   \boldsymbol{\beta_f} \sim \mathcal{N}(\mathbf{0}, \tau_f^2 \mathbf{I})
   \]
   - \( \tau_f^2 \) 是固定效应的先验方差，通常设定为较大的值以表示弱先验。

2. **SNP效应回归系数 \( \boldsymbol{\beta} \) 的先验分布**：
   \[
   \boldsymbol{\beta} | \boldsymbol{\lambda}^2 \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Lambda} \sigma_e^2)
   \]
   - \( \boldsymbol{\Lambda} = \text{diag}(\lambda_1^2, \lambda_2^2, \dots, \lambda_p^2) \) 是方差矩阵，其中 \( \lambda_j^2 \) 控制第 \( j \) 个SNP的效应方差。

3. **SNP效应方差 \( \lambda_j^2 \) 的先验分布**：
   \[
   \lambda_j^2 \sim \text{Exponential}(\theta) \quad \forall j = 1, 2, \dots, p
   \]
   - \( \theta \) 是指数分布的参数，通常设定为 \( \theta = 1 \) 或其他合适值，以反映对效应方差的先验信念。

4. **残差方差 \( \sigma_e^2 \) 的先验分布**：
   \[
   \sigma_e^2 \sim \text{Inverse-Gamma}(a_e, b_e)
   \]
   - \( a_e \) 和 \( b_e \) 是逆伽玛分布的超参数，通常设定为较小值以表示弱先验。

## 1. 先验分布假设

在BayesL模型中，各参数的先验分布假设如下：

1. **固定效应回归系数 \( \boldsymbol{\beta_f} \) 的先验分布**：
   \[
   \boldsymbol{\beta_f} \sim \mathcal{N}(\mathbf{0}, \tau_f^2 \mathbf{I})
   \]
   - **解释**：假设固定效应系数来自均值为零、方差为 \( \tau_f^2 \) 的多元正态分布。较大的 \( \tau_f^2 \) 表示对固定效应的弱先验，允许其具有较大的不确定性。

2. **SNP效应回归系数 \( \boldsymbol{\beta} \) 的先验分布**：
   \[
   \boldsymbol{\beta} | \boldsymbol{\lambda}^2 \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Lambda} \sigma_e^2)
   \]
   - **解释**：假设每个SNP效应 \( \beta_j \) 来自均值为零、方差为 \( \lambda_j^2 \sigma_e^2 \) 的正态分布。方差 \( \lambda_j^2 \) 控制每个SNP效应的大小，实现对效应的缩放和稀疏性。

3. **SNP效应方差 \( \lambda_j^2 \) 的先验分布**：
   \[
   \lambda_j^2 \sim \text{Exponential}(\theta) \quad \forall j = 1, 2, \dots, p
   \]
   - **解释**：假设每个SNP效应的方差 \( \lambda_j^2 \) 来自参数为 \( \theta \) 的指数分布。指数分布的记忆无关性质和单参数特性使其适合作为正则化先验，促进稀疏性。

4. **残差方差 \( \sigma_e^2 \) 的先验分布**：
   \[
   \sigma_e^2 \sim \text{Inverse-Gamma}(a_e, b_e)
   \]
   - **解释**：假设残差方差 \( \sigma_e^2 \) 来自参数为 \( a_e \) 和 \( b_e \) 的逆伽玛分布。较小的 \( a_e \) 和 \( b_e \) 表示对 \( \sigma_e^2 \) 的弱先验，允许其有较大的不确定性。

## 2. 推导y的似然函数，并根据贝叶斯定理列出后验分布

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

### 2.2 后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\lambda}^2, \sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \)

根据贝叶斯定理：
\[
p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\lambda}^2, \sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f}) \cdot p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) \cdot p(\boldsymbol{\lambda}^2) \cdot p(\sigma_e^2)
\]

具体来说：

- \( p(\mathbf{y} | \cdot) \) 是似然函数。
- \( p(\boldsymbol{\beta_f}) \) 是固定效应回归系数的先验分布。
- \( p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) \) 是SNP效应回归系数的先验分布。
- \( p(\boldsymbol{\lambda}^2) \) 是SNP效应方差的先验分布。
- \( p(\sigma_e^2) \) 是残差方差的先验分布。

由于各参数之间存在依赖关系，联合后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\lambda}^2, \sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \) 是高维且复杂的，难以直接从中采样或进行解析求解。因此，需要引入数值方法，如马尔可夫链蒙特卡洛（MCMC）中的Gibbs采样，来进行参数的迭代更新。

## 3. 说明分布的复杂性，引入MCMC中的Gibbs采样求解（简略说明）

后验分布 \( p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \boldsymbol{\lambda}^2, \sigma_e^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \) 是一个高维、耦合且复杂的分布。具体复杂性体现在：

1. **高维性**：参数空间包含固定效应回归系数 \( \boldsymbol{\beta_f} \)、SNP效应回归系数 \( \boldsymbol{\beta} \)、SNP效应方差 \( \boldsymbol{\lambda}^2 \)、以及残差方差 \( \sigma_e^2 \)，数量众多。
2. **参数耦合**：参数之间存在依赖关系，如 \( \boldsymbol{\beta} \) 和 \( \boldsymbol{\lambda}^2 \) 之间的依赖，\( \boldsymbol{\beta_f} \) 和 \( \sigma_e^2 \) 之间的依赖等。
3. **非标准分布**：虽然各参数的条件后验分布在某些情况下属于已知的分布族，但整体联合后验分布并不属于标准分布，无法直接进行采样。

### 引入Gibbs采样

**Gibbs采样** 是一种基于MCMC的采样方法，适用于从高维复杂后验分布中生成样本。其基本思想是依次从每个参数的条件后验分布中采样，逐步逼近联合后验分布。具体步骤如下：

1. **初始化**：为所有参数赋予初始值。
2. **迭代更新**：
   - 从 \( p(\boldsymbol{\beta_f} | \cdot) \) 中采样 \( \boldsymbol{\beta_f} \)。
   - 从 \( p(\boldsymbol{\beta} | \cdot) \) 中采样 \( \boldsymbol{\beta} \)。
   - 从 \( p(\boldsymbol{\lambda}^2 | \cdot) \) 中采样 \( \boldsymbol{\lambda}^2 \)。
   - 从 \( p(\sigma_e^2 | \cdot) \) 中采样 \( \sigma_e^2 \)。
3. **重复**：重复上述步骤，直到达到预设的迭代次数或满足收敛条件。
4. **汇总**：根据采样结果进行后验估计和参数推断。

由于Gibbs采样依赖于参数的条件后验分布，能够有效处理参数之间的依赖关系，并逐步探索高维后验空间。

## 4. 分别推导各参数的条件后验分布（详尽的数学推导过程）

在本节中，将详细推导BayesL模型中各参数的条件后验分布，包括固定效应回归系数 \( \boldsymbol{\beta_f} \)、SNP效应回归系数 \( \boldsymbol{\beta} \)、SNP效应方差 \( \boldsymbol{\lambda}^2 \)、以及残差方差 \( \sigma_e^2 \)。

### 4.1 固定效应回归系数 \( \boldsymbol{\beta_f} \) 的条件后验分布

#### 推导过程

条件后验分布为：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta}, \sigma_e^2) \propto p(\mathbf{y} | \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \cdot p(\boldsymbol{\beta_f})
\]

其中，
\[
p(\mathbf{y} | \cdot) = \mathcal{N}(\mathbf{y} | \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta}, \sigma_e^2 \mathbf{I})
\]
\[
p(\boldsymbol{\beta_f}) = \mathcal{N}(\boldsymbol{\beta_f} | \mathbf{0}, \tau_f^2 \mathbf{I})
\]

将其代入后：
\[
p(\boldsymbol{\beta_f} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot \exp\left( -\frac{1}{2\tau_f^2} \boldsymbol{\beta_f}^T \boldsymbol{\beta_f} \right)
\]

展开平方项：
\[
(\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) - 2 \boldsymbol{\beta_f}^T \mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) + \boldsymbol{\beta_f}^T \mathbf{X_f}^T \mathbf{X_f} \boldsymbol{\beta_f}
\]

忽略与 \( \boldsymbol{\beta_f} \) 无关的项，并整理得到：
\[
p(\boldsymbol{\beta_f} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} \boldsymbol{\beta_f}^T \mathbf{X_f}^T \mathbf{X_f} \boldsymbol{\beta_f} + \frac{1}{\sigma_e^2} \boldsymbol{\beta_f}^T \mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot \exp\left( -\frac{1}{2\tau_f^2} \boldsymbol{\beta_f}^T \boldsymbol{\beta_f} \right)
\]

合并二次项和一次项：
\[
p(\boldsymbol{\beta_f} | \cdot) \propto \exp\left( -\frac{1}{2} \boldsymbol{\beta_f}^T \left( \frac{\mathbf{X_f}^T \mathbf{X_f}}{\sigma_e^2} + \frac{1}{\tau_f^2} \mathbf{I} \right) \boldsymbol{\beta_f} + \boldsymbol{\beta_f}^T \frac{\mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})}{\sigma_e^2} \right)
\]

这是一个多元正态分布的指数形式，识别出协方差矩阵和均值向量：
\[
\mathbf{\Sigma_{\beta_f}} = \left( \frac{\mathbf{X_f}^T \mathbf{X_f}}{\sigma_e^2} + \frac{1}{\tau_f^2} \mathbf{I} \right)^{-1}
\]
\[
\boldsymbol{\mu_{\beta_f}} = \mathbf{\Sigma_{\beta_f}} \cdot \frac{\mathbf{X_f}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})}{\sigma_e^2}
\]

因此，
\[
\boldsymbol{\beta_f} | \cdot \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \mathbf{\Sigma_{\beta_f}})
\]

### 4.2 SNP效应回归系数 \( \boldsymbol{\beta} \) 的条件后验分布

#### 推导过程

条件后验分布为：
\[
p(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\lambda}^2, \sigma_e^2) \propto p(\mathbf{y} | \cdot) \cdot p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2)
\]

其中，
\[
p(\mathbf{y} | \cdot) = \mathcal{N}(\mathbf{y} | \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta}, \sigma_e^2 \mathbf{I})
\]
\[
p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) = \prod_{j=1}^p \mathcal{N}(\beta_j | 0, \lambda_j^2 \sigma_e^2)
\]

因此，
\[
p(\boldsymbol{\beta} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot \prod_{j=1}^p \exp\left( -\frac{1}{2\lambda_j^2 \sigma_e^2} \beta_j^2 \right)
\]

展开平方项：
\[
(\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) - 2 \boldsymbol{\beta}^T \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
\]

将其代入后：
\[
p(\boldsymbol{\beta} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} \left[ \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} - 2 \boldsymbol{\beta}^T \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) \right] \right) \cdot \prod_{j=1}^p \exp\left( -\frac{1}{2\lambda_j^2 \sigma_e^2} \beta_j^2 \right)
\]

结合所有 \( \beta_j \)：
\[
p(\boldsymbol{\beta} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} + \frac{1}{\sigma_e^2} \boldsymbol{\beta}^T \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) \right) \cdot \exp\left( -\frac{1}{2\sigma_e^2} \sum_{j=1}^p \frac{1}{\lambda_j^2} \beta_j^2 \right)
\]

重新组织二次项：
\[
p(\boldsymbol{\beta} | \cdot) \propto \exp\left( -\frac{1}{2\sigma_e^2} \boldsymbol{\beta}^T \left( \mathbf{X}^T \mathbf{X} + \mathbf{\Lambda}^{-1} \right) \boldsymbol{\beta} + \frac{1}{\sigma_e^2} \boldsymbol{\beta}^T \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) \right)
\]
其中，\( \mathbf{\Lambda} = \text{diag}(\lambda_1^2, \lambda_2^2, \dots, \lambda_p^2) \)。

这是一个多元正态分布的指数形式，识别出协方差矩阵和均值向量：
\[
\mathbf{\Sigma_{\beta}} = \sigma_e^2 \left( \mathbf{X}^T \mathbf{X} + \mathbf{\Lambda}^{-1} \right)^{-1}
\]
\[
\boldsymbol{\mu_{\beta}} = \mathbf{\Sigma_{\beta}} \cdot \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})
\]

因此，
\[
\boldsymbol{\beta} | \cdot \sim \mathcal{N}(\boldsymbol{\mu_{\beta}}, \mathbf{\Sigma_{\beta}})
\]

### 4.3 SNP效应方差 \( \boldsymbol{\lambda}^2 \) 的条件后验分布

#### 推导过程

条件后验分布为：
\[
p(\boldsymbol{\lambda}^2 | \mathbf{y}, \mathbf{X_f}, \mathbf{X}, \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) \cdot p(\boldsymbol{\lambda}^2)
\]

其中，
\[
p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) = \prod_{j=1}^p \mathcal{N}(\beta_j | 0, \lambda_j^2 \sigma_e^2)
\]
\[
p(\lambda_j^2) = \text{Exponential}(\theta) \quad \forall j = 1, 2, \dots, p
\]

因此，
\[
p(\lambda_j^2 | \cdot) \propto \mathcal{N}(\beta_j | 0, \lambda_j^2 \sigma_e^2) \cdot \text{Exponential}(\lambda_j^2 | \theta)
\]

展开正态分布和指数分布：
\[
\mathcal{N}(\beta_j | 0, \lambda_j^2 \sigma_e^2) = \frac{1}{\sqrt{2\pi \lambda_j^2 \sigma_e^2}} \exp\left( -\frac{\beta_j^2}{2\lambda_j^2 \sigma_e^2} \right)
\]
\[
\text{Exponential}(\lambda_j^2 | \theta) = \theta \exp(-\theta \lambda_j^2)
\]

因此，
\[
p(\lambda_j^2 | \cdot) \propto \frac{1}{\sqrt{\lambda_j^2}} \exp\left( -\frac{\beta_j^2}{2\lambda_j^2 \sigma_e^2} \right) \cdot \exp(-\theta \lambda_j^2)
\]

为了简化表达式，引入变量替换 \( \gamma_j = \frac{1}{\lambda_j^2} \)，则 \( \lambda_j^2 = \frac{1}{\gamma_j} \)，且 \( d\lambda_j^2 = -\frac{1}{\gamma_j^2} d\gamma_j \).

替换后：
\[
p(\gamma_j | \cdot) \propto \gamma_j^{1/2} \exp\left( -\frac{\beta_j^2 \gamma_j}{2 \sigma_e^2} - \frac{\theta}{\gamma_j} \right)
\]

这实际上是**广义逆高斯分布**（Generalized Inverse Gaussian, GIG）的形式：
\[
p(\gamma_j | \cdot) \sim \text{GIG}(3/2, \frac{\beta_j^2}{2\sigma_e^2}, \theta)
\]

因此，
\[
\lambda_j^2 | \cdot \sim \text{Inverse-Gaussian}\left( \sqrt{\frac{\theta \sigma_e^2}{\beta_j^2}}, \theta \right)
\]
或更准确地，根据广义逆高斯分布的性质。

### 4.4 残差方差 \( \sigma_e^2 \) 的条件后验分布

#### 推导过程

条件后验分布为：
\[
p(\sigma_e^2 | \cdot) \propto p(\mathbf{y} | \cdot) \cdot p(\boldsymbol{\beta_f}) \cdot p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) \cdot p(\boldsymbol{\lambda}^2) \cdot p(\sigma_e^2)
\]

由于 \( \sigma_e^2 \) 仅出现在似然函数和SNP效应回归系数的先验分布中，因此，可以将其简化为：
\[
p(\sigma_e^2 | \cdot) \propto p(\mathbf{y} | \cdot) \cdot p(\boldsymbol{\beta} | \boldsymbol{\lambda}^2, \sigma_e^2) \cdot p(\sigma_e^2)
\]

展开后验分布：
\[
p(\sigma_e^2 | \cdot) \propto \left( \sigma_e^{-n} \exp\left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \right) \cdot \prod_{j=1}^p \left( \sigma_e^{-1} \exp\left( -\frac{\beta_j^2}{2\lambda_j^2 \sigma_e^2} \right) \right) \cdot \sigma_e^{-2(a_e +1)} \exp\left( -\frac{b_e}{\sigma_e^2} \right)
\]

合并幂次和指数项：
\[
p(\sigma_e^2 | \cdot) \propto (\sigma_e^2)^{-\frac{n}{2}} \exp\left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) \cdot \prod_{j=1}^p \sigma_e^{-1} \exp\left( -\frac{\beta_j^2}{2\lambda_j^2 \sigma_e^2} \right) \cdot (\sigma_e^2)^{-a_e -1} \exp\left( -\frac{b_e}{\sigma_e^2} \right)
\]

简化得到：
\[
p(\sigma_e^2 | \cdot) \propto (\sigma_e^2)^{-\frac{n}{2} - \frac{p}{2} - a_e - 1} \exp\left( -\frac{1}{2\sigma_e^2} \left[ (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) + \sum_{j=1}^p \frac{\beta_j^2}{\lambda_j^2} + 2 b_e \right] \right)
\]

这符合**逆伽玛分布**的形式：
\[
\sigma_e^2 | \cdot \sim \text{Inverse-Gamma}\left(a_e + \frac{n + p}{2}, \ b_e + \frac{1}{2} \left[ (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) + \sum_{j=1}^p \frac{\beta_j^2}{\lambda_j^2} \right] \right)
\]

### 4.5 非零效应方差 \( \boldsymbol{\lambda}^2 \) 的条件后验分布

#### 推导过程

对于每个 SNP \( j = 1, 2, \dots, p \)，其条件后验分布为：
\[
p(\lambda_j^2 | \cdot) \sim \text{Inverse-Gaussian}\left( \sqrt{\frac{\theta \sigma_e^2}{\beta_j^2}}, \ \theta \right)
\]
或更准确地，通过广义逆高斯分布（Generalized Inverse Gaussian, GIG）表示。

由于BayesL的标准形式通常使用双指数（Laplace）分布来表示LASSO先验，通过尺度混合正态分布引入指数分布的尺度参数。具体地，\( \lambda_j^2 \) 的条件后验分布是一个广义逆高斯分布，适用于Gibbs采样。

**数学表达**：
\[
\lambda_j^2 | \cdot \sim \text{GIG}\left(\frac{1}{2}, \ \frac{\beta_j^2}{\sigma_e^2}, \ \theta \right)
\]

其中，GIG分布的参数分别为：
- \( \lambda = \frac{1}{2} \)（shape参数）
- \( \psi = \frac{\beta_j^2}{\sigma_e^2} \)（scale参数）
- \( \chi = \theta \)（shape参数）

实际应用中，可以通过数值方法或特定的采样算法来从GIG分布中采样。

## 5. 算法更新步骤

基于上述条件后验分布，BayesL模型的Gibbs采样算法更新步骤如下：

### 5.1 初始化

为所有参数赋予初始值：
- 固定效应回归系数 \( \boldsymbol{\beta_f}^{(0)} = \mathbf{0} \)
- SNP效应回归系数 \( \boldsymbol{\beta}^{(0)} = \mathbf{0} \)
- SNP效应方差 \( \boldsymbol{\lambda}^{2(0)} = \mathbf{1} \)（或其他合理值）
- 残差方差 \( \sigma_e^{2(0)} = 1 \)

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

**实现**：
```python
import numpy as np
from scipy.stats import multivariate_normal

# 计算协方差矩阵和均值向量
Sigma_beta_f_inv = (X_f.T @ X_f) / sigma_e2 + (1 / tau_f2) * np.eye(q)
Sigma_beta_f = np.linalg.inv(Sigma_beta_f_inv)
mu_beta_f = Sigma_beta_f @ (X_f.T @ (y - X @ beta)) / sigma_e2

# 采样
beta_f = multivariate_normal.rvs(mean=mu_beta_f, cov=Sigma_beta_f)
```

#### 步骤2：更新SNP效应回归系数 \( \boldsymbol{\beta} \)

根据条件后验分布：
\[
\boldsymbol{\beta} | \cdot \sim \mathcal{N}(\boldsymbol{\mu_{\beta}}, \mathbf{\Sigma_{\beta}})
\]
其中，
\[
\mathbf{\Sigma_{\beta}} = \sigma_e^2 \left( \mathbf{X}^T \mathbf{X} + \mathbf{\Lambda}^{-1} \right)^{-1}
\]
\[
\boldsymbol{\mu_{\beta}} = \mathbf{\Sigma_{\beta}} \cdot \mathbf{X}^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})
\]

**实现**：
```python
# 计算协方差矩阵和均值向量
Sigma_beta = sigma_e2 * np.linalg.inv(X.T @ X + np.linalg.inv(Lambda))
mu_beta = Sigma_beta @ (X.T @ (y - X_f @ beta_f))

# 采样
beta = multivariate_normal.rvs(mean=mu_beta, cov=Sigma_beta)
```

#### 步骤3：更新SNP效应方差 \( \boldsymbol{\lambda}^2 \)

根据条件后验分布，每个 \( \lambda_j^2 \) 独立采样：
\[
\lambda_j^2 | \cdot \sim \text{GIG}\left(\frac{1}{2}, \ \frac{\beta_j^2}{\sigma_e^2}, \ \theta \right)
\]

**实现**：
BayesLASSO中的GIG分布采样可以通过算法如 **Sibuya's method** 或使用现有的库函数。这里假设使用一个支持GIG采样的函数 `sample_gig`.

```python
from some_library import sample_gig  # 假设存在的GIG采样函数

for j in range(p):
    if beta[j] != 0:
        lambda2[j] = sample_gig(1/2, beta[j]**2 / sigma_e2, theta)
    else:
        lambda2[j] = np.inf  # 对应于β_j=0的情况
```

**注意**：实际实现中，需确保 \( \lambda_j^2 \) 的采样有效，并处理 \( \beta_j = 0 \) 的特殊情况。

#### 步骤4：更新残差方差 \( \sigma_e^2 \)

根据条件后验分布：
\[
\sigma_e^2 | \cdot \sim \text{Inverse-Gamma}\left(a_e + \frac{n + p}{2}, \ b_e + \frac{1}{2} \left[ (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) + \sum_{j=1}^p \frac{\beta_j^2}{\lambda_j^2} \right] \right)
\]

**实现**：
```python
from scipy.stats import invgamma

residual = y - X_f @ beta_f - X @ beta
shape_e = a_e + (n + p) / 2
scale_e = b_e + 0.5 * (residual.T @ residual + np.sum(beta**2 / lambda2))
sigma_e2 = invgamma.rvs(a=shape_e, scale=scale_e)
```

### 5.3 算法流程总结

1. **初始化**：
   - 设置固定效应回归系数 \( \boldsymbol{\beta_f}^{(0)} \)
   - 设置SNP效应回归系数 \( \boldsymbol{\beta}^{(0)} \)
   - 设置SNP效应方差 \( \boldsymbol{\lambda}^{2(0)} \)
   - 设置残差方差 \( \sigma_e^{2(0)} \)
   
2. **对于每一次迭代 \( t = 1, 2, \dots, T \)**，执行以下步骤：
   - **步骤1**：更新固定效应回归系数 \( \boldsymbol{\beta_f} \)
   - **步骤2**：更新SNP效应回归系数 \( \boldsymbol{\beta} \)
   - **步骤3**：更新SNP效应方差 \( \boldsymbol{\lambda}^2 \)
   - **步骤4**：更新残差方差 \( \sigma_e^2 \)
   - **步骤5**：存储采样结果
   
3. **重复**：重复迭代步骤，直到达到预设的迭代次数或满足收敛条件。

4. **结果汇总**：
   - 舍弃“burn-in”期的样本
   - 计算后验均值、标准差等统计量
   - 进行参数估计和变量选择

### 5.4 算法实现示例

以下是一个基于上述推导的BayesL模型的Gibbs采样算法的Python实现示例：

```python
import numpy as np
from scipy.stats import multivariate_normal, invgamma, geninvgauss
import matplotlib.pyplot as plt

def bayesL_gibbs_sampler(X_f, X, y, num_iterations, 
                         a_e=2.0, b_e=2.0, 
                         theta=1.0, 
                         tau_f2=1e6):
    """
    BayesL Model Gibbs Sampler using scipy.stats.geninvgauss

    Parameters:
    X_f: Fixed effects design matrix (n x q)
    X: SNP effects design matrix (n x p)
    y: Response vector (n,)
    num_iterations: Number of iterations
    a_e, b_e: Hyperparameters for sigma_e^2 prior
    theta: Parameter for the exponential prior on lambda_j^2
    tau_f2: Prior variance for fixed effects coefficients

    Returns:
    beta_f_samples: Samples of fixed effects coefficients (num_iterations x q)
    beta_samples: Samples of SNP effects coefficients (num_iterations x p)
    lambda2_samples: Samples of SNP effect variances (num_iterations x p)
    sigma_e2_samples: Samples of residual variance (num_iterations,)
    """
    n, p = X.shape
    q = X_f.shape[1]
    
    # Initialize parameters
    beta_f = np.zeros(q)
    beta = np.zeros(p)
    lambda2 = np.ones(p)
    sigma_e2 = 1.0
    
    # Store samples
    beta_f_samples = np.zeros((num_iterations, q))
    beta_samples = np.zeros((num_iterations, p))
    lambda2_samples = np.zeros((num_iterations, p))
    sigma_e2_samples = np.zeros(num_iterations)
    
    # Precompute X_f^T X_f
    X_fTX_f = X_f.T @ X_f
    
    for t in range(num_iterations):
        # Step 1: Update beta_f
        Sigma_beta_f_inv = (X_fTX_f / sigma_e2) + (1 / tau_f2) * np.eye(q)
        Sigma_beta_f = np.linalg.inv(Sigma_beta_f_inv)
        mu_beta_f = Sigma_beta_f @ (X_f.T @ (y - X @ beta)) / sigma_e2
        beta_f = multivariate_normal.rvs(mean=mu_beta_f, cov=Sigma_beta_f)
        
        # Step 2: Update beta
        Lambda_inv = np.diag(1 / lambda2)
        Sigma_beta_inv = (X.T @ X) / sigma_e2 + Lambda_inv
        Sigma_beta = np.linalg.inv(Sigma_beta_inv)
        mu_beta = Sigma_beta @ (X.T @ (y - X_f @ beta_f)) / sigma_e2
        beta = multivariate_normal.rvs(mean=mu_beta, cov=Sigma_beta)
        
        # Step 3: Update lambda_j^2
        for j in range(p):
            lambda_param = 0.5  # λ = 0.5
            beta_j_sq = beta[j] ** 2
            sigma_e2_val = sigma_e2

            if beta_j_sq != 0:
                chi = theta  # χ = θ
                psi = beta_j_sq / sigma_e2_val  # ψ = β_j^2 / σ_e^2

                # Compute b and scale parameters
                b_param = np.sqrt(chi * psi)
                scale_param = np.sqrt(psi / chi)

                # Sample from GIG distribution
                lambda2_j = geninvgauss.rvs(lambda_param, b_param, scale=scale_param)
                lambda2[j] = lambda2_j
            else:
                lambda2[j] = np.inf  # Assign a large number or handle appropriately
        
        # Step 4: Update sigma_e^2
        residual = y - X_f @ beta_f - X @ beta
        shape_e = a_e + n / 2
        scale_e = b_e + 0.5 * (residual.T @ residual)
        sigma_e2 = invgamma.rvs(a=shape_e, scale=scale_e)
        
        # Store samples
        beta_f_samples[t, :] = beta_f
        beta_samples[t, :] = beta
        lambda2_samples[t, :] = lambda2
        sigma_e2_samples[t] = sigma_e2
        
        # Optional: print progress
        if (t+1) % 1000 == 0 or t == 0:
            print(f"Iteration {t+1}/{num_iterations} completed.")
    
    return beta_f_samples, beta_samples, lambda2_samples, sigma_e2_samples
# 使用示例

# 假设数据已经准备好
# X_f: 固定效应的设计矩阵 (n x q)
# X: SNP 效应的设计矩阵 (n x p)
# y: 响应变量向量 (n,)

# 设置参数
num_iterations = 10000
a_e, b_e = 2.0, 2.0
theta = 1.0
tau_f2 = 1e6

# 运行 Gibbs 采样
# beta_f_samples, beta_samples, lambda2_samples, sigma_e2_samples = bayesL_gibbs_sampler(X_f, X, y, num_iterations, a_e, b_e, theta, tau_f2)

# 后续分析，如绘制参数的收敛诊断图
# import matplotlib.pyplot as plt
# plt.plot(beta_f_samples[:, 0])
# plt.title('Trace plot for beta_f[0]')
# plt.show()
```

### 注意事项

1. **数值稳定性**：
   - 在更新 \( \lambda_j^2 \) 时，处理 \( \beta_j = 0 \) 的特殊情况。
   - 在计算过程中，引入小的常数以避免除零错误。

2. **向量化优化**：
   - 当前实现中，SNP效应回归系数 \( \boldsymbol{\beta} \) 和效应方差 \( \boldsymbol{\lambda}^2 \) 的更新采用显式循环。对于大规模数据，可通过矩阵运算或批量更新来提高效率。

3. **收敛诊断**：
   - 使用收敛性诊断方法（如Gelman-Rubin诊断、参数轨迹图）确保采样过程已收敛。
   - 设置适当的“burn-in”期，舍弃前期未收敛的样本。

4. **后验样本处理**：
   - 舍弃“burn-in”期的样本
   - 计算后验均值、标准差和可信区间
   - 使用抽样间隔（thinning）减少样本自相关

5. **参数初始化**：
   - 不同的初始化策略可能影响采样收敛速度和结果质量。可尝试多种初始化方法，如随机初始化或基于先验知识的初始化。

## 总结

**BayesL**（Bayesian LASSO）模型通过引入贝叶斯框架，实现了对高维回归模型中SNP效应的稀疏性控制和参数不确定性的全面描述。通过将SNP效应方差设定为指数分布，并采用Gibbs采样进行参数迭代更新，BayesL能够有效地在大规模数据中进行变量选择和效应估计。

该模型不仅保留了LASSO的优势，还提供了更为丰富的统计推断工具，使其在遗传学和其他高维数据分析领域中具有广泛的应用潜力。在实际应用中，通过适当的参数设置、算法优化和收敛性诊断，BayesL能够实现高效、准确的变量选择和效应估计。\\[\\]\\(\\)