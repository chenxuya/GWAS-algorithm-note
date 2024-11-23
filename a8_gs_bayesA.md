让我们从头开始，基于之前讨论的**带有固定效应的BayesA模型**，推导其数学过程。我们将分步骤详细推导，解释每一步是如何推理出来的，并展示模型的贝叶斯推断过程。

### 1. 模型结构与假设

引入固定效应后，模型可以表示为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
其中：
- \( \mathbf{y} \) 是 \( n \times 1 \) 的表型向量，表示 \( n \) 个个体的表型。
- \( \mathbf{X_f} \) 是与固定效应相关的 \( n \times q \) 设计矩阵，表示 \( n \) 个个体的固定效应变量（如环境影响、性别等）。
- \( \boldsymbol{\beta_f} \) 是 \( q \times 1 \) 的固定效应系数向量。
- \( \mathbf{X} \) 是与SNP标记相关的 \( n \times p \) 基因型矩阵。
- \( \boldsymbol{\beta} \) 是 \( p \times 1 \) 的SNP标记效应向量。
- \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \) 是残差。

### 2. 似然函数

表型 \( \mathbf{y} \) 的似然函数基于正态分布假设。给定固定效应 \( \boldsymbol{\beta_f} \)、SNP 标记效应 \( \boldsymbol{\beta} \)、和残差方差 \( \sigma_e^2 \)，似然函数为：

\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2 \sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) \right)
\]

### 3. 先验分布

我们为每个参数设定先验分布：

#### 3.1 固定效应 \( \boldsymbol{\beta_f} \)

通常假设固定效应服从一个非信息先验或正态分布。我们假设：
\[
\boldsymbol{\beta_f} \sim \mathcal{N}(0, \sigma_f^2 \mathbf{I})
\]
其中 \( \sigma_f^2 \) 是固定效应的先验方差，通常设定为一个较大的值，以表示我们对固定效应的弱先验。

#### 3.2 SNP 标记效应 \( \boldsymbol{\beta} \)

如在 BayesA 模型中，我们假设每个SNP标记效应 \( \beta_j \) 服从正态分布，每个标记效应有不同的方差 \( \sigma_j^2 \)：

\[
\boldsymbol{\beta} \sim \mathcal{N}(0, \mathbf{D})
\]
其中，\( \mathbf{D} = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_p^2) \) 是一个对角矩阵，表示每个SNP标记效应的方差。

方差 \( \sigma_j^2 \) 假设服从逆卡方分布：
\[
\sigma_j^2 \sim \text{Inv-}\chi^2(\nu, S)
\]

#### 3.3 残差方差 \( \sigma_e^2 \)

残差 \( \mathbf{e} \) 的方差 \( \sigma_e^2 \) 也假设服从逆卡方分布：
\[
\sigma_e^2 \sim \text{Inv-}\chi^2(\nu_e, S_e)
\]

### 4. 后验分布

贝叶斯推断的目标是通过数据 \( \mathbf{y} \) 更新先验分布，得到各个参数的后验分布。根据贝叶斯定理，后验分布可以表示为：

\[
p(\boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2, \mathbf{D} | \mathbf{y}, \mathbf{X_f}, \mathbf{X}) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) p(\boldsymbol{\beta_f}) p(\boldsymbol{\beta} | \mathbf{D}) p(\mathbf{D}) p(\sigma_e^2)
\]

每一项对应的是：
- \( p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \)：似然函数。
- \( p(\boldsymbol{\beta_f}) \)：固定效应的先验分布。
- \( p(\boldsymbol{\beta} | \mathbf{D}) \)：SNP 标记效应的先验分布。
- \( p(\mathbf{D}) \)：方差的先验分布。
- \( p(\sigma_e^2) \)：残差方差的先验分布。


#### 4.1. \( p(\boldsymbol{\beta} | \mathbf{D}) \) 的解释

这个表达式表示**SNP 标记效应** \( \boldsymbol{\beta} \) 的先验分布，给定标记效应的方差矩阵 \( \mathbf{D} \)。在 **BayesA** 模型中，标记效应 \( \beta_j \) 被假设为服从一个零均值正态分布，并且每个标记的方差 \( \sigma_j^2 \) 是不同的。这意味着，对于每个标记 \( \beta_j \)，我们有：

\[
\beta_j \sim \mathcal{N}(0, \sigma_j^2)
\]

当我们将所有 \( p \) 个标记效应组合在一起时，它们的分布可以表示为多元正态分布：

\[
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \mathbf{D})
\]

其中 \( \mathbf{D} \) 是一个对角矩阵，表示每个标记的方差：
\[
\mathbf{D} = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_p^2)
\]

因此，**\( p(\boldsymbol{\beta} | \mathbf{D}) \)** 描述了每个 SNP 标记效应 \( \beta_j \) 的分布情况，假设我们知道其方差 \( \sigma_j^2 \)。在贝叶斯推断中，给定 \( \mathbf{D} \) 后，这些标记效应的分布就可以使用这个先验分布来描述。

**为何使用这个分布？**

- **正态分布的合理性**：对于大多数遗传效应，假设它们围绕零的正态分布是合理的。这符合量化性状的遗传学基础：大多数标记效应是较小的，只有少数标记具有较大的效应。
- **个别标记的方差不同**：每个标记 \( \beta_j \) 的效应可能不同，有的标记对性状的影响较大，方差 \( \sigma_j^2 \) 也较大，而有的标记效应很小，方差较小。这种灵活性使得模型可以适应复杂的遗传结构。

#### 4.2. \( p(\mathbf{D}) \) 的解释

\( p(\mathbf{D}) \) 表示方差矩阵 \( \mathbf{D} \) 的先验分布。因为 \( \mathbf{D} \) 是一个对角矩阵，其中每个对角元素是 \( \sigma_j^2 \)（即第 \( j \) 个标记效应的方差），所以我们实际上是在描述每个标记方差的先验分布。

在 **BayesA** 中，\( \sigma_j^2 \) 的先验分布假设为**逆卡方分布**（Inverse Chi-Square Distribution）：

\[
\sigma_j^2 \sim \text{Inv-}\chi^2(\nu, S)
\]

其中：
- \( \nu \) 是自由度参数。
- \( S \) 是比例参数，控制分布的规模。

**为何使用逆卡方分布？**

- **灵活性**：逆卡方分布是常用于方差参数的先验分布。它允许我们对方差进行一定程度的控制，但同时也足够灵活，适应方差大小的不同。
- **贝叶斯推断的简化**：使用逆卡方分布作为先验有助于推导后验分布的闭式解。在 Gibbs 采样过程中，我们可以直接从逆卡方分布中抽样更新 \( \sigma_j^2 \)。

**物理意义**：逆卡方分布允许我们假设大多数 SNP 标记效应的方差很小（即绝大多数标记对性状影响较小），但同时允许少数标记具有较大的效应方差。这符合许多复杂性状的遗传特性：大多数标记对性状的贡献很小，只有少数标记对性状有较大的影响。

#### 4.3. 结合这两者的推理

**\( p(\boldsymbol{\beta} | \mathbf{D}) \)** 和 **\( p(\mathbf{D}) \)** 共同描述了我们对标记效应及其方差的先验信念。在贝叶斯框架中，我们使用这些先验分布来反映在没有数据时对这些效应的预期，然后通过观测数据 \( \mathbf{y} \) 更新这些信念，得到后验分布。

贝叶斯推断过程中的核心在于：
- 我们假设标记效应 \( \boldsymbol{\beta} \) 是零均值的正态分布，但每个标记的方差不同。
- 方差 \( \sigma_j^2 \) 本身是随机的，我们通过逆卡方分布来描述其可能的取值范围。

在采样过程中，我们会交替地：
1. 从标记效应的条件后验分布中抽样（给定当前的方差 \( \sigma_j^2 \)）。
2. 从方差的条件后验分布中抽样（给定当前的标记效应 \( \beta_j \)）。

这种交替的采样过程能够逐步收敛到这些参数的后验分布，使我们能够有效估计标记效应和方差。

#### 4.4. 结论

- **\( p(\boldsymbol{\beta} | \mathbf{D}) \)**：描述的是给定方差矩阵 \( \mathbf{D} \) 后，SNP 标记效应 \( \boldsymbol{\beta} \) 的正态分布。
- **\( p(\mathbf{D}) \)**：描述的是方差矩阵 \( \mathbf{D} \) 的先验分布，每个方差 \( \sigma_j^2 \) 假设服从逆卡方分布。
- **这些先验的生物学意义**：它们共同反映了我们对遗传效应的先验假设——大多数 SNP 标记效应较小，但允许少数标记具有较大效应，而每个标记的方差是不同的。

### 5.Gibbs 采样
由于无法直接求解后验分布，通常使用 Gibbs 采样 来逐步抽取每个参数的条件后验分布。

### 6. \( \boldsymbol{\beta_f} \)的条件后验分布
#### 6.1. 问题背景简要回顾

我们有一个贝叶斯模型，目标是从 \( \boldsymbol{\beta_f} \) 的条件后验分布中推导出它的均值和协方差矩阵。给定的模型为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
其中：
- \( \mathbf{y} \) 是表型数据向量。
- \( \mathbf{X_f} \) 是固定效应矩阵，\( \boldsymbol{\beta_f} \) 是固定效应参数。
- \( \mathbf{X} \) 是基因型矩阵，\( \boldsymbol{\beta} \) 是基因效应参数。
- \( \mathbf{e} \) 是残差项，假设 \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \)。

我们假设固定效应 \( \boldsymbol{\beta_f} \) 服从正态先验分布 \( \boldsymbol{\beta_f} \sim \mathcal{N}(0, \sigma_f^2 \mathbf{I}) \)，即：
\[
p(\boldsymbol{\beta_f}) \propto \exp \left( -\frac{1}{2\sigma_f^2} \boldsymbol{\beta_f}^\top \boldsymbol{\beta_f} \right)
\]

#### 6.2. 条件后验分布

根据贝叶斯定理，条件后验分布 \( p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \) 与似然函数和先验分布成正比：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) p(\boldsymbol{\beta_f})
\]

**似然函数**为：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) \right)
\]

**先验分布**为：
\[
p(\boldsymbol{\beta_f}) \propto \exp \left( -\frac{1}{2\sigma_f^2} \boldsymbol{\beta_f}^\top \boldsymbol{\beta_f} \right)
\]

将两者结合，得到：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) - \frac{1}{2\sigma_f^2} \boldsymbol{\beta_f}^\top \boldsymbol{\beta_f} \right)
\]

#### 6.3. 展开并配平方

接下来，我们展开其中的平方项，特别是与 \( \boldsymbol{\beta_f} \) 相关的部分：
\[
(\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})
\]

展开得到：
\[
= (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) - 2(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top \mathbf{X_f} \boldsymbol{\beta_f} + \boldsymbol{\beta_f}^\top \mathbf{X_f}^\top \mathbf{X_f} \boldsymbol{\beta_f}
\]

代入后，结合先验分布的平方项：
\[
p(\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2} \left( \boldsymbol{\beta_f}^\top \left( \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I} \right) \boldsymbol{\beta_f} - 2 \frac{1}{\sigma_e^2} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top \mathbf{X_f} \boldsymbol{\beta_f} \right) \right)
\]

#### 6.4. 与标准正态分布对比

我们将这个结果与标准多元正态分布形式对比：
\[
p(\boldsymbol{\beta_f}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{\beta_f} - \boldsymbol{\mu_{\beta_f}})^\top \boldsymbol{\Sigma_{\beta_f}}^{-1} (\boldsymbol{\beta_f} - \boldsymbol{\mu_{\beta_f}}) \right)
\]
指数项展开后
$$
\begin{aligned}
p(\boldsymbol{\beta_f}) &= \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{\beta_f}^T \boldsymbol{\Sigma_{\beta_f}}^{-1} \boldsymbol{\beta_f} - \boldsymbol{\beta_f}^T \boldsymbol{\Sigma_{\beta_f}}^{-1}\boldsymbol{\mu_{\beta_f}} - \boldsymbol{\mu^T_{\beta_f}}\boldsymbol{\Sigma_{\beta_f}}^{-1}\boldsymbol{\beta_f} + \boldsymbol{\mu^T_{\beta_f}}\boldsymbol{\Sigma_{\beta_f}}^{-1}\boldsymbol{\mu_{\beta_f}}) \right) \\
&= \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{\beta_f}^T \boldsymbol{\Sigma_{\beta_f}}^{-1} \boldsymbol{\beta_f} - 2\boldsymbol{\mu^T_{\beta_f}}\boldsymbol{\Sigma_{\beta_f}}^{-1}\boldsymbol{\beta_f} + \boldsymbol{\mu^T_{\beta_f}}\boldsymbol{\Sigma_{\beta_f}}^{-1}\boldsymbol{\mu_{\beta_f}}) \right)
\end{aligned}
$$


通过配平方，我们可以整理出条件后验分布的均值和协方差矩阵的表达式。

##### 6.4.1. 配平方确定协方差矩阵

从展开式中的二次项 \( \boldsymbol{\beta_f}^\top \left( \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I} \right) \boldsymbol{\beta_f} \)，我们可以直接得到条件协方差矩阵的逆：
\[
\boldsymbol{\Sigma_{\beta_f}}^{-1} = \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I}
\]

因此，条件协方差矩阵为：
\[
\boldsymbol{\Sigma_{\beta_f}} = \left( \frac{1}{\sigma_e^2} \mathbf{X_f}^\top \mathbf{X_f} + \frac{1}{\sigma_f^2} \mathbf{I} \right)^{-1}
\]

##### 6.4.2. 配平方确定均值

再来看线性项 \( -2 \frac{1}{\sigma_e^2} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top \mathbf{X_f} \boldsymbol{\beta_f} \)，它对应于多元正态分布中的 \( \boldsymbol{\mu_{\beta_f}} \) 和 \( \boldsymbol{\Sigma_{\beta_f}}^{-1} \) 的线性部分。通过配平方，我们可以得出均值 \( \boldsymbol{\mu_{\beta_f}} \)：
\[
\boldsymbol{\mu_{\beta_f}} = \boldsymbol{\Sigma_{\beta_f}} \cdot \frac{1}{\sigma_e^2} \mathbf{X_f}^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\]

### 7. SNP 标记效应 \( \boldsymbol{\beta} \) 的条件后验分布

#### 7.1. 问题背景简要回顾

我们同样有贝叶斯模型，目标是从 **SNP 标记效应 \( \boldsymbol{\beta} \)** 的条件后验分布中推导出它的均值和协方差矩阵。模型为：
\[
\mathbf{y} = \mathbf{X_f} \boldsymbol{\beta_f} + \mathbf{X} \boldsymbol{\beta} + \mathbf{e}
\]
其中：
- \( \mathbf{y} \) 是表型数据向量。
- \( \mathbf{X} \) 是 SNP 基因型矩阵，\( \boldsymbol{\beta} \) 是 SNP 标记效应参数。
- \( \mathbf{X_f} \) 是固定效应矩阵，\( \boldsymbol{\beta_f} \) 是固定效应参数。
- \( \mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I}) \) 是残差。

我们假设 SNP 标记效应 \( \boldsymbol{\beta} \) 服从正态先验分布：
\[
\boldsymbol{\beta} \sim \mathcal{N}(0, \mathbf{D})
\]
其中 \( \mathbf{D} \) 是一个对角矩阵，表示每个标记的方差 \( \sigma_j^2 \)。

#### 7.2. 条件后验分布

根据贝叶斯定理，\( \boldsymbol{\beta} \) 的条件后验分布为：
\[
p(\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) p(\boldsymbol{\beta})
\]

**似然函数**为：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) \right)
\]

**先验分布**为：
\[
p(\boldsymbol{\beta}) \propto \exp \left( -\frac{1}{2} \boldsymbol{\beta}^\top \mathbf{D}^{-1} \boldsymbol{\beta} \right)
\]

将两者结合，得到：
\[
p(\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2) \propto \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) - \frac{1}{2} \boldsymbol{\beta}^\top \mathbf{D}^{-1} \boldsymbol{\beta} \right)
\]

#### 7.3. 展开并配平方

接下来，我们展开其中的平方项，特别是与 \( \boldsymbol{\beta} \) 相关的部分：
\[
(\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})
\]

展开得到：
\[
= (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f}) - 2(\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})^\top \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^\top \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta}
\]

代入后，结合先验分布中的平方项：
\[
p(\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2) \propto \exp \left( -\frac{1}{2} \left( \boldsymbol{\beta}^\top \left( \frac{1}{\sigma_e^2} \mathbf{X}^\top \mathbf{X} + \mathbf{D}^{-1} \right) \boldsymbol{\beta} - 2 \frac{1}{\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})^\top \mathbf{X} \boldsymbol{\beta} \right) \right)
\]

#### 7.4. 与标准正态分布对比

我们将这个结果与标准多元正态分布形式对比：
\[
p(\boldsymbol{\beta}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{\beta} - \boldsymbol{\mu_{\beta}})^\top \boldsymbol{\Sigma_{\beta}}^{-1} (\boldsymbol{\beta} - \boldsymbol{\mu_{\beta}}) \right)
\]

通过配平方，我们可以整理出条件后验分布的均值和协方差矩阵的表达式。

##### 7.4.1. 配平方确定协方差矩阵

从展开式中的二次项 \( \boldsymbol{\beta}^\top \left( \frac{1}{\sigma_e^2} \mathbf{X}^\top \mathbf{X} + \mathbf{D}^{-1} \right) \boldsymbol{\beta} \)，我们可以直接得到条件协方差矩阵的逆：
\[
\boldsymbol{\Sigma_{\beta}}^{-1} = \frac{1}{\sigma_e^2} \mathbf{X}^\top \mathbf{X} + \mathbf{D}^{-1}
\]

因此，条件协方差矩阵为：
\[
\boldsymbol{\Sigma_{\beta}} = \left( \frac{1}{\sigma_e^2} \mathbf{X}^\top \mathbf{X} + \mathbf{D}^{-1} \right)^{-1}
\]

##### 7.4.2. 配平方确定均值

再来看线性项 \( -2 \frac{1}{\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})^\top \mathbf{X} \boldsymbol{\beta} \)，它对应于多元正态分布中的 \( \boldsymbol{\mu_{\beta}} \) 和 \( \boldsymbol{\Sigma_{\beta}}^{-1} \) 的线性部分。通过配平方，我们可以得出均值 \( \boldsymbol{\mu_{\beta}} \)：
\[
\boldsymbol{\mu_{\beta}} = \boldsymbol{\Sigma_{\beta}} \cdot \frac{1}{\sigma_e^2} \mathbf{X}^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})
\]

#### 7.5. 最终结果

通过配平方操作，我们得到了 SNP 标记效应 \( \boldsymbol{\beta} \) 的条件后验分布，其结果为多元正态分布。结果为：

- **条件均值 \( \boldsymbol{\mu_{\beta}} \)**：
  \[
  \boldsymbol{\mu_{\beta}} = \left( \mathbf{X}^\top \mathbf{X} + \sigma_e^2 \mathbf{D}^{-1} \right)^{-1} \mathbf{X}^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})
  \]

- **条件协方差矩阵 \( \boldsymbol{\Sigma_{\beta}} \)**：
  \[
  \boldsymbol{\Sigma_{\beta}} = \sigma_e^2 \left( \mathbf{X}^\top \mathbf{X} + \sigma_e^2 \mathbf{D}^{-1} \right)^{-1}
  \]

### 8. 残差方差 \( \sigma_e^2 \) 的条件后验分布
#### 8.1. 残差方差的模型假设

残差项 \( \mathbf{e} = \mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta} \) 假设服从正态分布：
\[
\mathbf{e} \sim \mathcal{N}(0, \sigma_e^2 \mathbf{I})
\]
我们希望推导 \( \sigma_e^2 \) 的条件后验分布。

#### 8.2. 似然函数

表型数据 \( \mathbf{y} \) 的似然函数可以写作：
\[
p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \propto (\sigma_e^2)^{-n/2} \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X}\boldsymbol{\beta}) \right)
\]
这个式子展示了数据 \( \mathbf{y} \) 在给定 \( \boldsymbol{\beta_f} \)、\( \boldsymbol{\beta} \) 和 \( \sigma_e^2 \) 下的概率。

#### 8.3. 先验分布

我们假设残差方差 \( \sigma_e^2 \) 服从**逆卡方分布**，即：
\[
\sigma_e^2 \sim \text{Inv-}\chi^2(\nu_e, S_e)
\]
其概率密度函数为：
\[
p(\sigma_e^2) \propto (\sigma_e^2)^{-(\nu_e/2 + 1)} \exp \left( -\frac{S_e}{2\sigma_e^2} \right)
\]
其中，\( \nu_e \) 是自由度，\( S_e \) 是比例参数。

#### 8.4. 条件后验分布

根据贝叶斯定理，条件后验分布为：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) p(\sigma_e^2)
\]
这意味着我们需要将似然函数和先验分布相乘。

##### 8.4.1. 将似然函数和先验分布相乘

首先，我们将 \( p(\mathbf{y} | \boldsymbol{\beta_f}, \boldsymbol{\beta}, \sigma_e^2) \) 和 \( p(\sigma_e^2) \) 结合在一起，得到：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto (\sigma_e^2)^{-n/2} \exp \left( -\frac{1}{2\sigma_e^2} (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) \right) (\sigma_e^2)^{-(\nu_e/2 + 1)} \exp \left( -\frac{S_e}{2\sigma_e^2} \right)
\]

我们可以将相同指数部分合并为一个整体：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto (\sigma_e^2)^{-(n/2 + \nu_e/2 + 1)} \exp \left( -\frac{1}{2\sigma_e^2} \left[ (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta}) + S_e \right] \right)
\]

##### 8.4.2. 将公式整理为逆卡方分布的形式

为了使表达式变得清晰，我们将 \( \mathbf{e} = \mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta} \) 的平方和记为 \( SS_{\text{residual}} \)，即：
\[
SS_{\text{residual}} = (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})
\]

因此，条件后验分布可以写为：
\[
p(\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta}) \propto (\sigma_e^2)^{-(n/2 + \nu_e/2 + 1)} \exp \left( -\frac{1}{2\sigma_e^2} (SS_{\text{residual}} + S_e) \right)
\]

这就是**逆卡方分布**的形式，因此：
\[
\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta} \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]

#### 8.5. 结论

我们完整推导了残差方差 \( \sigma_e^2 \) 的条件后验分布，得到：
\[
\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta} \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]
其中，\( SS_{\text{residual}} \) 是残差平方和。

### 9. SNP 标记效应方差矩阵 \( \mathbf{D} \) 的条件后验分布

#### 9.1. 模型假设

SNP 标记效应 \( \beta_j \) 假设服从正态分布，且每个标记的效应方差不同，即：
\[
\beta_j \sim \mathcal{N}(0, \sigma_j^2)
\]
其中 \( \sigma_j^2 \) 是第 \( j \) 个 SNP 标记效应的方差。

假设每个方差 \( \sigma_j^2 \) 的先验分布为**逆卡方分布**，即：
\[
\sigma_j^2 \sim \text{Inv-}\chi^2(\nu_j, S_j)
\]
其中 \( \nu_j \) 是自由度，\( S_j \) 是比例参数。

#### 9.2. 似然函数

对于每个 SNP 标记效应 \( \beta_j \)，其似然函数为：
\[
p(\beta_j | \sigma_j^2) \propto (\sigma_j^2)^{-1/2} \exp \left( -\frac{\beta_j^2}{2\sigma_j^2} \right)
\]
这个式子表示，在给定 \( \sigma_j^2 \) 的情况下，标记效应 \( \beta_j \) 的概率。

#### 9.3. 先验分布

每个标记效应方差 \( \sigma_j^2 \) 的先验分布为：
\[
p(\sigma_j^2) \propto (\sigma_j^2)^{-(\nu_j/2 + 1)} \exp \left( -\frac{S_j}{2\sigma_j^2} \right)
\]
这描述了我们在没有观察到数据时对 \( \sigma_j^2 \) 的先验信念。

#### 9.4. 条件后验分布

根据贝叶斯定理，条件后验分布为：
\[
p(\sigma_j^2 | \beta_j) \propto p(\beta_j | \sigma_j^2) p(\sigma_j^2)
\]
将似然函数和先验分布相乘，得到：
\[
p(\sigma_j^2 | \beta_j) \propto (\sigma_j^2)^{-1/2} \exp \left( -\frac{\beta_j^2}{2\sigma_j^2} \right) (\sigma_j^2)^{-(\nu_j/2 + 1)} \exp \left( -\frac{S_j}{2\sigma_j^2} \right)
\]

##### 9.4.1. 合并相同指数项

我们将所有指数项合并在一起：
\[
p(\sigma_j^2 | \beta_j) \propto (\sigma_j^2)^{-(\nu_j/2 + 1 + 1/2)} \exp \left( -\frac{1}{2\sigma_j^2} (\beta_j^2 + S_j) \right)
\]
整理后得到：
\[
p(\sigma_j^2 | \beta_j) \propto (\sigma_j^2)^{-(\nu_j/2 + 1)} \exp \left( -\frac{1}{2\sigma_j^2} (\beta_j^2 + S_j) \right)
\]

##### 9.4.2. 确定分布形式

这个表达式与**逆卡方分布**的形式一致，因此 \( \sigma_j^2 \) 的条件后验分布为：
\[
\sigma_j^2 | \beta_j \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]

#### 9.5. 总结

通过详细推导，我们得到了每个 SNP 标记效应方差 \( \sigma_j^2 \) 的条件后验分布，结果为：
\[
\sigma_j^2 | \beta_j \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]
这个条件后验分布基于观察到的标记效应 \( \beta_j \) 更新了每个 SNP 标记效应的方差。

### 10. Gibbs 采样优化过程
通过 **Gibbs 采样** 方法，我们可以从每个参数的条件后验分布中采样，逐步更新模型参数，最终得到参数的后验分布。具体到我们讨论的贝叶斯模型中，主要参数包括：固定效应 \( \boldsymbol{\beta_f} \)、SNP 标记效应 \( \boldsymbol{\beta} \)、残差方差 \( \sigma_e^2 \)，以及 SNP 标记效应的方差矩阵 \( \mathbf{D} \)。以下是通过 **Gibbs 采样** 估计这些参数的过程。

#### 10.1. Gibbs 采样的基本思路

**Gibbs 采样**是一种马尔科夫链蒙特卡洛（MCMC）方法，它通过从各个参数的**条件后验分布**中抽样，迭代更新参数，最终逼近每个参数的后验分布。在每次迭代中，给定当前其他参数的值，我们从每个参数的条件分布中抽样，然后继续更新其他参数。

#### 10.2. 采样的具体步骤

假设我们已经推导出每个参数的条件后验分布，下面展示如何利用 **Gibbs 采样** 逐步更新这些参数。

##### 10.2.1. 固定效应 \( \boldsymbol{\beta_f} \) 的采样

从条件后验分布：
\[
\boldsymbol{\beta_f} | \mathbf{y}, \boldsymbol{\beta}, \sigma_e^2 \sim \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \boldsymbol{\Sigma_{\beta_f}})
\]
其中，均值和协方差矩阵已经推导出来，分别为：
\[
\boldsymbol{\mu_{\beta_f}} = \left( \mathbf{X_f}^\top \mathbf{X_f} + \frac{\sigma_e^2}{\sigma_f^2} \mathbf{I} \right)^{-1} \mathbf{X_f}^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
\]
\[
\boldsymbol{\Sigma_{\beta_f}} = \sigma_e^2 \left( \mathbf{X_f}^\top \mathbf{X_f} + \frac{\sigma_e^2}{\sigma_f^2} \mathbf{I} \right)^{-1}
\]

**采样过程**：
1. 计算当前的 \( \boldsymbol{\mu_{\beta_f}} \) 和 \( \boldsymbol{\Sigma_{\beta_f}} \)，它们依赖于当前的 \( \sigma_e^2 \) 和 \( \boldsymbol{\beta} \)。
2. 从多元正态分布 \( \mathcal{N}(\boldsymbol{\mu_{\beta_f}}, \boldsymbol{\Sigma_{\beta_f}}) \) 中抽样 \( \boldsymbol{\beta_f} \)。

##### 10.2.2. SNP 标记效应 \( \boldsymbol{\beta} \) 的采样

从条件后验分布：
\[
\boldsymbol{\beta} | \mathbf{y}, \boldsymbol{\beta_f}, \sigma_e^2, \mathbf{D} \sim \mathcal{N}(\boldsymbol{\mu_{\beta}}, \boldsymbol{\Sigma_{\beta}})
\]
其中均值和协方差矩阵为：
\[
\boldsymbol{\mu_{\beta}} = \left( \mathbf{X}^\top \mathbf{X} + \sigma_e^2 \mathbf{D}^{-1} \right)^{-1} \mathbf{X}^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f})
\]
\[
\boldsymbol{\Sigma_{\beta}} = \sigma_e^2 \left( \mathbf{X}^\top \mathbf{X} + \sigma_e^2 \mathbf{D}^{-1} \right)^{-1}
\]

**采样过程**：
1. 根据当前的 \( \boldsymbol{\beta_f} \)、\( \sigma_e^2 \) 和 \( \mathbf{D} \) 计算均值 \( \boldsymbol{\mu_{\beta}} \) 和协方差矩阵 \( \boldsymbol{\Sigma_{\beta}} \)。
2. 从多元正态分布 \( \mathcal{N}(\boldsymbol{\mu_{\beta}}, \boldsymbol{\Sigma_{\beta}}) \) 中抽样 \( \boldsymbol{\beta} \)。

##### 10.2.3. 残差方差 \( \sigma_e^2 \) 的采样

从条件后验分布：
\[
\sigma_e^2 | \mathbf{y}, \boldsymbol{\beta_f}, \boldsymbol{\beta} \sim \text{Inv-}\chi^2 \left( \nu_e + n, \frac{SS_{\text{residual}} + S_e}{\nu_e + n} \right)
\]
其中，\( SS_{\text{residual}} \) 是残差平方和：
\[
SS_{\text{residual}} = (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X_f} \boldsymbol{\beta_f} - \mathbf{X} \boldsymbol{\beta})
\]

**采样过程**：
1. 计算当前的残差平方和 \( SS_{\text{residual}} \)。
2. 从逆卡方分布中抽样 \( \sigma_e^2 \)。

##### 10.2.4. SNP 标记效应方差 \( \sigma_j^2 \) 的采样

对于每个 \( \sigma_j^2 \)，从条件后验分布：
\[
\sigma_j^2 | \beta_j \sim \text{Inv-}\chi^2 \left( \nu_j + 1, \frac{\nu_j S_j + \beta_j^2}{\nu_j + 1} \right)
\]

**采样过程**：
1. 计算当前的 \( \beta_j^2 \)。
2. 从逆卡方分布中抽样每个 \( \sigma_j^2 \)。

#### 10.3. Gibbs 采样的完整过程

1. **初始化**：为每个参数赋予初始值。例如，可以通过随机初始化 \( \boldsymbol{\beta_f} \)、\( \boldsymbol{\beta} \)、\( \sigma_e^2 \) 和 \( \mathbf{D} \)。
2. **循环更新**：对于每次迭代，依次从每个参数的条件后验分布中抽样，更新参数值。具体顺序如下：
   - 从 \( \boldsymbol{\beta_f} \) 的条件后验分布中抽样。
   - 从 \( \boldsymbol{\beta} \) 的条件后验分布中抽样。
   - 从 \( \sigma_e^2 \) 的条件后验分布中抽样。
   - 从每个 \( \sigma_j^2 \) 的条件后验分布中抽样。
3. **收敛**：经过若干次迭代后，参数的采样结果逐渐逼近其真实后验分布。为了确保采样的稳定性，前几次迭代（称为“burn-in”阶段）可以舍弃。

#### 10.4. 输出后验分布

在 Gibbs 采样完成后，可以通过生成的样本来估计每个参数的后验分布。例如，均值估计可以通过所有抽样值的平均得到，方差和其他统计量也可以从抽样结果中计算。

如果需要更多关于某个具体步骤的解释，或希望看到代码实现，请告诉我！\\[\\]\\(\\)