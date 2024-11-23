贝叶斯估计在广义线性模型 (GLM) 中的应用主要是通过将参数视为随机变量，结合先验分布和数据的似然函数，通过贝叶斯公式来计算参数的后验分布。这里是 GLM 的贝叶斯估计的数学推导过程：

# 1. 问题背景
GLM 的形式为：
\[
Y = X\beta + \epsilon
\]
其中：
- \( Y \) 是 \( n \times 1 \) 的因变量向量。
- \( X \) 是 \( n \times p \) 的设计矩阵。
- \( \beta \) 是 \( p \times 1 \) 的回归系数向量。
- \( \epsilon \sim N(0, \sigma^2 I) \) 是误差项，假设服从均值为 0，方差为 \( \sigma^2 \) 的正态分布。

我们的目标是通过贝叶斯估计求解 \( \beta \) 的后验分布。

# 2. 定义似然函数
在 GLM 中，给定参数 \( \beta \) 和 \( \sigma^2 \)，观测值 \( Y \) 的条件分布为：
\[
Y | X, \beta, \sigma^2 \sim N(X\beta, \sigma^2 I)
\]
因此，观测数据的似然函数可以写作：
\[
p(Y | X, \beta, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta)\right)
\]

# 3. 选择先验分布
为了进行贝叶斯估计，我们需要为 \( \beta \) 和 \( \sigma^2 \) 选择先验分布。常见的选择包括：

- 对 \( \beta \) 采用正态分布先验：
  \[
  \beta \sim N(\mu_0, \Sigma_0)
  \]
  其中，\( \mu_0 \) 和 \( \Sigma_0 \) 是先验均值和协方差矩阵。

- 对 \( \sigma^2 \) 采用逆伽马分布先验（如果不确定，可以用 Jeffreys 不可信息先验）：
  \[
  \sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
  \]
  其中，\( \alpha_0 \) 和 \( \beta_0 \) 是逆伽马分布的超参数。

# 4. 后验分布
根据贝叶斯公式，后验分布可以写作：
\[
p(\beta, \sigma^2 | Y, X) = \frac{ p(Y | X, \beta, \sigma^2) p(\beta,\sigma^2 | X)}{p(Y|X)} 
\]
由于\(\beta\)和 \(\sigma^2\)是先验假设，与X 无关，且\(\beta\)和 \(\sigma^2\)相互独立，因此，
\[
p(\beta, \sigma^2 | Y, X)  =\frac{ p(Y | X, \beta, \sigma^2) p(\beta) p(\sigma^2)}{p(Y|X)}
\]
其中p(Y|X)是**边缘似然** 或 **模型证据**它表示在给定 \( X \) 的情况下观察到 \( Y \) 的概率。这个项用于 **标准化** 后验分布，使得所有可能的参数值 \( \beta \) 和 \( \sigma^2 \) 的概率密度函数的总和等于 1。

具体来说：

- \( p(Y | X) \) 是通过对所有可能的 \( \beta \) 和 \( \sigma^2 \) 值进行积分得到的，即
  \[
  p(Y | X) = \int \int p(Y | X, \beta, \sigma^2) p(\beta) p(\sigma^2) \, d\beta \, d\sigma^2
  \]
  
- 这个项确保了后验分布 \( p(\beta, \sigma^2 | Y, X) \) 是一个 **有效的概率分布**。
此项与\(\beta\)和 \(\sigma^2\)的取值无关。因此：
\[
p(\beta, \sigma^2 | Y, X) \propto p(Y | X, \beta, \sigma^2) p(\beta) p(\sigma^2)
\]
用“\(\propto\)”来表示比例关系，因为只关心与 \(\beta\) 有关的部分，忽略了与 \(\beta\) 无关的常数项。
将似然函数和先验分布代入，得到：
\[
p(\beta, \sigma^2 | Y, X) \propto \frac{1}{(2\pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta)\right) \times \exp\left(-\frac{1}{2} (\beta - \mu_0)^T \Sigma_0^{-1} (\beta - \mu_0)\right) \times \left(\frac{1}{\sigma^2}\right)^{\alpha_0 + 1} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
\]

# 5. 求解后验分布
由于直接求解后验分布可能非常复杂，通常我们采用以下方法来简化计算：

## 5.1 条件后验分布
通过分解后验分布，可以得到条件后验分布：
### 5.1.1 给定 \( \sigma^2 \) 的 \( \beta \) 的条件后验分布
#### 1. 给定 \( \sigma^2 \) 的 \( \beta \) 的条件后验分布
根据贝叶斯公式，后验分布（给定数据后，我们对 \(\beta\) 的更新看法）为：
\[
p(\beta | Y, X, \sigma^2) = \frac{p(Y | X, \beta, \sigma^2) p(\beta | \sigma^2,X)}{p(Y | X, \sigma^2)} \\
= \frac{p(Y | X, \beta, \sigma^2) p(\beta)}{p(Y | X, \sigma^2)}
\]
其中\( p(Y | X, \sigma^2) \)是归一化常数，称为边际似然或证据（Evidence）。它是对所有可能的 \( \beta \) 值积分后得到的结果：
   \[
   p(Y | X, \sigma^2) = \int p(Y | X, \beta, \sigma^2) p(\beta) \, d\beta
   \]
这个项确保了后验分布 \( p(\beta | Y, X, \sigma^2) \) 的积分为 1。这一项不随\(\beta\) 的更新而变化，因此，它是一个常数项。
于是：
\[
p(\beta | Y, X, \sigma^2) \propto p(Y | X, \beta, \sigma^2) \cdot p(\beta)
\]
其中 \(p(Y | X, \beta, \sigma^2)\) 是似然函数，\(p(\beta)\) 是先验分布。我们用“\(\propto\)”来表示比例关系，因为只关心与 \(\beta\) 有关的部分，忽略了与 \(\beta\) 无关的常数项。
#### 2. 代入似然函数和先验分布
将似然函数和先验分布代入后，得到：
\[
p(\beta | Y, X, \sigma^2) \propto \exp\left(-\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta)\right) \cdot \exp\left(-\frac{1}{2} (\beta - \mu_0)^T \Sigma_0^{-1} (\beta - \mu_0)\right)
\]
这一步中，我们用“\(\propto\)”来表示比例关系，因为只关心与 \(\beta\) 有关的部分，忽略了与 \(\beta\) 无关的常数项。

#### 3. 合并指数项
将两个指数项合并成一个，得到：
\[
p(\beta | Y, X, \sigma^2) \propto \exp\left(-\frac{1}{2} \left(\frac{1}{\sigma^2} (Y - X\beta)^T (Y - X\beta) + (\beta - \mu_0)^T \Sigma_0^{-1} (\beta - \mu_0)\right)\right)
\]
接下来，我们要将上式化简，以便找出 \(\beta\) 的分布形式。

#### 4. 展开并整理
首先，展开第一个括号：
\[
(Y - X\beta)^T (Y - X\beta) = Y^T Y - 2Y^T X\beta + \beta^T X^T X \beta
\]
然后，将整个公式代入并整理得到：
\[
p(\beta | Y, X, \sigma^2) \propto \exp\left(-\frac{1}{2} \left(\beta^T \left(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\right) \beta - 2\beta^T \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right) + \text{常数项}\right)\right)
\]
忽略掉与 \(\beta\) 无关的常数项。

#### 5. 化简为正态分布形式
##### 1. 从二次型形式开始
我们有如下的表达式：
\[
p(\beta | Y, X, \sigma^2) \propto \exp\left(-\frac{1}{2} \left(\beta^T \left(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\right) \beta - 2\beta^T \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right)\right)\right)
\]

忽略与 \(\beta\) 无关的常数项，我们可以将该表达式重新写成关于 \(\beta\) 的二次型形式。

##### 2. 写成标准正态分布的二次型形式
我们希望将上式写成标准的正态分布二次型形式：
\[
-\frac{1}{2}(\beta - \tilde{\mu})^T \tilde{\Sigma}^{-1} (\beta - \tilde{\mu}) = -\frac{1}{2} \left( \beta^T \tilde{\Sigma}^{-1} \beta - 2 \beta^T \tilde{\Sigma}^{-1} \tilde{\mu} + \tilde{\mu}^T \tilde{\Sigma}^{-1} \tilde{\mu} \right).
\]
在这个形式中，\(\tilde{\mu}\) 表示均值向量，\(\tilde{\Sigma}\) 表示协方差矩阵。为了达到这种形式，我们需要完成平方（Complete the square）。

##### 3. 完成平方过程
我们将表达式中的二次型项进行展开：

\[
\beta^T \left(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\right) \beta - 2\beta^T \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right)
\]

###### 步骤 1：识别协方差矩阵 \(\tilde{\Sigma}\)
首先，我们观察到二次型系数 \(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\) 对应于正态分布中的协方差矩阵的逆，因此我们有：
\[
\tilde{\Sigma}^{-1} = \frac{X^T X}{\sigma^2} + \Sigma_0^{-1}
\]
从而得到协方差矩阵为：
\[
\tilde{\Sigma} = \left(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\right)^{-1}
\]

###### 步骤 2：确定均值向量 \(\tilde{\mu}\)
为了使得线性项 \( -2\beta^T \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right) \) 可以写成标准形式中的 \(-2\tilde{\Sigma}^{-1}\tilde{\mu}^T \beta\)，我们将：
\[
\tilde{\Sigma}^{-1} \tilde{\mu} = \frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0
\]

然后将 \(\tilde{\Sigma}\) 左乘到等式两边，得到：
\[
\tilde{\mu} = \tilde{\Sigma} \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right)
\]

##### 最终结果
因此，我们得到了 \(\beta\) 的后验分布,均值和协方差矩阵分别为：
\[p(\beta | Y, X, \sigma^2) \sim N(\tilde{\mu}, \tilde{\Sigma}) \\
\tilde{\mu} = \tilde{\Sigma} \left(\frac{X^T Y}{\sigma^2} + \Sigma_0^{-1} \mu_0\right)\\
\tilde{\Sigma} = \left(\frac{X^T X}{\sigma^2} + \Sigma_0^{-1}\right)^{-1}
\]

### 5.1.2给定 \( \beta \) 的 \( \sigma^2 \) 的条件后验分布

#### 1. 根据贝叶斯公式计算条件后验分布
根据贝叶斯公式，我们可以写出给定 \(\beta\) 和数据 \(Y\) 的情况下，\(\sigma^2\) 的条件后验分布：

\[
p(\sigma^2 | Y, X, \beta) = \frac{p(Y | X, \beta, \sigma^2) p(\sigma^2|X, \beta)}{p(Y | X, \beta)}
\]

其中：
- \(p(Y | X, \beta, \sigma^2)\) 是在已知 \(\beta\) 和 \(\sigma^2\) 时的似然函数。
- \(p(\sigma^2 | X, \beta)\) 是 \(\sigma^2\) 的先验分布。与\(X\)和 \(\beta\)无关。因此可直接写成\(p(\sigma^2)\)
- \(p(Y | X, \beta)\) 是一个与 \(\sigma^2\) 无关的归一化常数。

由于 \(p(Y | X, \beta)\) 不随 \(\sigma^2\) 变化，因此我们可以用比例符号 \(\propto\) 表示这一关系：

\[
p(\sigma^2 | Y, X, \beta) \propto p(Y | X, \beta, \sigma^2) \cdot p(\sigma^2)
\]

#### 2. 写出似然函数和先验分布

##### 2.1 似然函数
在给定 \(\beta\) 的条件下，\(Y\) 的分布为：

\[
Y | X, \beta, \sigma^2 \sim N(X\beta, \sigma^2 I)
\]

因此，\(Y\) 的条件概率密度为：

\[
p(Y | X, \beta, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta)\right)
\]

##### 2.2 \(\sigma^2\) 的先验分布
假设 \(\sigma^2\) 的先验分布是逆伽马分布：

\[
p(\sigma^2) = \text{Inv-Gamma}(\alpha_0, \beta_0) \propto (\sigma^2)^{-(\alpha_0 + 1)} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
\]

这里：
- \(\alpha_0\) 和 \(\beta_0\) 是逆伽马分布的参数。

#### 3. 合并似然和先验分布
将似然函数和先验分布代入后，我们得到：

\[
p(\sigma^2 | Y, X, \beta) \propto \frac{1}{(\sigma^2)^{n/2}} \exp\left(-\frac{1}{2\sigma^2} (Y - X\beta)^T (Y - X\beta)\right) \cdot (\sigma^2)^{-(\alpha_0 + 1)} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
\]

#### 4. 将表达式整理为逆伽马分布的形式
我们可以将上式合并，得到关于 \(\sigma^2\) 的表达式：

\[
p(\sigma^2 | Y, X, \beta) \propto \frac{1}{(\sigma^2)^{n/2 + \alpha_0 + 1}} \exp\left(-\frac{1}{2\sigma^2} \left((Y - X\beta)^T (Y - X\beta) + 2\beta_0\right)\right)
\]

#### 5. 重写为逆伽马分布的标准形式
注意到上式中的 \((\sigma^2)^{n/2 + \alpha_0 + 1}\) 和指数部分的形式，我们可以看到给定 \(\beta\) 的 \(\sigma^2\) 的条件后验分布是一个逆伽马分布：

\[
\sigma^2 | Y, X, \beta \sim \text{Inv-Gamma}\left(\alpha_0 + \frac{n}{2}, \, \beta_0 + \frac{1}{2}(Y - X\beta)^T (Y - X\beta)\right)
\]

### 最终结果
因此，给定 \(\beta\) 的 \(\sigma^2\) 的条件后验分布为：

\[
p(\sigma^2 | Y, X, \beta) \sim \text{Inv-Gamma}\left(\alpha_0 + \frac{n}{2}, \, \beta_0 + \frac{1}{2}(Y - X\beta)^T (Y - X\beta)\right)
\]



## 5.2 吉布斯采样
由于我们得到了条件后验分布，可以使用吉布斯采样来迭代生成 \( \beta \) 和 \( \sigma^2 \) 的样本，进而估计其后验分布。

1. 初始化 \( \beta \) 和 \( \sigma^2 \) 的初始值。
2. 按以下步骤迭代：
   - 根据当前的 \( \sigma^2 \) 样本，从 \( \beta | Y, X, \sigma^2 \) 的条件后验分布中采样。
   - 根据新的 \( \beta \) 样本，从 \( \sigma^2 | Y, X, \beta \) 的条件后验分布中采样。
3. 重复上述步骤，直到收敛，最终的样本集可以用于近似 \( \beta \) 和 \( \sigma^2 \) 的后验分布。

# 6. 预测
在得到 \( \beta \) 和 \( \sigma^2 \) 的后验分布后，可以进行预测。对于新输入数据 \( X_{\text{new}} \)，预测分布为：
\[
Y_{\text{new}} | Y, X, X_{\text{new}} \sim N(X_{\text{new}} \tilde{\mu}, X_{\text{new}} \tilde{\Sigma} X_{\text{new}}^T + \sigma^2)
\]
其中 \( \tilde{\mu} \) 和 \( \tilde{\Sigma} \) 是后验分布的均值和方差。

# 总结
通过贝叶斯估计求解 GLM，能够为模型参数提供不确定性估计，而不是单一的点估计。这在数据量较小或模型复杂的情况下尤为有用，因为它利用先验信息增强了估计的鲁棒性。吉布斯采样为高维模型提供了一个有效的计算手段，通过条件后验分布逐步逼近参数的联合后验分布。\\[\\]\\(\\)