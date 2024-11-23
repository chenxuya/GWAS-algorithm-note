## 1. **模型定义**

首先，我们明确线性混合模型的定义：

\[
y = W a + x b + Z u + \varepsilon
\]

其中：
- \( y \) 是 \( n \times 1 \) 的响应变量向量。
- \( W \) 是 \( n \times p \) 的固定效应协变量矩阵。
- \( a \) 是 \( p \times 1 \) 的固定效应系数向量。
- \( x \) 是 \( n \times 1 \) 的标记效应向量。
- \( b \) 是标记效应的大小（标量）。
- \( Z \) 是 \( n \times m \) 的随机效应载荷矩阵。
- \( u \) 是 \( m \times 1 \) 的随机效应向量。
- \( \varepsilon \) 是 \( n \times 1 \) 的误差项向量。

**关键假设：**
1. **随机效应 \( u \)：**
   \[
   u \sim \mathcal{N}(0, \lambda \tau^{-1} K)
   \]
   其中，\( K \) 是已知的 \( m \times m \) 协方差矩阵，\( \lambda \) 和 \( \tau \) 是待估参数。

2. **误差项 \( \varepsilon \)：**
   \[
   \varepsilon \sim \mathcal{N}(0, \tau^{-1} I_n)
   \]
   \( I_n \) 是 \( n \times n \) 的单位矩阵。

3. **独立性：**
   随机效应 \( u \) 与误差项 \( \varepsilon \) 相互独立。

---

## 2. **协方差结构**

基于上述假设，我们可以推导出响应变量 \( y \) 的协方差矩阵。

1. **随机效应 \( u \) 的协方差：**
   \[
   \text{Cov}(u) = \lambda \tau^{-1} K
   \]

2. **误差项 \( \varepsilon \) 的协方差：**
   \[
   \text{Cov}(\varepsilon) = \tau^{-1} I_n
   \]

3. **响应变量 \( y \) 的协方差：**
   \[
   \text{Cov}(y) = \text{Cov}(Z u + \varepsilon) = Z \text{Cov}(u) Z^T + \text{Cov}(\varepsilon) = \lambda \tau^{-1} Z K Z^T + \tau^{-1} I_n
   \]
   简化得：
   \[
   \text{Cov}(y) = \tau^{-1} (\lambda G + I_n) = \tau^{-1} H
   \]
   其中：
   \[
   H = \lambda G + I_n \quad \text{且} \quad G = Z K Z^T
   \]

---

## 3. **似然函数的构建**

假设 \( y \) 服从多元正态分布，由于模型的线性性和高斯假设，\( y \) 的分布为：

\[
y \sim \mathcal{N}(W a + x b, \tau^{-1} H)
\]

### 3.1 **概率密度函数（PDF）**

多元正态分布的概率密度函数为：

\[
f(y) = \frac{|\tau H|^{1/2}}{(2\pi)^{n/2}} \exp\left( -\frac{1}{2} (y - W a - x b)^T (\tau H) (y - W a - x b) \right)
\]

### 3.2 **对数似然函数**

取对数得到对数似然函数：

\[
\ell = \log f(y) = \frac{1}{2} \log |\tau H| - \frac{n}{2} \log (2\pi) - \frac{1}{2} (y - W a - x b)^T (\tau H) (y - W a - x b)
\]

为了简化，我们可以忽略常数项 \( -\frac{n}{2} \log(2\pi) \)：

\[
\ell = \frac{1}{2} \log |\tau H| - \frac{1}{2} (y - W a - x b)^T (\tau H) (y - W a - x b) + \text{const}
\]

进一步简化：

\[
\ell = \frac{1}{2} \log |\tau H| - \frac{\tau}{2} (y - W a - x b)^T H (y - W a - x b) + \text{const}
\]


## 4. 固定效应 \( a \) 和 \( b \) 的估计

在给定方差组分 \( \lambda \) 和 \( \tau \) 的情况下，我们需要估计固定效应 \( a \) 和 \( b \)。由于 \( a \) 和 \( b \) 是固定效应，它们的估计可以通过**广义最小二乘法（Generalized Least Squares, GLS）**获得。

### 步骤 1：模型的矩阵表示

首先，将模型重新整理为矩阵形式：

\[
y = X \beta + Z u + \varepsilon
\]

其中：
- \( \beta = \begin{bmatrix} a \\ b \end{bmatrix} \) 是固定效应系数向量，维度为 \( (p + 1) \times 1 \)。
- \( X = \begin{bmatrix} W & x \end{bmatrix} \) 是固定效应协变量矩阵，维度为 \( n \times (p + 1) \)。

### 步骤 2：响应变量的期望和协方差

根据模型和假设条件：

\[
\mathbb{E}[y] = X \beta
\]

\[
\text{Cov}(y) = \text{Cov}(Z u + \varepsilon) = Z \text{Cov}(u) Z^T + \text{Cov}(\varepsilon) = \lambda \tau^{-1} Z K Z^T + \tau^{-1} I_n
\]

简化协方差矩阵：

\[
\text{Cov}(y) = \tau^{-1} (\lambda Z K Z^T + I_n) = \tau^{-1} H
\]

其中：

\[
H = \lambda G + I_n, \quad G = Z K Z^T
\]

### 步骤 3：广义最小二乘法（GLS）估计

广义最小二乘法的目标是最小化加权残差平方和，即：

\[
\text{minimize} \quad (y - X \beta)^T V^{-1} (y - X \beta)
\]

其中，\( V = \text{Cov}(y) = \tau^{-1} H \)，因此 \( V^{-1} = \tau H^{-1} \)。

**目标函数：**

\[
Q(\beta) = (y - X \beta)^T V^{-1} (y - X \beta) = \tau (y - X \beta)^T H^{-1} (y - X \beta)
\]

为了简化推导，我们可以忽略常数 \( \tau \)，因为最小化 \( Q(\beta) \) 与最小化 \( (y - X \beta)^T H^{-1} (y - X \beta) \) 是等价的。

**目标函数简化为：**

\[
Q(\beta) = (y - X \beta)^T H^{-1} (y - X \beta)
\]

### 步骤 4：求导并求解最小化问题

为了找到最小化 \( Q(\beta) \) 的 \( \beta \)，我们对 \( Q(\beta) \) 关于 \( \beta \) 求导并设导数为零。

**对 \( \beta \) 求导：**

\[
\frac{\partial Q(\beta)}{\partial \beta} = -2 X^T H^{-1} (y - X \beta) = 0
\]

将方程整理：

\[
X^T H^{-1} y = X^T H^{-1} X \beta
\]

**求解 \( \beta \)：**

\[
\beta = (X^T H^{-1} X)^{-1} X^T H^{-1} y
\]

这就是固定效应 \( a \) 和 \( b \) 的广义最小二乘估计量。

### 步骤 5：\( \beta \)估计表达分块表示

#### 1. **构造矩阵 \( X \) 和向量 \( \beta \) 的分块形式**

首先，将设计矩阵 \( X \) 和系数向量 \( \beta \) 进行分块：

\[
X = \begin{bmatrix} W & x \end{bmatrix}
\]
其中：
- \( W \) 是 \( n \times p \) 的协变量矩阵。
- \( x \) 是 \( n \times 1 \) 的标记效应向量。

对应地，将 \( \beta \) 表示为：

\[
\beta = \begin{bmatrix} a \\ b \end{bmatrix}
\]
其中：
- \( a \) 是 \( p \times 1 \) 的固定效应系数向量。
- \( b \) 是标量，代表标记效应的大小。

---

#### 2. **分块表示 \( X^T H^{-1} X \)**

根据广义最小二乘估计公式，我们需要计算 \( X^T H^{-1} X \) 的分块形式。展开后得到：

\[
X^T H^{-1} X = \begin{bmatrix} W^T \\ x^T \end{bmatrix} H^{-1} \begin{bmatrix} W & x \end{bmatrix}
\]

展开矩阵乘法得到：

\[
X^T H^{-1} X = \begin{bmatrix} W^T H^{-1} W & W^T H^{-1} x \\ x^T H^{-1} W & x^T H^{-1} x \end{bmatrix}
\]

这里的分块矩阵项分别表示如下：
- **左上块 \( W^T H^{-1} W \)**：是一个 \( p \times p \) 的矩阵，对应协变量矩阵 \( W \) 与 \( H^{-1} \) 的乘积。
- **右上块 \( W^T H^{-1} x \)**：是一个 \( p \times 1 \) 的向量，对应协变量矩阵 \( W \) 与标记效应 \( x \) 的交叉项。
- **左下块 \( x^T H^{-1} W \)**：是一个 \( 1 \times p \) 的向量，与右上块对称。
- **右下块 \( x^T H^{-1} x \)**：是一个标量，对应标记效应向量 \( x \) 自身的平方和。

---

#### 3. **分块表示 \( X^T H^{-1} y \)**

接下来，我们计算 \( X^T H^{-1} y \) 的分块形式：

\[
X^T H^{-1} y = \begin{bmatrix} W^T \\ x^T \end{bmatrix} H^{-1} y
\]

这可以分块表示为：

\[
X^T H^{-1} y = \begin{bmatrix} W^T H^{-1} y \\ x^T H^{-1} y \end{bmatrix}
\]

其中：
- **上部分 \( W^T H^{-1} y \)**：是一个 \( p \times 1 \) 的向量，对应协变量矩阵 \( W \) 的贡献。
- **下部分 \( x^T H^{-1} y \)**：是一个标量，对应标记效应向量 \( x \) 的贡献。

---

#### 4. **求解分块矩阵方程**

现在我们得到了分块矩阵 \( X^T H^{-1} X \) 和分块向量 \( X^T H^{-1} y \)：

\[
\begin{bmatrix} W^T H^{-1} W & W^T H^{-1} x \\ x^T H^{-1} W & x^T H^{-1} x \end{bmatrix} \begin{bmatrix} \hat{a} \\ \hat{b} \end{bmatrix} = \begin{bmatrix} W^T H^{-1} y \\ x^T H^{-1} y \end{bmatrix}
\]

这是一个分块线性方程组。我们可以将其拆解为两个部分：

1. **固定效应 \( a \) 的估计方程：**

   \[
   W^T H^{-1} W \hat{a} + W^T H^{-1} x \hat{b} = W^T H^{-1} y
   \]

2. **标记效应 \( b \) 的估计方程：**

   \[
   x^T H^{-1} W \hat{a} + x^T H^{-1} x \hat{b} = x^T H^{-1} y
   \]

这两个方程同时解出 \( \hat{a} \) 和 \( \hat{b} \) 的估计量。

---

#### 5. **进一步求解 \( \hat{a} \) 和 \( \hat{b} \)**

为了更清晰地展示如何求解 \( \hat{a} \) 和 \( \hat{b} \)，我们可以首先对第二个方程进行整理，解出 \( \hat{b} \)：

\[
\hat{b} = \frac{x^T H^{-1} y - x^T H^{-1} W \hat{a}}{x^T H^{-1} x}
\]

将 \( \hat{b} \) 的表达式代入第一个方程：

\[
W^T H^{-1} W \hat{a} + W^T H^{-1} x \left( \frac{x^T H^{-1} y - x^T H^{-1} W \hat{a}}{x^T H^{-1} x} \right) = W^T H^{-1} y
\]

进一步整理，得到 \( \hat{a} \) 的表达式。这个过程会略复杂，但在实际应用中可以使用数值方法来求解这个线性方程组。

通过分块矩阵的展开，我们能够逐步分离和求解出 \( \hat{a} \) 和 \( \hat{b} \) 的估计量，从而明确各个协变量和标记效应的独立贡献。

## 5. 最大化似然函数以估计方差组分 \( \lambda \) 和 \( \tau \)
我们需要对方差组分 \( \lambda \) 和 \( \tau \) 进行估计。这涉及到对数似然函数 \( \ell \) 关于这两个参数的偏导数，并将其设为零，从而得到估计方程。以下将详细推导每一步。

### 5.1 **固定效应的估计回顾**

在估计方差组分之前，我们通常需要先估计固定效应 \( a \) 和 \( b \)。这可以通过广义最小二乘法（Generalized Least Squares, GLS）完成，估计公式为：

\[
\hat{\beta} = \begin{bmatrix} \hat{a} \\ \hat{b} \end{bmatrix} = (X^T H^{-1} X)^{-1} X^T H^{-1} y
\]

其中：
\[
X = \begin{bmatrix} W & x \end{bmatrix}, \quad \beta = \begin{bmatrix} a \\ b \end{bmatrix}
\]

在此基础上，我们将 \( \hat{\beta} \) 代入对数似然函数，得到仅依赖于 \( \lambda \) 和 \( \tau \) 的对数似然函数。

### 5.2 **对数似然函数重新表示**

将固定效应的估计量 \( \hat{\beta} \) 代入对数似然函数：

\[
\ell(\lambda, \tau) = \frac{1}{2} \log |\tau H| - \frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) + \text{const}
\]

进一步展开：

\[
\ell(\lambda, \tau) = \frac{1}{2} \log |\tau (\lambda G + I_n)| - \frac{\tau}{2} (y - X \hat{\beta})^T (\lambda G + I_n) (y - X \hat{\beta}) + \text{const}
\]

### 5.3 **对 \( \tau \) 的偏导数**

为了估计 \( \tau \)，我们首先对 \( \tau \) 求偏导，并令其等于零。

**步骤 1：求导表达式**

\[
\frac{\partial \ell}{\partial \tau} = \frac{\partial}{\partial \tau} \left[ \frac{1}{2} \log |\tau H| - \frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \right] = 0
\]

**步骤 2：分别对两项求导**

1. **第一项：** \( \frac{1}{2} \log |\tau H| \)

   使用矩阵微积分中的性质，对 \( \tau \) 求导：

   \[
   \frac{\partial}{\partial \tau} \log |\tau H| = \frac{\partial}{\partial \tau} \log (\tau^n |H|) = \frac{\partial}{\partial \tau} (n \log \tau + \log |H|) = \frac{n}{\tau}
   \]

2. **第二项：** \( -\frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \)

   对 \( \tau \) 求导：

   \[
   \frac{\partial}{\partial \tau} \left[ -\frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \right] = -\frac{1}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta})
   \]

**步骤 3：合并导数结果并设为零**

将两部分导数相加：

\[
\frac{\partial \ell}{\partial \tau} = \frac{n}{2\tau} - \frac{1}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) = 0
\]

**步骤 4：解方程求 \( \tau \)**

移项并解得：

\[
\frac{n}{2\tau} = \frac{1}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta})
\]

消去 \( \frac{1}{2} \)：

\[
\frac{n}{\tau} = (y - X \hat{\beta})^T H (y - X \hat{\beta})
\]

解得 \( \tau \) 的估计值：

\[
\hat{\tau} = \frac{n}{(y - X \hat{\beta})^T H (y - X \hat{\beta})}
\]

### 5.4 **对 \( \lambda \) 的偏导数**

接下来，我们对 \( \lambda \) 求偏导，并令其等于零。

**步骤 1：求导表达式**

\[
\frac{\partial \ell}{\partial \lambda} = \frac{\partial}{\partial \lambda} \left[ \frac{1}{2} \log |\tau H| - \frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \right] = 0
\]

**步骤 2：分别对两项求导**

1. **第一项：** \( \frac{1}{2} \log |\tau H| \)

   使用矩阵微积分中的性质，对 \( \lambda \) 求导：

   \[
   \frac{\partial}{\partial \lambda} \log |\tau H| = \frac{\partial}{\partial \lambda} \log |\tau (\lambda G + I_n)| = \frac{\partial}{\partial \lambda} \log |\lambda G + I_n| = \text{tr} \left( (\lambda G + I_n)^{-1} G \right)
   \]

   因为 \( \tau \) 是关于 \( \lambda \) 的常数，相对于 \( \lambda \) 无关。

   因此：

   \[
   \frac{\partial}{\partial \lambda} \left[ \frac{1}{2} \log |\tau H| \right] = \frac{1}{2} \text{tr} \left( H^{-1} G \right)
   \]

2. **第二项：** \( -\frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \)

   对 \( \lambda \) 求导：

   \[
   \frac{\partial}{\partial \lambda} \left[ -\frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \right] = -\frac{\tau}{2} (y - X \hat{\beta})^T \frac{\partial H}{\partial \lambda} (y - X \hat{\beta})
   \]

   因为 \( H = \lambda G + I_n \)，所以：

   \[
   \frac{\partial H}{\partial \lambda} = G
   \]

   因此：

   \[
   \frac{\partial}{\partial \lambda} \left[ -\frac{\tau}{2} (y - X \hat{\beta})^T H (y - X \hat{\beta}) \right] = -\frac{\tau}{2} (y - X \hat{\beta})^T G (y - X \hat{\beta})
   \]

**步骤 3：合并导数结果并设为零**

将两部分导数相加：

\[
\frac{\partial \ell}{\partial \lambda} = \frac{1}{2} \text{tr}(H^{-1} G) - \frac{\tau}{2} (y - X \hat{\beta})^T G (y - X \hat{\beta}) = 0
\]

消去 \( \frac{1}{2} \)：

\[
\text{tr}(H^{-1} G) - \tau (y - X \hat{\beta})^T G (y - X \hat{\beta}) = 0
\]

**步骤 4：解方程求 \( \lambda \)**

整理得：

\[
\text{tr}(H^{-1} G) = \tau (y - X \hat{\beta})^T G (y - X \hat{\beta})
\]

这个方程通常是非线性的，无法直接解析求解 \( \lambda \)。因此，我们需要采用数值迭代方法（如牛顿-拉夫森法）来求解 \( \lambda \)。

### 5.5 **牛顿-拉夫森法求解 \( \lambda \)**

牛顿-拉夫森法是一种迭代方法，用于求解非线性方程。对于方程 \( f(\lambda) = 0 \)，其迭代公式为：

\[
\lambda^{(k+1)} = \lambda^{(k)} - \frac{f(\lambda^{(k)})}{f'(\lambda^{(k)})}
\]

其中，\( f(\lambda) = \text{tr}(H^{-1} G) - \tau (y - X \hat{\beta})^T G (y - X \hat{\beta}) \)。

**步骤 1：计算 \( f(\lambda) \)**

\[
f(\lambda) = \text{tr}(H^{-1} G) - \tau (y - X \hat{\beta})^T G (y - X \hat{\beta})
\]

**步骤 2：计算 \( f'(\lambda) \)**

为了应用牛顿-拉夫森法，我们需要计算 \( f(\lambda) \) 关于 \( \lambda \) 的导数 \( f'(\lambda) \)。

首先，记 \( H = \lambda G + I_n \)，则：

\[
\frac{\partial H}{\partial \lambda} = G
\]

利用矩阵微积分中的性质，计算导数：

1. **计算 \( \frac{\partial}{\partial \lambda} \text{tr}(H^{-1} G) \)**

   \[
   \frac{\partial}{\partial \lambda} \text{tr}(H^{-1} G) = \text{tr}\left( \frac{\partial}{\partial \lambda} (H^{-1} G) \right) = \text{tr}\left( -H^{-1} \frac{\partial H}{\partial \lambda} H^{-1} G \right) = -\text{tr}(H^{-1} G H^{-1} G)
   \]

   因为 \( \frac{\partial H}{\partial \lambda} = G \)，且利用 \( \frac{\partial}{\partial \lambda} H^{-1} = -H^{-1} \frac{\partial H}{\partial \lambda} H^{-1} \)。

2. **计算 \( \frac{\partial}{\partial \lambda} [\tau (y - X \hat{\beta})^T G (y - X \hat{\beta})] \)**

   由于 \( \tau \) 已经通过 \( \hat{\beta} \) 的估计与 \( H \) 相关，我们需要考虑 \( \hat{\beta} \) 随 \( \lambda \) 的变化。然而，为了简化推导，通常假设 \( \hat{\beta} \) 对 \( \lambda \) 的依赖较弱，可以近似忽略 \( \hat{\beta} \) 的导数。这种近似在高斯-马尔可夫假设下是合理的。

   因此：

   \[
   \frac{\partial}{\partial \lambda} [\tau (y - X \hat{\beta})^T G (y - X \hat{\beta})] = \tau (y - X \hat{\beta})^T \frac{\partial}{\partial \lambda} (G) (y - X \hat{\beta}) = 0
   \]

   因为 \( G \) 是与 \( \lambda \) 无关的已知矩阵。

   但是，严格来说，如果 \( G \) 与 \( \lambda \) 有关（例如 \( G \) 本身依赖于 \( \lambda \)），则需要考虑其导数。在本模型中，假设 \( G = Z K Z^T \) 是已知且与 \( \lambda \) 无关的。

   因此：

   \[
   \frac{\partial}{\partial \lambda} [\tau (y - X \hat{\beta})^T G (y - X \hat{\beta})] = 0
   \]

**综合上述导数结果：**

\[
f'(\lambda) = -\text{tr}(H^{-1} G H^{-1} G)
\]

**步骤 3：更新 \( \lambda \)**

根据牛顿-拉夫森法的迭代公式：

\[
\lambda^{(k+1)} = \lambda^{(k)} - \frac{f(\lambda^{(k)})}{f'(\lambda^{(k)})} = \lambda^{(k)} - \frac{\text{tr}(H^{-1} G) - \tau (y - X \hat{\beta})^T G (y - X \hat{\beta})}{ -\text{tr}(H^{-1} G H^{-1} G) }
\]

简化得：

\[
\lambda^{(k+1)} = \lambda^{(k)} + \frac{\text{tr}(H^{-1} G) - \tau (y - X \hat{\beta})^T G (y - X \hat{\beta})}{\text{tr}(H^{-1} G H^{-1} G)}
\]

### 5.6 **迭代算法总结**

结合上述推导，我们可以总结出一个迭代算法，用于估计方差组分 \( \lambda \) 和 \( \tau \)：

1. **初始化：**
   - 选择初始值 \( \lambda^{(0)} \) 和 \( \tau^{(0)} \)（例如，设为1）。
   
2. **迭代步骤：** 对于每次迭代 \( k = 0, 1, 2, \ldots \)，执行以下操作：
   
   a. **计算 \( H^{(k)} = \lambda^{(k)} G + I_n \)**
   
   b. **估计固定效应 \( \hat{\beta}^{(k)} = (X^T H^{-1} X)^{-1} X^T H^{-1} y \)**
   
   c. **计算残差向量 \( r^{(k)} = y - X \hat{\beta}^{(k)} \)**
   
   d. **更新 \( \tau \)：**
   
      \[
      \tau^{(k+1)} = \frac{n}{(r^{(k)})^T H^{(k)} r^{(k)}}
      \]
   
   e. **计算 \( f(\lambda^{(k)}) = \text{tr}(H^{-1} G) - \tau^{(k+1)} (r^{(k)})^T G r^{(k)} \)**
   
   f. **计算 \( f'(\lambda^{(k)}) = -\text{tr}(H^{-1} G H^{-1} G) \)**
   
   g. **更新 \( \lambda \)：**
   
      \[
      \lambda^{(k+1)} = \lambda^{(k)} - \frac{f(\lambda^{(k)})}{f'(\lambda^{(k)})} = \lambda^{(k)} + \frac{\text{tr}(H^{-1} G) - \tau^{(k+1)} (r^{(k)})^T G r^{(k)}}{\text{tr}(H^{-1} G H^{-1} G)}
      \]
   
   h. **检查收敛条件：**
      - 如果 \( |\lambda^{(k+1)} - \lambda^{(k)}| < \epsilon \) 且 \( |\tau^{(k+1)} - \tau^{(k)}| < \epsilon \)，则停止迭代。
      - 否则，设 \( k = k + 1 \)，返回步骤 a。

3. **终止并输出：**
   - 当迭代满足收敛条件时，输出最终的 \( \hat{\lambda} = \lambda^{(k+1)} \) 和 \( \hat{\tau} = \tau^{(k+1)} \) 作为方差组分的估计值。

## 代码实现
### 示例代码
```python
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv

# 设置随机数种子，方便结果重现
np.random.seed(0)

# 模拟参数
n = 500   # 样本量
p = 5     # 协变量个数
m = 10    # 随机效应变量数

# 真实参数值
true_a = np.random.normal(0, 1, p)  # 随机生成 p 个协变量的固定效应
true_b = 1.5                        # SNP 效应
true_lambda = 0.8                   # 随机效应方差
true_tau = 0.2                      # 残差方差

# 生成数据
W = np.random.normal(0, 1, (n, p))   # 协变量矩阵
x = np.random.binomial(1, 0.5, (n, 1))   # SNP 矩阵
Z = np.random.normal(0, 1, (n, m))   # 随机效应载荷矩阵
K = np.identity(m)                   # 随机效应协方差矩阵（假设单位矩阵）

# 根据真实参数生成随机效应 u 和误差项 epsilon
u = multivariate_normal.rvs(mean=np.zeros(m), cov=true_lambda * true_tau**-1 * K)
epsilon = np.random.normal(0, np.sqrt(true_tau**-1), n)

# 生成响应变量 y
y = W @ true_a + x.flatten() * true_b + Z @ u + epsilon

# 估计参数初始值
lambda_est = 0.5
tau_est = 0.1

# 牛顿-拉夫森法迭代估计
max_iter = 500
tolerance = 1e-6
# 增加正则化下限
min_lambda = 0.001
min_tau = 0.01


for i in range(max_iter):
    # 计算 H 矩阵
    G = Z @ K @ Z.T
    H = lambda_est * G + np.identity(n)
    H_inv = inv(H)
    
    # 固定效应估计
    X = np.hstack((W, x))
    beta_hat = inv(X.T @ H_inv @ X) @ (X.T @ H_inv @ y)
    a_hat, b_hat = beta_hat[:-1], beta_hat[-1]
    
    # 计算残差向量
    r = y - X @ beta_hat
    
    # 更新 tau
    tau_est_new = max(n / (r.T @ H_inv @ r), min_tau)
    
    # 更新 lambda
    f_lambda = np.trace(H_inv @ G) - tau_est_new * (r.T @ H_inv @ G @ H_inv @ r)
    f_lambda_prime = -np.trace(H_inv @ G @ H_inv @ G)
    lambda_est_new = max(lambda_est - f_lambda / f_lambda_prime, min_lambda)
    
    # 检查收敛
    if abs(tau_est_new - tau_est) < tolerance and abs(lambda_est_new - lambda_est) < tolerance:
        tau_est, lambda_est = tau_est_new, lambda_est_new
        break
    
    tau_est, lambda_est = tau_est_new, lambda_est_new

print(f"True lambda: {true_lambda}, Estimated lambda: {lambda_est:.4f}")
print(f"True tau: {true_tau}, Estimated tau: {tau_est:.4f}")
print(f"True a: {true_a}, Estimated a: {a_hat}")
print(f"True b: {true_b}, Estimated b: {b_hat:.4f}")
```
### 示例输出
```
True lambda: 0.8, Estimated lambda: 0.0010
True tau: 0.2, Estimated tau: 0.0338
True a: [1.76405235 0.40015721 0.97873798 2.2408932  1.86755799], Estimated a: [1.78032821 0.71215985 1.0792115  2.27846428 1.96754172]
True b: 1.5, Estimated b: 1.3991
```
可以发在样本量设置为500时，估计的a和b与真实值差异比较小。\\[\\]\\(\\)