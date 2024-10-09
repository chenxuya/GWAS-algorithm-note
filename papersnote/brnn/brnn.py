import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import pandas as pd

# Function for truncated normal distribution
def rtrun(mu, sigma, a, b):
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(a) and np.isscalar(b):
        return stats.truncnorm.rvs((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    else:
        mu = np.asarray(mu)
        sigma = np.asarray([sigma])
        a = np.asarray(a)
        b = np.asarray(b)
        n = max(len(mu), len(sigma), len(a), len(b))
        return stats.truncnorm.rvs(
            (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma, size=n
        )
# Extract values of z such that y[i] == j
def extract(z, y, j):
    return z[y == j]

# Normalize a vector or matrix
def normalize(x, base, spread):
    if isinstance(x, np.ndarray) and x.ndim == 2:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(x)
    else:
        return 2 * (x - base) / spread - 1

# Un-normalize a vector or matrix
def un_normalize(z, base, spread):
    if isinstance(z, np.ndarray) and z.ndim == 2:
        return base + 0.5 * spread * (z + 1)
    else:
        return base + 0.5 * spread * (z + 1)

# Create a diagonal matrix
def ii(element=1, times=1):
    return np.diag([element] * times)

# Calculate the sum of squares of weights and biases
def Ew(theta):
    return np.sum(np.square(np.concatenate(theta)))

# Initialize weights and biases using the Nguyen-Widrow method
def initnw(neurons, p, n, npar):
    theta = [np.random.uniform(-0.5, 0.5, npar // neurons) for _ in range(neurons)]
    scaling_factor = 0.7 * (neurons ** (1.0 / n)) if p > 1 else 0.7 * neurons
    b = np.linspace(-1, 1, neurons)
    for i in range(neurons):
        lambda_param = theta[i]
        weight = lambda_param[0]
        bias = lambda_param[1]
        lambda_param = lambda_param[2:]
        norm = np.linalg.norm(lambda_param)
        lambda_param = scaling_factor * lambda_param / norm
        bias = scaling_factor * b[i] * np.sign(lambda_param[0])
        theta[i] = np.concatenate(([weight, bias], lambda_param))
    return theta

# Predictions function for a neural network
def predictions_nn(vecX, n, p, theta, neurons):
    yhat = np.zeros(n)
    for i in range(n):
        sum_ = 0
        for k in range(neurons):
            z = 0
            for j in range(p):
                z += vecX[i * p + j] * theta[k][j + 2]
            z += theta[k][1]
            sum_ += theta[k][0] * tansig(z)
        yhat[i] = sum_
    return yhat

# Tansig function
def tansig(x):
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

# Placeholder for calculating the Jacobian
def jacobian(vecX, n, p, npar, theta, neurons):
    vecJ = np.zeros((n, npar))
    for i in range(n):
        for k in range(neurons):
            z = 0
            for j in range(p):
                z += vecX[i * p + j] * theta[k][j + 2]
            z += theta[k][1]
            dtansig = np.power(sech(z), 2.0)
            vecJ[i, (p + 2) * k] = -tansig(z)
            vecJ[i, (p + 2) * k + 1] = -theta[k][0] * dtansig
            for j in range(p):
                vecJ[i, (p + 2) * k + j + 2] = -theta[k][0] * dtansig * vecX[i * p + j]
    return vecJ

# Sech function
def sech(x):
    return 2.0 * np.exp(x) / (np.exp(2.0 * x) + 1.0)

# Bayesian Regularized Neural Network ordinal MCMC function
def brnn_ordinal_mcmc(y, X, normalize=False, neurons=2, epochs=1000, mu=0.005, mu_dec=0.1, mu_inc=10, mu_max=1e10,
                      min_grad=1e-10, change=0.01, nIter=200, burnIn=100, thin=10, verbose=False):
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y must be a vector")
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a matrix")

    if normalize:
        x_base = X.min(axis=0)
        x_spread = X.max(axis=0) - x_base
        X_normalized = (X - x_base) / x_spread * 2 - 1
    else:
        X_normalized = X
        x_base, x_spread = None, None

    vecX = X_normalized.flatten()
    p = X_normalized.shape[1]
    n = len(y)
    countsY = np.bincount(y)
    nclass = len(countsY)
    threshold = np.concatenate(([-np.inf], stats.norm.ppf(np.cumsum(countsY[:-1]) / n), [np.inf]))
    m = -threshold[1]
    threshold[1:nclass] += m

    # Initial value of latent variable
    yStar = rtrun(mu=np.zeros(n), sigma=np.ones(n), a=threshold[y], b=threshold[y + 1])

    npar = neurons * (1 + 1 + p)
    theta = initnw(neurons, p, n, npar)
    gamma = npar
    alpha = gamma / (2 * Ew(theta))
    beta = 0.5

    yHat_mcmc = np.full((nIter, n), np.nan)
    yStar_mcmc = np.full((nIter, n), np.nan)
    threshold_mcmc = np.full((nIter, len(threshold)), np.nan)

    mu_orig = mu

    for iter in range(1, nIter + 2):
        if verbose:
            print("**********************************************************************")
            print(f"iter = {iter}")

        reason = "UNKNOWN"
        epoch = 1
        flag_gradient = True
        flag_mu = True
        flag_change_F = True
        flag_change_Ed = True
        F_history = []

        C_new = 0

        while epoch <= epochs and flag_mu and flag_change_Ed and flag_change_F:
            if verbose:
                print("----------------------------------------------------------------------")
                print(f"Epoch = {epoch}")

            J = jacobian(vecX, n, p, npar, theta, neurons)
            H = np.dot(J.T, J)
            e = yStar - predictions_nn(vecX, n, p, theta, neurons)

            g = 2 * (beta * np.dot(e.T, J) + alpha * np.concatenate(theta))
            mg = np.max(np.abs(g))
            flag_gradient = mg > min_grad
            Ed = np.sum(e ** 2)
            Ew_theta = Ew(theta)
            C = beta * Ed + alpha * Ew_theta

            if verbose:
                print(f"C = {C}\tEd = {Ed}\tEw = {Ew_theta}")
                print(f"gradient = {mg}")

            F_history.append(C_new)

            if epoch > 3:
                if np.max(np.abs(np.diff(F_history[-3:]))) < change:
                    flag_change_F = False
                    reason = f"Changes in F= beta*SCE + alpha*Ew in last 3 iterations less than {change}"

            flag_C = True
            flag_mu = mu <= mu_max
            while flag_C and flag_mu:
                U = np.linalg.cholesky(H + ii(2 * alpha + mu, npar))
                tmp = np.concatenate(theta) - np.linalg.solve(U, np.linalg.solve(U.T, g))
                theta_new = []
                for i in range(neurons):
                    theta_new.append(tmp[: 2 + p])
                    tmp = tmp[2 + p :]

                e_new = yStar - predictions_nn(vecX, n, p, theta_new, neurons)
                Ed = np.sum(e_new ** 2)
                Ew_theta = Ew(theta_new)
                C_new = beta * Ed + alpha * Ew_theta

                if verbose:
                    print(f"C_new = {C_new}\tEd = {Ed}\tEw = {Ew_theta}")

                if C_new < C:
                    mu *= mu_dec
                    if mu < 1e-20:
                        mu = 1e-20
                    flag_C = False
                else:
                    mu *= mu_inc

                if verbose:
                    print(f"mu = {mu}")
                flag_mu = mu <= mu_max

            theta = theta_new
            epoch += 1

            gamma = npar - 2 * alpha * np.sum(np.diag(np.linalg.inv(H + ii(2 * alpha, npar))))
            alpha = gamma / (2 * Ew_theta)

            if Ed < change:
                flag_change_Ed = False
                reason = f"SCE <= {change}"

            if verbose:
                print(f"gamma = {round(gamma, 4)}\talpha = {round(alpha, 4)}\tbeta = {round(beta, 4)}")

        if epoch - 1 == epochs:
            reason = "Maximum number of epochs reached"
        if not flag_mu:
            reason = "Maximum mu reached"
        if not flag_gradient:
            reason = "Minimum gradient reached"
        if not verbose:
            print(f"iter = {iter}\tgamma = {round(gamma, 4)}\talpha = {round(alpha, 4)}\tbeta = {round(beta, 4)}")

        if iter <= nIter:
            yHat = predictions_nn(vecX, n, p, theta, neurons)
            yStar = rtrun(mu=yHat, sigma=1, a=threshold[y], b=threshold[y + 1])

            threshold[1] = 0
            for m in range(2, nclass):
                lo = max(np.max(extract(yStar, y, m - 1)), threshold[m - 1])
                hi = min(np.min(extract(yStar, y, m)), threshold[m + 1])
                threshold[m] = np.random.uniform(lo, hi)

            yHat_mcmc[iter - 1] = yHat
            yStar_mcmc[iter - 1] = yStar
            threshold_mcmc[iter - 1] = threshold
        else:
            yStar = np.mean(yHat_mcmc[burnIn - 1 : nIter : thin], axis=0)

        mu = mu_orig

    threshold = np.mean(threshold_mcmc[burnIn - 1 : nIter : thin], axis=0)

    return {
        "theta": theta,
        "alpha": alpha,
        "gamma": gamma,
        "threshold": threshold,
        "n": n,
        "p": p,
        "neurons": neurons,
        "x_normalized": X_normalized,
        "x_base": x_base,
        "x_spread": x_spread,
        "normalize": normalize
    }

# Predict probability
def predict_probability(threshold, predictor):
    threshold = threshold[np.isfinite(threshold)]
    cum_prob = np.full((len(predictor), len(threshold)), np.nan)
    prob = np.full((len(predictor), len(threshold) + 1), np.nan)

    for j in range(len(threshold)):
        cum_prob[:, j] = stats.norm.cdf(threshold[j] - predictor)

    prob[:, 0] = cum_prob[:, 0]
    for j in range(1, len(threshold)):
        prob[:, j] = cum_prob[:, j] - cum_prob[:, j - 1]
    prob[:, len(threshold)] = 1 - cum_prob[:, -1]

    return prob

# Predict class and probabilities for Bayesian Regularized Neural Network ordinal model
def predict_brnn_ordinal(object, newdata=None):
    if newdata is None:
        y = predictions_nn(
            vecX=object['x_normalized'].flatten(), n=object['n'], p=object['p'],
            theta=object['theta'], neurons=object['neurons']
        )
        probability = predict_probability(object['threshold'], y)
        pred_class = np.digitize(y, object['threshold'])
    else:
        if newdata.ndim == 1:
            newdata = newdata.reshape(1, -1)
        if newdata.shape[1] != object['p']:
            raise ValueError("Number of predictors does not match model")

        if object['normalize']:
            newdata = normalize(newdata, base=object['x_base'], spread=object['x_spread'])

        y = predictions_nn(
            vecX=newdata.flatten(), n=newdata.shape[0], p=newdata.shape[1],
            theta=object['theta'], neurons=object['neurons']
        )
        probability = predict_probability(object['threshold'], y)
        pred_class = np.digitize(y, object['threshold'])

    return {'class': pred_class, 'probability': probability}

# Example usage
np.random.seed(42)
X_random = np.random.rand(100, 3)
true_params = np.array([[0.5, 0.2, 0.8, 0.3, 0.1],
                         [0.1, 0.4, 0.7, 0.6, 0.2],
                         [0.2, 0.1, 0.9, 0.5, 0.3]])

# Generate y using true parameters
y_true = predictions_nn(X_random.flatten(), X_random.shape[0], X_random.shape[1], true_params, true_params.shape[0])
# 计算25%，50%，75%分位数
arr = y_true
q25 = np.percentile(arr, 25)
q50 = np.percentile(arr, 50)
q75 = np.percentile(arr, 75)
# 将值分配到区间
discretized = np.digitize(y_true, np.array([-np.inf, q25, q50, q75, np.inf]))-1

# Fit the model to estimate parameters
model_estimated = brnn_ordinal_mcmc(discretized, X_random, neurons=3, epochs=500, nIter=100,burnIn=50, verbose=False)
threshold_estimated = model_estimated['threshold']
# Print estimated and true parameters
print("True parameters:", true_params)
print("Estimated parameters:", model_estimated['theta'])

# Predict y using the estimated model
y_pred = predictions_nn(X_random.flatten(), X_random.shape[0], X_random.shape[1], model_estimated['theta'], len(model_estimated['theta']))
# 根据threshold预测类别
y_pred_class = np.digitize(y_pred, threshold_estimated) -1

# Calculate Pearson correlation coefficient
correlation, _ = pearsonr(y_true, y_pred)
print("Pearson correlation coefficient between true and predicted y:", correlation)
# calculate spearman correlation coefficient
correlation, _ = spearmanr(y_true, y_pred)
print("Spearman correlation coefficient between true and predicted y:", correlation)

# Scatter plot of true y vs predicted y and scatter plot of discretized vs y_pred_class
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred)
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.title("Scatter plot of true y vs predicted y")

plt.subplot(1, 2, 2)
plt.scatter(discretized, y_pred_class)
plt.xlabel("Discretized")
plt.ylabel("y_pred_class")
plt.title("Scatter plot of discretized vs y_pred_class")
plt.savefig('brnn_ordinal.png')
# 保存y_true, y_pred, discretized, y_pred_class 到一个dataframe中
df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'discretized': discretized, 'y_pred_class': y_pred_class})
df.to_csv('brnn_ordinal.csv', index=False)