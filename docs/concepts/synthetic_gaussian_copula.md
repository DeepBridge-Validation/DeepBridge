# Explanation of Gaussian Copula Synthetic Data Generation

The **Gaussian Copula** method is a statistical technique used to generate synthetic data by modeling both the marginal distributions and the dependencies (correlations) between variables. Below is a rigorous mathematical explanation of this method.

---

## 1. General Mathematical Background

A Copula is a statistical concept that separates the dependency structure of random variables from their marginal distributions. Formally, according to **Sklar’s theorem**, any multivariate joint distribution can be expressed as:

\[
F(x_1, x_2, \dots, x_K) = C\left(F_1(x_1), F_2(x_2), \dots, F_K(x_K)\right)
\]

where:

- \(F(x_1, x_2, \dots, x_K)\) is the joint cumulative distribution function (CDF) of the variables.
- \(F_k(x_k)\) are the marginal CDFs of each individual variable \(X_k\).
- \(C\) is a **copula function** that models the dependencies among variables.

The **Gaussian Copula** specifically uses a Gaussian (normal) distribution to model the dependency structure among variables.

---

## 2. Gaussian Copula Definition

The Gaussian copula is defined mathematically as follows:

\[
C(u_1, u_2, \dots, u_K) = \Phi_{\Sigma}\left(\Phi^{-1}(u_1), \Phi^{-1}(u_2), \dots, \Phi^{-1}(u_K)\right)
\]

where:

- \(u_k = F_k(x_k)\), with \(u_k \sim Uniform(0,1)\).
- \(\Phi^{-1}\) is the inverse standard normal CDF (quantile function).
- \(\Phi_{\Sigma}\) is the joint CDF of a multivariate normal distribution with mean vector \(0\) and correlation matrix \(\Sigma\).

The correlation matrix \(\Sigma\) captures the linear dependencies among variables after transformation.

---

## 3. Fitting Process (Estimating the Gaussian Copula)

### Step-by-step mathematical procedure:

#### Step 3.1: Estimate Marginal Distributions
For each variable \(X_k\) in the dataset \(X\), estimate the marginal CDF \(F_k(x)\):

- **Numerical variables:** Estimated through empirical distributions (e.g., Kernel Density Estimation - KDE) or parametric distributions (e.g., normal, beta, gamma).
  
  \[
  \hat{F}_k(x) = P(X_k \leq x)
  \]

- **Categorical variables:** Estimated via empirical frequency distributions:

  \[
  \hat{F}_k(c) = P(X_k \leq c) = \frac{\text{number of observations } \leq c}{N}
  \]

#### Step 3.2: Transform Data into Uniform Marginals
Convert each observation \( x_{ik} \) into uniform space using estimated marginals:

\[
u_{ik} = \hat{F}_k(x_{ik}), \quad u_{ik} \sim Uniform(0,1)
\]

#### Step 3.3: Convert Uniform Marginals into Gaussian Space
Next, apply the inverse Gaussian CDF (\(\Phi^{-1}\)) to transform \(u_{ik}\) into the standard normal space \(Z\):

\[
z_{ik} = \Phi^{-1}(u_{ik}), \quad z_{ik} \sim \mathcal{N}(0,1)
\]

This creates a transformed dataset \(Z\) that captures dependencies among variables in Gaussian space.

#### Step 3.4: Estimate Correlation Matrix (\(\Sigma\))
Calculate the correlation matrix \(\Sigma\) from the transformed Gaussian data:

\[
\Sigma_{ij} = \frac{\text{Cov}(Z_i,Z_j)}{\sqrt{\text{Var}(Z_i)\text{Var}(Z_j)}}
\]

This correlation matrix captures the linear dependency structure of the data in Gaussian-transformed space.

---

## 4. Synthetic Data Generation Procedure

After fitting the Gaussian Copula, synthetic data generation proceeds as follows:

### Step 4.1: Generate Synthetic Gaussian Data
Draw synthetic samples from the multivariate normal distribution using the estimated correlation matrix \(\Sigma\):

\[
Z^{*} \sim \mathcal{N}(0,\Sigma)
\]

### Step 4.2: Transform Back to Uniform Space
Convert the synthetic Gaussian data back to the uniform scale using the standard normal CDF (\(\Phi\)):

\[
U_k^{*} = \Phi(Z_k^{*}), \quad U_k^{*}\sim Uniform(0,1)
\]

### Step 4.3: Inverse Transform to Original Marginals
Apply inverse transformations based on previously estimated marginals (\(\hat{F}_k^{-1}\)) to obtain synthetic data samples on their original scale:

- **Numerical variables:**
  
  \[
  X_k^{*} = \hat{F}_k^{-1}(U_k^{*})
  \]

- **Categorical variables:** Synthetic categories are chosen based on quantiles defined by empirical distributions.

---

## 5. Handling Constraints and Marginal Enforcement

To ensure realism, additional constraints can be imposed:

- **Enforcing Min-Max Values:** Generated synthetic numerical values can be clipped or transformed to lie within the original data ranges, ensuring realistic values:

  \[
  x_{k,\min} \leq x_{ik}^{*} \leq x_{k,\max}
  \]

- **Distribution Enforcement:** By carefully choosing marginal distributions (empirical KDE, parametric distributions), synthetic data closely match original distributions.

---

## 6. Evaluating Synthetic Data Quality

The quality and realism of synthetic data generated via Gaussian Copula methods can be evaluated using statistical metrics, including:

- **Kolmogorov-Smirnov (KS) Test:**  
  Evaluates similarity between marginal distributions of synthetic and real data.
  \[
  KS = \sup_x|F_{real}(x) - F_{synthetic}(x)|
  \]

- **Correlation Similarity:**  
  Comparing synthetic correlation matrix \(\Sigma^{*}\) with original correlation \(\Sigma\) to ensure the dependency structure is maintained:

  \[
  \text{CorrDiff} = \|\Sigma - \Sigma^{*}\|_F
  \]

where \(\|\cdot\|_F\) denotes the Frobenius norm.

---

## 7. Advantages of Gaussian Copula

- **Flexibility:** Separates modeling of marginals from dependencies, offering significant flexibility.
- **Scalability:** Efficiently handles large datasets through optimized correlation computation and chunked processing.
- **Realistic dependencies:** Captures complex multivariate relationships among variables.

---

## 8. Mathematical Summary

In summary, the Gaussian Copula synthetic data generation method mathematically involves:

- Estimating marginal distributions and transforming data to uniform distributions.
- Modeling dependencies via Gaussian correlations in transformed Gaussian space.
- Generating new samples from a multivariate Gaussian distribution and transforming them back to the original marginal distributions.

This rigorous statistical approach ensures synthetic data faithfully represents both individual variable distributions and their complex interdependencies.

---