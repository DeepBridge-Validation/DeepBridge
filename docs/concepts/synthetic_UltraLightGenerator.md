# Explanation of Synthetic Data Generation Method

The synthetic data generation method employed by **UltraLightGenerator** relies on fundamental statistical techniques. Below is a detailed mathematical description without direct reference to code-specific functions.

---

## 1. General Definitions

Consider an original dataset \( X \), composed of \( N \) observations and \( K \) variables:

\[
X = \{x_{ik}\}, \quad i = 1,\dots,N; \quad k = 1,\dots,K
\]

Each variable can be classified as either:

- **Continuous numerical**: \( x_{ik} \in \mathbb{R} \)
- **Categorical**: \( x_{ik} \in \{c_1, c_2, \dots, c_m\} \), with \( m \) distinct categories.

---

## 2. Modeling Numerical Variables

Each numerical variable \( X_k \) is considered to follow a probability distribution approximated by basic statistical parameters estimated from original data. Initially, for each numerical variable, the following statistics are computed:

- **Sample mean**:
\[
\mu_k = \frac{1}{N}\sum_{i=1}^{N} x_{ik}
\]

- **Sample standard deviation**:
\[
\sigma_k = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_{ik}-\mu_k)^2}
\]

- **Observed minimum and maximum**:
\[
x_{k,\min} = \min(x_{1k}, x_{2k}, \dots, x_{Nk}), \quad x_{k,\max} = \max(x_{1k}, x_{2k}, \dots, x_{Nk})
\]

### **Generation Procedure for Numerical Data:**

To generate synthetic numerical data \( x_{ik}^{*} \), a truncated normal distribution is used within observed data limits:

\[
x_{ik}^{*} \sim \mathcal{N}(\mu_k, \sigma_k^2), \quad \text{subject to} \quad x_{k,\min} \leq x_{ik}^{*} \leq x_{k,\max}
\]

In practice, this is performed by generating a random number from the normal distribution, then clipping to observed limits:

\[
x_{ik}^{*} = \max\left[x_{k,\min}, \min\left(x_{k,\max}, z_{ik}\right)\right], \quad z_{ik} \sim \mathcal{N}(\mu_k,\sigma_k^2)
\]

This ensures synthetic values remain realistic and bounded.

---

## 3. Modeling Categorical Variables

Categorical variables are modeled as discrete random variables defined by empirical probability distributions based on original data.

Let the possible categories of a categorical variable \( X_j \) be:

\[
C_j = \{c_{j1}, c_{j2}, \dots, c_{jm_j}\}
\]

The relative frequency (empirical probability) of each category \( c_{jl} \) is estimated as:

\[
p(c_{jl}) = \frac{f(c_{jl})}{N}, \quad \text{where } f(c_{jl}) = \text{absolute frequency of category } c_{jl}
\]

To avoid zero probabilities, smoothing is applied:

\[
p_s(c_{jl}) = \frac{f(c_{jl}) + \alpha}{N + m_j \cdot \alpha}, \quad \text{where } 0 < \alpha \ll 1
\]

### **Generation Procedure for Categorical Data:**

Synthetic categorical value \( x_{ij}^{*} \) is sampled from a multinomial distribution defined by smoothed probabilities:

\[
x_{ij}^{*} \sim \text{Multinomial}\left(1, [p_s(c_{j1}), p_s(c_{j2}), \dots, p_s(c_{jm_j})]\right)
\]

Simply put, categories are randomly selected according to their smoothed empirical probabilities.

---

## 4. Similarity Evaluation

To evaluate how similar the synthetic data is to the original data, a normalized Euclidean distance metric is used after transforming data into a standardized numerical space.

Given a subset of selected variables \( \mathbf{x}_i = [x_{i1}, x_{i2}, \dots, x_{id}] \), normalized (e.g., using Min-Max scaling to [0,1]), the distance between a synthetic sample \(\mathbf{x}^{*}\) and the nearest original sample \(\mathbf{x}\) is calculated as:

\[
d(\mathbf{x}^{*}, \mathbf{x}) = \sqrt{\sum_{k=1}^{d}(x_k^{*}-x_k)^2}
\]

This metric ensures synthetic samples maintain a proper distance from the original data, thus preserving privacy and diversity.

---

## 5. Generation with Constraints and Novelty

### **Generation with Constraints:**

To generate data respecting a constraint for numerical variable \( X_k \), we fix a desired value \( x_k^c \):

\[
x_{ik}^{*} = x_k^{c} + \epsilon_{ik}, \quad \epsilon_{ik} \sim \mathcal{N}(0, (\sigma_k \cdot \delta)^2)
\]

where \( 0 \leq \delta \ll 1 \), controlling how strictly the constraint is enforced.

### **Generation with Novelty:**

To introduce novel patterns, perturbations are added to the standard synthetic values:

\[
x_{ik}^{*} = x_{ik}^{0} + \eta_{ik}, \quad \eta_{ik} \sim \mathcal{N}(0, (\sigma_k \cdot \gamma)^2)
\]

where:

- \( x_{ik}^{0} \) is the initially generated synthetic value.
- \( 0 < \gamma \leq 1 \) controls the desired degree of novelty.

---

## 6. Mathematical Conclusion

In summary, the UltraLightGenerator methodology involves straightforward yet effective statistical approaches, including:

- Parameter estimation from original data (mean, standard deviation, categorical distributions).
- Generation of synthetic samples via probability distributions derived from original parameters.
- Evaluation using normalized Euclidean distance to guarantee diversity and similarity.

These techniques ensure that the synthetic data generated maintains essential statistical properties of the original dataset while ensuring computational efficiency and scalability.

---