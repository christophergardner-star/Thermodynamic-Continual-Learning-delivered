"""
tar_lab/stat_utils.py
=====================
Single, authoritative source for all statistical inference in the TAR
Thermodynamic Continual Learning research system.

Design rationale
----------------
Continual-learning claims require PhD-standard statistical rigour:

  1. Normality is tested before choosing a parametric or non-parametric
     test (Shapiro & Wilk, 1965). At n < 8 the test has no meaningful
     power, so the decision is deferred.

  2. Non-parametric tests (Wilcoxon signed-rank, Mann-Whitney U) are
     preferred as primary evidence because forgetting distributions are
     often skewed and sample sizes are small (n = 5-10 seeds).

  3. Familywise error is controlled via Bonferroni correction (simple)
     or the Holm step-down procedure (more powerful). Both are exposed.

  4. Effect size (Cohen's d) is always reported alongside p-values so
     that statistical and practical significance can be distinguished.

  5. Sequential analysis is supported via Wald's SPRT so that the
     system can make early-stopping decisions without inflating Type I
     error.

  6. Bayesian posterior probability supplements frequentist p-values for
     paper reporting, using a t-posterior approximation (no MCMC).

Dependency policy
-----------------
scipy and statsmodels are imported *inside* every function that needs
them.  The module is therefore importable even when those packages are
absent; affected functions return None / NaN and set ci_method to an
informative string rather than raising ImportError.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    # dataclasses
    "CIResult",
    "BonferroniResult",
    "PowerResult",
    "ComparisonResult",
    "SPRTResult",
    "BayesianResult",
    # functions
    "compute_ci95",
    "bonferroni_correct",
    "holm_bonferroni",
    "power_analysis",
    "required_seeds",
    "compare_methods_paired",
    "compare_methods_independent",
    "sprt_boundary",
    "bayesian_evidence",
    "apply_bonferroni_to_comparison",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CIResult:
    """Result of compute_ci95."""

    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    ci_method: str                  # "t_distribution" | "bootstrap" | "unavailable"
    p_shapiro: Optional[float]      # None when n < 8 or scipy absent
    is_normal: Optional[bool]       # None when p_shapiro is None


@dataclass(frozen=True)
class BonferroniResult:
    """Result of bonferroni_correct."""

    k: int
    threshold: float
    p_values: List[float]
    significant: List[bool]
    method: str = "bonferroni"


@dataclass(frozen=True)
class PowerResult:
    """Result of power_analysis."""

    effect_size_d: float
    n_seeds: int
    alpha: float
    achieved_power: Optional[float]
    seeds_needed_80pct: Optional[int]
    seeds_needed_90pct: Optional[int]


@dataclass(frozen=True)
class ComparisonResult:
    """Result of compare_methods_paired / compare_methods_independent."""

    test_type: str                          # "paired" | "independent"
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float                       # mean(a) - mean(b)
    p_nonparametric: float
    statistic_nonparametric: float
    test_nonparametric: str
    p_parametric: float
    statistic_parametric: float
    test_parametric: str
    cohens_d: float
    ci_result: CIResult                     # CI on deltas (or a for independent)
    bonferroni_k: int = 1
    bonferroni_threshold: float = 0.05
    bonferroni_significant: bool = False


@dataclass(frozen=True)
class SPRTResult:
    """Result of sprt_boundary."""

    n: int
    decision: str                           # "accept_H1" | "accept_H0" | "continue"
    log_lr: float
    A: float                                # upper boundary
    B: float                                # lower boundary


@dataclass(frozen=True)
class BayesianResult:
    """Result of bayesian_evidence."""

    posterior_p_better: float
    posterior_mean_delta: float
    credible_interval_95: tuple                # (low, high)
    interpretation: str


# ---------------------------------------------------------------------------
# 1. compute_ci95
# ---------------------------------------------------------------------------

def compute_ci95(
    values: Sequence[float],
    alpha: float = 0.05,
) -> CIResult:
    """
    Compute a 95 % confidence interval for the mean of *values*.

    Algorithm
    ---------
    - Compute mean and sample standard deviation (ddof=1).
    - If n >= 8: apply the Shapiro-Wilk normality test (Shapiro & Wilk,
      1965, *Biometrika* 52(3):591-611).  p_shapiro > 0.05 is treated as
      consistent with normality (is_normal=True).
    - If n < 8: the Shapiro-Wilk test has insufficient power; p_shapiro
      and is_normal are set to None and the t-distribution CI is used.
    - When is_normal is False (confirmed non-normal at n >= 8): use a
      bootstrap percentile CI (Efron, 1979, *Ann. Stat.* 7(1):1-26) with
      1 000 resamples and a fixed RNG seed of 42 for reproducibility.
    - Otherwise: use the t-distribution CI (Student, 1908).

    Parameters
    ----------
    values : sequence of float
        Observations.  Must contain at least 2 elements.
    alpha : float
        Significance level; default 0.05 yields a 95 % CI.

    Returns
    -------
    CIResult

    Raises
    ------
    ValueError
        If n < 2.
    """
    vals = list(values)
    n = len(vals)
    if n < 2:
        raise ValueError(
            f"compute_ci95 requires at least 2 observations; got {n}."
        )

    mean_v = sum(vals) / n
    variance = sum((x - mean_v) ** 2 for x in vals) / (n - 1)
    std_v = math.sqrt(variance)

    # Shapiro-Wilk normality test
    p_shapiro: Optional[float] = None
    is_normal: Optional[bool] = None

    if n >= 8:
        try:
            from scipy import stats as _scipy_stats  # noqa: PLC0415
            _stat_sw, p_sw = _scipy_stats.shapiro(vals)
            p_shapiro = float(p_sw)
            is_normal = p_shapiro > 0.05
        except ImportError:
            pass  # scipy absent — leave p_shapiro and is_normal as None

    # Choose CI method
    if is_normal is False:
        # Bootstrap percentile CI (Efron, 1979)
        try:
            import numpy as _np  # noqa: PLC0415
            rng = _np.random.default_rng(seed=42)
            arr = _np.array(vals, dtype=float)
            boot_means = _np.array([
                rng.choice(arr, size=n, replace=True).mean()
                for _ in range(1000)
            ])
            ci_low = float(_np.percentile(boot_means, 100 * alpha / 2))
            ci_high = float(_np.percentile(boot_means, 100 * (1 - alpha / 2)))
            ci_method = "bootstrap"
        except ImportError:
            # numpy absent — fall back to t-distribution
            ci_low, ci_high, ci_method = _t_distribution_ci(
                mean_v, std_v, n, alpha
            )
    else:
        ci_low, ci_high, ci_method = _t_distribution_ci(mean_v, std_v, n, alpha)

    return CIResult(
        mean=mean_v,
        std=std_v,
        n=n,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=ci_method,
        p_shapiro=p_shapiro,
        is_normal=is_normal,
    )


def _t_distribution_ci(
    mean_v: float,
    std_v: float,
    n: int,
    alpha: float,
) -> tuple:
    """
    Return (ci_low, ci_high, ci_method) using the t-distribution.

    Reference: Student (1908), *Biometrika* 6(1):1-25.
    """
    try:
        from scipy import stats as _scipy_stats  # noqa: PLC0415
        t_crit = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    except ImportError:
        # Fallback: use z = 1.96 for alpha=0.05 (crude but importable)
        t_crit = 1.96
    half = t_crit * std_v / math.sqrt(n)
    return mean_v - half, mean_v + half, "t_distribution"


# ---------------------------------------------------------------------------
# 2. bonferroni_correct
# ---------------------------------------------------------------------------

def bonferroni_correct(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> BonferroniResult:
    """
    Apply Bonferroni correction for multiple comparisons.

    The family-wise error rate (FWER) is controlled by dividing alpha by
    the number of simultaneous tests k (Dunn, 1961, *J. Am. Stat. Assoc.*
    56(293):52-64; Bonferroni, 1936).

    Parameters
    ----------
    p_values : sequence of float
    alpha : float
        Uncorrected significance level; default 0.05.

    Returns
    -------
    BonferroniResult
    """
    p_list = list(p_values)
    k = len(p_list)
    if k == 0:
        raise ValueError("p_values must be non-empty.")
    threshold = alpha / k
    significant = [p < threshold for p in p_list]
    return BonferroniResult(
        k=k,
        threshold=threshold,
        p_values=p_list,
        significant=significant,
        method="bonferroni",
    )


# ---------------------------------------------------------------------------
# 3. holm_bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Holm step-down procedure for multiple testing correction.

    More powerful than the standard Bonferroni correction while still
    controlling the FWER.  Reference: Holm (1979), *Scand. J. Stat.*
    6(2):65-70.

    Algorithm
    ---------
    1. Sort p-values ascending; record original indices.
    2. For rank r (0-indexed), the threshold is alpha / (k - r).
    3. A test is significant if it and all lower-ranked tests also pass
       their respective thresholds (step-down: stop at the first failure
       and mark all remaining tests non-significant).

    Parameters
    ----------
    p_values : sequence of float
    alpha : float

    Returns
    -------
    list[bool] in original index order.
    """
    p_list = list(p_values)
    k = len(p_list)
    if k == 0:
        raise ValueError("p_values must be non-empty.")

    # Sort by p-value ascending; track original indices
    indexed = sorted(enumerate(p_list), key=lambda x: x[1])
    significant = [False] * k
    stopped = False

    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (k - rank)
        if stopped or p >= threshold:
            stopped = True
            significant[orig_idx] = False
        else:
            significant[orig_idx] = True

    return significant


# ---------------------------------------------------------------------------
# Power helpers (scipy non-central t — no statsmodels dependency)
# ---------------------------------------------------------------------------

def _nct_power(d_abs: float, n: int, alpha: float = 0.05) -> float:
    """
    Power of a one-tailed one-sample t-test (alternative='less') via the
    non-central t distribution.

    Under H1: delta = -d_abs * sqrt(n)  (negative NCP shifts left)
    Power = P(T < t_crit | ncp)  where t_crit = t.ppf(alpha, df=n-1).

    Reference: Lehmann & Romano (2005), *Testing Statistical Hypotheses*,
    Springer, Ch. 5.
    """
    from scipy import stats as _sc  # noqa: PLC0415
    df = n - 1
    t_crit = float(_sc.t.ppf(alpha, df=df))          # negative threshold
    ncp = -d_abs * math.sqrt(n)                       # shift distribution left
    return float(_sc.nct.cdf(t_crit, df=df, nc=ncp))


def _solve_n_for_power(
    d_abs: float,
    power_target: float = 0.80,
    alpha: float = 0.05,
    max_n: int = 500,
) -> int:
    """
    Binary search for the minimum n achieving *power_target* at given d_abs.
    Returns max_n if the target is not reachable within the search range.
    """
    for n in range(2, max_n + 1):
        if _nct_power(d_abs, n, alpha) >= power_target:
            return n
    return max_n


# ---------------------------------------------------------------------------
# 4. power_analysis
# ---------------------------------------------------------------------------

def power_analysis(
    effect_size_d: float,
    n_seeds: int,
    alpha: float = 0.05,
) -> PowerResult:
    """
    Compute achieved statistical power and sample sizes for target power.

    Uses a one-sample t-test power model (TTestOneSamplePower from
    statsmodels) with alternative='smaller', matching the CL forgetting
    hypothesis that TCL forgetting is *less than* the comparator.

    Reference: Cohen (1988), *Statistical Power Analysis for the
    Behavioral Sciences* (2nd ed.), Lawrence Erlbaum Associates.

    Parameters
    ----------
    effect_size_d : float
        Cohen's d for the expected effect.
    n_seeds : int
        Number of seeds in the current study arm.
    alpha : float
        Type I error rate; default 0.05.

    Returns
    -------
    PowerResult
        achieved_power, seeds_needed_80pct, and seeds_needed_90pct are
        None when statsmodels is unavailable.
    """
    abs_d = abs(effect_size_d)
    try:
        achieved_power = _nct_power(abs_d, n_seeds, alpha)
        seeds_80 = max(5, _solve_n_for_power(abs_d, 0.80, alpha))
        seeds_90 = max(5, _solve_n_for_power(abs_d, 0.90, alpha))
    except Exception:
        return PowerResult(
            effect_size_d=effect_size_d,
            n_seeds=n_seeds,
            alpha=alpha,
            achieved_power=None,
            seeds_needed_80pct=None,
            seeds_needed_90pct=None,
        )

    return PowerResult(
        effect_size_d=effect_size_d,
        n_seeds=n_seeds,
        alpha=alpha,
        achieved_power=achieved_power,
        seeds_needed_80pct=seeds_80,
        seeds_needed_90pct=seeds_90,
    )


# ---------------------------------------------------------------------------
# 5. required_seeds
# ---------------------------------------------------------------------------

def required_seeds(
    target_effect_size_d: float,
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> int:
    """
    Minimum number of seeds to achieve *power_target* for the given
    Cohen's d, using a one-sample t-test (alternative='smaller').

    Reference: Cohen (1988), *Statistical Power Analysis for the
    Behavioral Sciences* (2nd ed.).

    Returns
    -------
    int
        max(5, ceil(n)).  Returns 5 (the minimum useful seed count) when
        statsmodels is unavailable.
    """
    try:
        return max(5, _solve_n_for_power(abs(target_effect_size_d), power_target, alpha))
    except Exception:
        return 5


# ---------------------------------------------------------------------------
# 6. compare_methods_paired
# ---------------------------------------------------------------------------

def compare_methods_paired(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "less",
) -> ComparisonResult:
    """
    Paired statistical comparison: a vs b (same subjects / seeds).

    Intended use: a = TCL forgetting scores, b = comparator forgetting
    scores across matched seeds.  alternative='less' tests H1: TCL < comparator
    (i.e. TCL has lower forgetting = better).

    Non-parametric test
    -------------------
    Wilcoxon signed-rank test (Wilcoxon, 1945, *Biometrics Bulletin*
    1(6):80-83).  Handles tied ranks and is the standard non-parametric
    paired test for small samples.  Edge case: if all pairwise differences
    are zero, p = 1.0 and statistic = 0.0.

    Parametric test
    ---------------
    Paired t-test (Student, 1908, *Biometrika* 6(1):1-25).

    Effect size
    -----------
    Cohen's d on the paired differences: mean(d) / std(d, ddof=1).
    If std(d) = 0, d = 0.

    Multiple comparisons
    --------------------
    bonferroni_k defaults to 1 (no correction).  Use
    apply_bonferroni_to_comparison to update for k simultaneous tests.
    """
    a_list = list(a)
    b_list = list(b)
    n = len(a_list)
    if n != len(b_list):
        raise ValueError("a and b must have the same length for paired comparison.")
    if n < 2:
        raise ValueError("compare_methods_paired requires at least 2 paired observations.")

    deltas = [ai - bi for ai, bi in zip(a_list, b_list)]
    mean_a = sum(a_list) / n
    mean_b = sum(b_list) / n
    mean_delta = sum(deltas) / n

    # Non-parametric: Wilcoxon signed-rank
    try:
        from scipy import stats as _scipy_stats  # noqa: PLC0415
        try:
            stat_np, p_np = _scipy_stats.wilcoxon(
                a_list, b_list, alternative=alternative
            )
            stat_np = float(stat_np)
            p_np = float(p_np)
        except ValueError:
            # All differences are zero
            stat_np = 0.0
            p_np = 1.0
    except ImportError:
        stat_np = float("nan")
        p_np = float("nan")

    # Parametric: paired t-test
    try:
        from scipy import stats as _scipy_stats  # noqa: PLC0415
        stat_t, p_t = _scipy_stats.ttest_rel(
            a_list, b_list, alternative=alternative
        )
        stat_t = float(stat_t)
        p_t = float(p_t)
    except ImportError:
        stat_t = float("nan")
        p_t = float("nan")

    # Cohen's d on deltas
    cohens_d = _cohens_d_paired(deltas)

    # CI on deltas
    ci_result = compute_ci95(deltas)

    bonferroni_threshold = 0.05
    bonferroni_significant = p_np < bonferroni_threshold

    return ComparisonResult(
        test_type="paired",
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=mean_delta,
        p_nonparametric=p_np,
        statistic_nonparametric=stat_np,
        test_nonparametric="wilcoxon_signed_rank",
        p_parametric=p_t,
        statistic_parametric=stat_t,
        test_parametric="paired_t_test",
        cohens_d=cohens_d,
        ci_result=ci_result,
        bonferroni_k=1,
        bonferroni_threshold=bonferroni_threshold,
        bonferroni_significant=bonferroni_significant,
    )


def _cohens_d_paired(deltas: List[float]) -> float:
    """Cohen's d for paired differences: mean / std(ddof=1)."""
    n = len(deltas)
    mean_d = sum(deltas) / n
    if n < 2:
        return 0.0
    variance = sum((x - mean_d) ** 2 for x in deltas) / (n - 1)
    std_d = math.sqrt(variance)
    if std_d == 0.0:
        return 0.0
    return mean_d / std_d


# ---------------------------------------------------------------------------
# 7. compare_methods_independent
# ---------------------------------------------------------------------------

def compare_methods_independent(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "less",
) -> ComparisonResult:
    """
    Independent-samples statistical comparison: a vs b.

    Non-parametric test
    -------------------
    Mann-Whitney U test (Mann & Whitney, 1947, *Ann. Math. Stat.*
    18(1):50-60).  Does not assume equal variances or normality.

    Parametric test
    ---------------
    Welch's independent t-test (Welch, 1947, *Biometrika* 34(1-2):28-35)
    via scipy.stats.ttest_ind (equal_var=False by default).

    Effect size
    -----------
    Cohen's d using the pooled standard deviation:
        pooled_sd = sqrt(((na-1)*var_a + (nb-1)*var_b) / (na+nb-2))
    If pooled_sd = 0, d = 0.

    CI
    --
    compute_ci95 is called on the *a* sample (the primary method under
    evaluation) since independent samples do not have pairwise deltas.
    mean_delta = mean(a) - mean(b) for reporting.
    """
    a_list = list(a)
    b_list = list(b)
    na, nb = len(a_list), len(b_list)
    if na < 2 or nb < 2:
        raise ValueError(
            "compare_methods_independent requires at least 2 observations per group."
        )

    mean_a = sum(a_list) / na
    mean_b = sum(b_list) / nb
    mean_delta = mean_a - mean_b

    # Non-parametric: Mann-Whitney U
    try:
        from scipy import stats as _scipy_stats  # noqa: PLC0415
        stat_np, p_np = _scipy_stats.mannwhitneyu(
            a_list, b_list, alternative=alternative
        )
        stat_np = float(stat_np)
        p_np = float(p_np)
    except ImportError:
        stat_np = float("nan")
        p_np = float("nan")

    # Parametric: independent t-test (Welch)
    try:
        from scipy import stats as _scipy_stats  # noqa: PLC0415
        stat_t, p_t = _scipy_stats.ttest_ind(
            a_list, b_list, alternative=alternative
        )
        stat_t = float(stat_t)
        p_t = float(p_t)
    except ImportError:
        stat_t = float("nan")
        p_t = float("nan")

    # Cohen's d (pooled SD)
    cohens_d = _cohens_d_independent(a_list, b_list, mean_a, mean_b, na, nb)

    # CI on a (the primary method)
    ci_result = compute_ci95(a_list)

    bonferroni_threshold = 0.05
    bonferroni_significant = p_np < bonferroni_threshold

    return ComparisonResult(
        test_type="independent",
        n=na + nb,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=mean_delta,
        p_nonparametric=p_np,
        statistic_nonparametric=stat_np,
        test_nonparametric="mann_whitney_u",
        p_parametric=p_t,
        statistic_parametric=stat_t,
        test_parametric="independent_t_test",
        cohens_d=cohens_d,
        ci_result=ci_result,
        bonferroni_k=1,
        bonferroni_threshold=bonferroni_threshold,
        bonferroni_significant=bonferroni_significant,
    )


def _cohens_d_independent(
    a: List[float],
    b: List[float],
    mean_a: float,
    mean_b: float,
    na: int,
    nb: int,
) -> float:
    """Cohen's d for independent samples using pooled standard deviation."""
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_sd = math.sqrt(pooled_var)
    if pooled_sd == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


# ---------------------------------------------------------------------------
# 8. sprt_boundary
# ---------------------------------------------------------------------------

def sprt_boundary(
    n_seeds_run: int,
    p_values_so_far: Sequence[float],
    alpha: float = 0.05,
    beta: float = 0.10,
    h1_effect_size: float = 0.5,
) -> SPRTResult:
    """
    Wald's Sequential Probability Ratio Test (SPRT) for early stopping.

    Reference: Wald (1947), *Sequential Analysis*, Wiley.

    Boundaries
    ----------
    - A = log((1 - beta) / alpha)    Upper boundary → accept H1
    - B = log(beta / (1 - alpha))    Lower boundary → accept H0

    Log-likelihood ratio
    --------------------
    Each seed is treated as one Bernoulli SIGN trial: ``p < alpha`` marks a
    favorable seed (one-tailed evidence for H1), otherwise unfavorable. The
    per-trial increment is the true log-likelihood ratio log(f1/f0) under

        H0: P(favorable) = 0.5            (no preference)
        H1: P(favorable) = Phi(h1_effect_size)   (min meaningful Cohen's d)

        favorable:   increment = log(p1 / 0.5)
        unfavorable: increment = log((1 - p1) / 0.5)

    so a near-even sign split sits between the boundaries (decision="continue")
    and only a genuine excess of favorable seeds crosses A. Earlier versions used
    the BOUNDARY value log((1-beta)/alpha) as the increment, which let a single
    favorable trial reach A — accepting H1 on near-even evidence (the n=12 HPC bug).

    The caller invokes this after each batch of seeds (every 4 is recommended);
    p-proxies are clamped to (0, 1) to avoid log blow-up.

    Parameters
    ----------
    n_seeds_run : int
        Total number of seeds completed so far.
    p_values_so_far : sequence of float
        One p-value per seed (or per comparison), ordered chronologically.
    alpha : float
        Target Type I error rate; default 0.05.
    beta : float
        Target Type II error rate (1 - power); default 0.10.
    h1_effect_size : float
        Minimum meaningful effect (Cohen's d) under H1; sets the favorable-trial
        probability p1 = Phi(d) that scales each per-seed LLR increment. Default
        0.5 matches the d>=0.5 success bar of the HPC replication.

    Returns
    -------
    SPRTResult
    """
    A = math.log((1.0 - beta) / alpha)       # upper boundary (accept H1)
    B = math.log(beta / (1.0 - alpha))        # lower boundary (accept H0)

    # Per-observation log-likelihood-ratio increments for the per-seed SIGN test.
    # Each seed is ONE Bernoulli trial: p < alpha == "favorable" (one-tailed
    # evidence for H1), else "unfavorable". The increment is the TRUE per-trial
    # LLR log(f1/f0) — NOT the boundary value:
    #   H0: P(favorable) = 0.5         (no preference)
    #   H1: P(favorable) = Phi(d)      (d = h1_effect_size, the minimum meaningful
    #                                   Cohen's d; a single paired delta lands
    #                                   favorable with probability Phi(d))
    # Using the boundary log((1-beta)/alpha) AS the increment (the previous
    # implementation) is a textbook SPRT error: a single favorable trial would
    # already reach boundary A, so a near-even sign split spuriously crosses it.
    # That is what stopped the HPC replication at n=12 (7 vs 5) with log_LR=8.98
    # while the effect was honestly inconclusive (d=-0.44, power=0.41).
    p0 = 0.5
    p1 = 0.5 * (1.0 + math.erf(abs(h1_effect_size) / math.sqrt(2.0)))   # Phi(d)
    p1 = min(max(p1, 0.5 + 1e-6), 1.0 - 1e-9)                           # H1 must favor success
    _succ_inc = math.log(p1 / p0)                        # favorable seed (p < alpha)
    _fail_inc = math.log((1.0 - p1) / (1.0 - p0))        # unfavorable seed

    log_lr = 0.0
    for p in p_values_so_far:
        p = max(1e-300, min(1.0 - 1e-10, float(p)))
        increment = _succ_inc if p < alpha else _fail_inc
        log_lr += increment

    if log_lr >= A:
        decision = "accept_H1"
    elif log_lr <= B:
        decision = "accept_H0"
    else:
        decision = "continue"

    return SPRTResult(
        n=n_seeds_run,
        decision=decision,
        log_lr=log_lr,
        A=A,
        B=B,
    )


# ---------------------------------------------------------------------------
# 9. bayesian_evidence
# ---------------------------------------------------------------------------

def bayesian_evidence(
    deltas: Sequence[float],
    prior_scale: float = 0.5,
) -> BayesianResult:
    """
    Bayesian posterior probability that TCL is better than the comparator.

    Uses a t-posterior approximation (no MCMC required).  The approach is
    analytically equivalent to integrating a flat (improper) prior over
    the normal likelihood, yielding a t-distributed posterior for the mean.

    Reference: Gelman et al. (2013), *Bayesian Data Analysis* (3rd ed.),
    CRC Press, Ch. 3 (normal model with unknown mean and variance).

    Convention
    ----------
    deltas = TCL_forgetting - comparator_forgetting.
    Negative delta means TCL has lower (better) forgetting.
    P(TCL better) = P(delta < 0).

    Parameters
    ----------
    deltas : sequence of float
        Paired differences (a - b).  Must contain at least 2 elements.
    prior_scale : float
        Retained as a parameter for API stability; the current
        t-posterior approximation uses a flat prior (prior_scale has no
        numerical effect in this formulation).

    Returns
    -------
    BayesianResult
    """
    d_list = list(deltas)
    n = len(d_list)
    if n < 2:
        raise ValueError("bayesian_evidence requires at least 2 observations.")

    mean_d = sum(d_list) / n
    variance_d = sum((x - mean_d) ** 2 for x in d_list) / (n - 1)
    std_d = math.sqrt(variance_d)
    se = std_d / math.sqrt(n) if std_d > 0 else 0.0
    df = n - 1

    if se == 0.0:
        t_stat = 0.0
        p_better = 0.5
        ci_half = 0.0
    else:
        t_stat = mean_d / se
        try:
            from scipy import stats as _scipy_stats  # noqa: PLC0415
            # t-posterior: mu | data ~ t(df, loc=mean_d, scale=se)
            # P(TCL better) = P(mu < 0 | data) = P(T < -mean_d/se) where T ~ t(df)
            # This is correct for any sign of mean_d — no branching needed.
            p_better = float(_scipy_stats.t.cdf(-mean_d / se, df=df))
            ci_half = float(_scipy_stats.t.ppf(0.975, df=df)) * se
        except ImportError:
            # scipy absent — normal approximation of the same posterior CDF
            p_better = _normal_cdf(-mean_d / se)
            ci_half = 1.96 * se

    credible_interval_95 = (mean_d - ci_half, mean_d + ci_half)
    interpretation = f"P(TCL better) = {p_better:.1%}"

    return BayesianResult(
        posterior_p_better=p_better,
        posterior_mean_delta=mean_d,
        credible_interval_95=credible_interval_95,
        interpretation=interpretation,
    )


def _normal_cdf(z: float) -> float:
    """
    Standard normal CDF via math.erfc (fallback when scipy is absent).
    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# 10. apply_bonferroni_to_comparison
# ---------------------------------------------------------------------------

def apply_bonferroni_to_comparison(
    result: ComparisonResult,
    k: int,
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Return a copy of *result* with Bonferroni correction applied for k
    simultaneous comparisons.

    Updates bonferroni_k, bonferroni_threshold, and bonferroni_significant.
    The primary p-value used for the significance decision is
    p_nonparametric (the non-parametric test is the primary evidence
    source in this system).

    Reference: Dunn (1961), *J. Am. Stat. Assoc.* 56(293):52-64.

    Parameters
    ----------
    result : ComparisonResult
    k : int
        Total number of simultaneous tests in the family.
    alpha : float
        Family-wise error rate; default 0.05.

    Returns
    -------
    ComparisonResult (frozen copy with updated Bonferroni fields)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    threshold = alpha / k
    significant = result.p_nonparametric < threshold
    # dataclass is frozen; use object.__setattr__ pattern via reconstruction
    return ComparisonResult(
        test_type=result.test_type,
        n=result.n,
        mean_a=result.mean_a,
        mean_b=result.mean_b,
        mean_delta=result.mean_delta,
        p_nonparametric=result.p_nonparametric,
        statistic_nonparametric=result.statistic_nonparametric,
        test_nonparametric=result.test_nonparametric,
        p_parametric=result.p_parametric,
        statistic_parametric=result.statistic_parametric,
        test_parametric=result.test_parametric,
        cohens_d=result.cohens_d,
        ci_result=result.ci_result,
        bonferroni_k=k,
        bonferroni_threshold=threshold,
        bonferroni_significant=significant,
    )
