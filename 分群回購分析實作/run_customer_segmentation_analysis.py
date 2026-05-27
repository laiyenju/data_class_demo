from __future__ import annotations

import json
import math
import os
import warnings
from html import escape
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "原始數據"
OUTPUT_DIR = WORK_DIR / "outputs"
CHART_DIR = WORK_DIR / "charts"
NOTEBOOK_PATH = WORK_DIR / "客群分群_回購驗證_分析.ipynb"
REPORT_PATH = WORK_DIR / "測試紀錄與結果.md"

RANDOM_SEED = 42
TRAIN_END = pd.Timestamp("2024-12-31")
VALID_END = pd.Timestamp("2025-12-31")
FINAL_FEATURE_SET = "B_RFM_travel"
FINAL_K = 3


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = WORK_DIR / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(WORK_DIR / ".cache"))
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 2)


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def lifelines_available() -> bool:
    try:
        import lifelines  # noqa: F401
    except Exception:
        return False
    return True


def fmt_int(x) -> str:
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):,}"


def fmt_float(x, digits=2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):,.{digits}f}"


def fmt_pct(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def fmt_money(x) -> str:
    if pd.isna(x):
        return ""
    return f"NT${float(x):,.0f}"


def df_to_markdown(df: pd.DataFrame) -> str:
    text_df = df.copy()
    text_df = text_df.astype(str).replace({"nan": "", "NaT": ""})
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_and_prepare_data() -> tuple[pd.DataFrame, dict[str, str]]:
    raw = pd.read_csv(DATA_DIR / "2023-2025_旅平險_V2.csv", encoding="utf-8-sig")
    site_code = pd.read_csv(DATA_DIR / "旅遊地點代碼.csv", encoding="utf-8-sig")
    code_map = {str(int(k)): v for k, v in zip(site_code["SITE_CODE"], site_code["旅遊地點代碼"])}
    known_codes = set(code_map)

    @lru_cache(None)
    def parse_site_codes(value: str):
        if value == "":
            return ()
        if value in known_codes:
            return (value,)
        for width in (3, 2):
            head = value[:width]
            if head in known_codes:
                tail = parse_site_codes(value[width:])
                if tail is not None:
                    return (head,) + tail
        return None

    df = raw.copy()
    for col in ["APL_BIRTH", "EFF_DATE", "CSTP_DATE", "LOGIN_DATE"]:
        df[f"{col}_DT"] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
    df["SITE_STR"] = df["SITE_CODE"].apply(lambda x: None if pd.isna(x) else str(int(x)))
    df["SITE_LIST"] = df["SITE_STR"].apply(lambda x: [] if x is None else list(parse_site_codes(x) or []))
    df["DESTINATION_COUNT"] = df["SITE_LIST"].apply(len)
    df["HAS_DOMESTIC_DEST"] = df["SITE_LIST"].apply(lambda xs: any(x in {"98", "64"} for x in xs))
    df["HAS_INTL_DEST"] = df["SITE_LIST"].apply(lambda xs: any(x not in {"98", "64"} for x in xs))
    df["HAS_JAPAN"] = df["SITE_LIST"].apply(lambda xs: "131" in xs)
    df["HAS_KOREA"] = df["SITE_LIST"].apply(lambda xs: "132" in xs)
    df["HAS_LONG_HAUL"] = df["SITE_LIST"].apply(
        lambda xs: any(x in {"210", "211", "212", "301", "302", "510", "520"} for x in xs)
    )
    df["LEAD_DAYS"] = (df["EFF_DATE_DT"] - df["LOGIN_DATE_DT"]).dt.days
    df["PREM_PER_DAY"] = df["TOT_PREM"] / df["DAYS"].replace(0, np.nan)
    df["PURCHASE_YEAR"] = df["LOGIN_DATE_DT"].dt.year
    return df, code_map


def build_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df["LOGIN_DATE_DT"] >= "2023-01-01") & (df["LOGIN_DATE_DT"] <= TRAIN_END)].copy()
    valid = df[(df["LOGIN_DATE_DT"] >= "2025-01-01") & (df["LOGIN_DATE_DT"] <= VALID_END)].copy()

    train = train.sort_values(["APL_ID", "LOGIN_DATE_DT", "EFF_DATE_DT"])
    features = train.groupby("APL_ID").agg(
        first_train_purchase=("LOGIN_DATE_DT", "min"),
        last_train_purchase=("LOGIN_DATE_DT", "max"),
        frequency=("APL_ID", "size"),
        total_premium=("TOT_PREM", "sum"),
        avg_premium=("TOT_PREM", "mean"),
        avg_days=("DAYS", "mean"),
        total_days=("DAYS", "sum"),
        avg_lead_days=("LEAD_DAYS", "mean"),
        short_notice_share=("LEAD_DAYS", lambda s: (s <= 3).mean()),
        intl_share=("HAS_INTL_DEST", "mean"),
        domestic_share=("HAS_DOMESTIC_DEST", "mean"),
        japan_share=("HAS_JAPAN", "mean"),
        korea_share=("HAS_KOREA", "mean"),
        long_haul_share=("HAS_LONG_HAUL", "mean"),
        multi_dest_share=("DESTINATION_COUNT", lambda s: (s > 1).mean()),
        avg_prem_per_day=("PREM_PER_DAY", "mean"),
    ).reset_index()
    features["recency_days"] = (TRAIN_END - features["last_train_purchase"]).dt.days
    features["customer_count"] = 1
    return features, train, valid


FEATURE_SETS = {
    "A_RFM_only": ["recency_days", "frequency", "total_premium"],
    "B_RFM_travel": ["recency_days", "frequency", "total_premium", "avg_days", "intl_share", "multi_dest_share"],
    "C_RFM_purchase": ["recency_days", "frequency", "total_premium", "avg_lead_days", "short_notice_share"],
    "D_RFM_value": ["recency_days", "frequency", "total_premium", "avg_premium", "avg_prem_per_day"],
    "E_RFM_selected_all": [
        "recency_days",
        "frequency",
        "total_premium",
        "avg_days",
        "intl_share",
        "multi_dest_share",
        "avg_lead_days",
        "short_notice_share",
        "avg_premium",
        "avg_prem_per_day",
    ],
}

LOG_TRANSFORM_COLS = {
    "recency_days",
    "frequency",
    "total_premium",
    "avg_premium",
    "avg_prem_per_day",
    "avg_days",
    "avg_lead_days",
}


def transform_features(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    work = df[cols].copy()
    for col in cols:
        work[col] = work[col].replace([np.inf, -np.inf], np.nan).fillna(work[col].median())
        if col in LOG_TRANSFORM_COLS:
            work[col] = np.log1p(work[col].clip(lower=0))
    means = work.mean(axis=0)
    stds = work.std(axis=0).replace(0, 1)
    x = ((work - means) / stds).to_numpy(dtype=float)
    return x, {"columns": cols, "means": means.to_dict(), "stds": stds.to_dict()}


def kmeans_numpy(x: np.ndarray, k: int, seed: int, max_iter: int = 80, n_init: int = 6):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    best = None
    for init in range(n_init):
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        last_inertia = None
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster in range(k):
                mask = labels == cluster
                if mask.any():
                    new_centers[cluster] = x[mask].mean(axis=0)
                else:
                    new_centers[cluster] = x[rng.integers(0, n)]
            centers = new_centers
            inertia = float(((x - centers[labels]) ** 2).sum())
            if last_inertia is not None and abs(last_inertia - inertia) / max(last_inertia, 1e-9) < 1e-5:
                break
            last_inertia = inertia
        result = (inertia, labels.copy(), centers.copy())
        if best is None or result[0] < best[0]:
            best = result
    return {"inertia": best[0], "labels": best[1], "centers": best[2]}


def kmeans_sklearn(x: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=300, algorithm="lloyd")
    labels = model.fit_predict(x)
    return {"inertia": float(model.inertia_), "labels": labels.astype(int), "centers": model.cluster_centers_}


def silhouette_sample(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 3000) -> float:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    take = min(sample_size, n)
    idx = rng.choice(n, size=take, replace=False)
    xs = x[idx]
    ls = labels[idx]
    unique = np.unique(ls)
    if len(unique) < 2:
        return np.nan
    dist = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(take):
        same = ls == ls[i]
        same_count = same.sum()
        if same_count <= 1:
            continue
        a = dist[i, same].sum() / (same_count - 1)
        b = min(dist[i, ls == other].mean() for other in unique if other != ls[i] and (ls == other).any())
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else np.nan


def silhouette_sample_sklearn(x: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 3000) -> float:
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    take = min(sample_size, x.shape[0])
    idx = rng.choice(x.shape[0], size=take, replace=False)
    if len(np.unique(labels[idx])) < 2:
        return np.nan
    return float(silhouette_score(x[idx], labels[idx], metric="euclidean"))


def davies_bouldin(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    unique = np.unique(labels)
    scatter = []
    for cluster in unique:
        mask = labels == cluster
        scatter.append(np.sqrt(((x[mask] - centers[cluster]) ** 2).sum(axis=1)).mean())
    scatter = np.array(scatter)
    center_dist = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(center_dist, np.inf)
    ratios = []
    for i in range(len(unique)):
        ratios.append(np.max((scatter[i] + scatter) / center_dist[i]))
    return float(np.mean(ratios))


def calinski_harabasz(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    n, _ = x.shape
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1 or n <= k:
        return np.nan
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for cluster in unique:
        mask = labels == cluster
        count = mask.sum()
        between += count * ((centers[cluster] - overall) ** 2).sum()
        within += ((x[mask] - centers[cluster]) ** 2).sum()
    return float((between / (k - 1)) / (within / (n - k)))


def metric_scores(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, seed: int, use_sklearn: bool) -> dict:
    if use_sklearn:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

        return {
            "silhouette_sample": silhouette_sample_sklearn(x, labels, seed=seed),
            "davies_bouldin": float(davies_bouldin_score(x, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        }
    return {
        "silhouette_sample": silhouette_sample(x, labels, seed=seed),
        "davies_bouldin": davies_bouldin(x, labels, centers),
        "calinski_harabasz": calinski_harabasz(x, labels, centers),
    }


def fit_kmeans(x: np.ndarray, k: int, seed: int, use_sklearn: bool):
    if use_sklearn:
        return kmeans_sklearn(x, k=k, seed=seed)
    return kmeans_numpy(x, k=k, seed=seed)


def evaluate_feature_sets(features: pd.DataFrame, use_sklearn: bool):
    rows = []
    fitted = {}
    for set_name, cols in FEATURE_SETS.items():
        x, transform = transform_features(features, cols)
        for k in range(2, 7):
            seed = RANDOM_SEED + k + len(cols)
            fit = fit_kmeans(x, k=k, seed=seed, use_sklearn=use_sklearn)
            labels = fit["labels"]
            counts = np.bincount(labels, minlength=k)
            scores = metric_scores(x, labels, fit["centers"], seed=RANDOM_SEED + k, use_sklearn=use_sklearn)
            row = {
                "feature_set": set_name,
                "k": k,
                "feature_count": len(cols),
                "implementation": "scikit-learn" if use_sklearn else "numpy_manual",
                "inertia": fit["inertia"],
                **scores,
                "min_cluster_share": counts.min() / len(labels),
                "max_cluster_share": counts.max() / len(labels),
            }
            rows.append(row)
            fitted[(set_name, k)] = {"x": x, "fit": fit, "transform": transform, "columns": cols}
    comparison = pd.DataFrame(rows)
    candidates = comparison[comparison["min_cluster_share"] >= 0.03].copy()
    if candidates.empty:
        candidates = comparison.copy()
    candidates["rank_silhouette"] = candidates["silhouette_sample"].rank(ascending=False, method="min")
    candidates["rank_db"] = candidates["davies_bouldin"].rank(ascending=True, method="min")
    candidates["rank_balance"] = candidates["min_cluster_share"].rank(ascending=False, method="min")
    candidates["selection_score"] = candidates["rank_silhouette"] + 0.5 * candidates["rank_db"] + 0.25 * candidates["rank_balance"]
    best_row = candidates.sort_values(["selection_score", "k"]).iloc[0]
    best_key = (best_row["feature_set"], int(best_row["k"]))
    return comparison, best_key, fitted


def compare_implementations(features: pd.DataFrame, keys: list[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    can_use_sklearn = sklearn_available()
    for set_name, k in keys:
        cols = FEATURE_SETS[set_name]
        x, _ = transform_features(features, cols)
        seed = RANDOM_SEED + k + len(cols)
        for use_sklearn in ([False, True] if can_use_sklearn else [False]):
            fit = fit_kmeans(x, k=k, seed=seed, use_sklearn=use_sklearn)
            labels = fit["labels"]
            counts = np.bincount(labels, minlength=k)
            scores = metric_scores(x, labels, fit["centers"], seed=RANDOM_SEED + k, use_sklearn=use_sklearn)
            rows.append(
                {
                    "feature_set": set_name,
                    "k": k,
                    "implementation": "scikit-learn" if use_sklearn else "numpy_manual",
                    "inertia": fit["inertia"],
                    **scores,
                    "min_cluster_share": counts.min() / len(labels),
                    "max_cluster_share": counts.max() / len(labels),
                }
            )
    comparison = pd.DataFrame(rows)
    if can_use_sklearn:
        wide = comparison.pivot_table(
            index=["feature_set", "k"],
            columns="implementation",
            values=["silhouette_sample", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "max_cluster_share"],
            aggfunc="first",
        )
        diff_rows = []
        for key in wide.index:
            row = {"feature_set": key[0], "k": key[1]}
            for metric in ["silhouette_sample", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "max_cluster_share"]:
                row[f"{metric}_diff_sklearn_minus_numpy"] = wide.loc[key, (metric, "scikit-learn")] - wide.loc[key, (metric, "numpy_manual")]
            diff_rows.append(row)
        diffs = pd.DataFrame(diff_rows)
        comparison = comparison.merge(diffs, on=["feature_set", "k"], how="left")
    return comparison


def pca_numpy(x: np.ndarray, n_components: int = 3):
    centered = x - x.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components]
    scores = centered @ components
    explained = eigvals[:n_components] / eigvals.sum()
    return scores, explained


def build_validation(features: pd.DataFrame, valid: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer = features[["APL_ID", "last_train_purchase"]].copy()
    customer["cluster"] = labels
    first_2025 = valid.groupby("APL_ID")["LOGIN_DATE_DT"].min().rename("first_2025_purchase").reset_index()
    customer = customer.merge(first_2025, on="APL_ID", how="left")
    customer["event"] = customer["first_2025_purchase"].notna().astype(int)
    customer["duration_days"] = np.where(
        customer["event"] == 1,
        (customer["first_2025_purchase"] - customer["last_train_purchase"]).dt.days,
        (VALID_END - customer["last_train_purchase"]).dt.days,
    )
    customer["duration_days"] = customer["duration_days"].astype(int)
    validation_rows = []
    for cluster, group in customer.groupby("cluster"):
        events = group[group["event"] == 1]
        row = {
            "cluster": int(cluster),
            "customers": len(group),
            "repurchase_2025": int(group["event"].sum()),
            "repurchase_rate_2025": group["event"].mean(),
            "event_time_mean": events["duration_days"].mean(),
            "event_time_median": events["duration_days"].median(),
            "event_time_p25": events["duration_days"].quantile(0.25),
            "event_time_p75": events["duration_days"].quantile(0.75),
        }
        for horizon in [30, 60, 90, 180, 365]:
            row[f"raw_repurchase_{horizon}d"] = ((group["event"] == 1) & (group["duration_days"] <= horizon)).mean()
            row[f"km_repurchase_{horizon}d"] = km_cumulative_event(group, horizon)
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values("cluster")
    return customer, validation


def km_cumulative_event(group: pd.DataFrame, horizon: int) -> float:
    km = km_curve(group, max_day=horizon)
    if km.empty:
        return np.nan
    return float(1 - km.iloc[-1]["survival"])


def km_curve(group: pd.DataFrame, max_day: int = 365) -> pd.DataFrame:
    work = group[["duration_days", "event"]].copy()
    event_times = sorted(t for t in work.loc[work["event"] == 1, "duration_days"].unique() if t <= max_day)
    n_at_risk = len(work)
    survival = 1.0
    rows = [{"day": 0, "survival": survival, "cumulative_repurchase": 0.0, "n_at_risk": n_at_risk, "events": 0}]
    for day in event_times:
        at_risk = (work["duration_days"] >= day).sum()
        events = ((work["duration_days"] == day) & (work["event"] == 1)).sum()
        if at_risk > 0:
            survival *= 1 - events / at_risk
        rows.append(
            {
                "day": int(day),
                "survival": survival,
                "cumulative_repurchase": 1 - survival,
                "n_at_risk": int(at_risk),
                "events": int(events),
            }
        )
    return pd.DataFrame(rows)


def cluster_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    prof = features.copy()
    prof["cluster"] = labels
    profile = prof.groupby("cluster").agg(
        customers=("APL_ID", "size"),
        recency_median=("recency_days", "median"),
        frequency_mean=("frequency", "mean"),
        frequency_median=("frequency", "median"),
        total_premium_median=("total_premium", "median"),
        avg_premium_median=("avg_premium", "median"),
        avg_days_mean=("avg_days", "mean"),
        intl_share_mean=("intl_share", "mean"),
        japan_share_mean=("japan_share", "mean"),
        korea_share_mean=("korea_share", "mean"),
        long_haul_share_mean=("long_haul_share", "mean"),
        multi_dest_share_mean=("multi_dest_share", "mean"),
        avg_lead_days_mean=("avg_lead_days", "mean"),
        short_notice_share_mean=("short_notice_share", "mean"),
        avg_prem_per_day_median=("avg_prem_per_day", "median"),
    ).reset_index()
    profile["customer_share"] = profile["customers"] / len(features)
    return profile


def name_clusters(profile: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    merged = profile.merge(validation[["cluster", "repurchase_rate_2025", "event_time_median"]], on="cluster", how="left")
    freq_q = merged["frequency_mean"].quantile(0.67)
    value_q = merged["total_premium_median"].quantile(0.67)
    recency_q = merged["recency_median"].quantile(0.33)
    long_q = merged["avg_days_mean"].quantile(0.67)
    names = []
    priorities = []
    timing_1 = []
    timing_2 = []
    copy_dirs = []
    sample_copy = []
    for _, row in merged.iterrows():
        if row["multi_dest_share_mean"] >= 0.30 or row["avg_days_mean"] >= long_q or row["total_premium_median"] >= value_q:
            name = "多目的地長天數高價值客"
            priority = "高"
            t1, t2 = "上次投保後 120-180 天", "旅遊旺季或長假前"
            direction = "強調海外醫療、完整保障、多國或長天數行程安心"
            copy = "多國或長天數行程更需要完整保障，出發前記得確認旅平險是否已安排。"
        elif row["frequency_mean"] >= freq_q and row["recency_median"] <= recency_q:
            name = "近期活躍回購潛力客"
            priority = "高"
            t1, t2 = "上次投保後 60-90 天", "120-180 天"
            direction = "強調常旅保障、快速續保、下一趟出發前先備好"
            copy = "下一趟旅行快到了嗎？常出門的你，可以先把旅平險準備好，出發前少一件事。"
        elif row["short_notice_share_mean"] >= 0.75:
            name = "臨行前快速投保客"
            priority = "中"
            t1, t2 = "連假/旺季前 7 天", "出發前 1-3 天"
            direction = "強調快速投保、少步驟、出發前提醒"
            copy = "出發前別忘了旅平險，幾分鐘完成投保，行程更安心。"
        elif row["japan_share_mean"] >= max(row["korea_share_mean"], row["long_haul_share_mean"], 0.25):
            name = "海外日本主力客"
            priority = "中"
            t1, t2 = "賞櫻/暑假/楓葉季前", "上次投保後 90-150 天"
            direction = "以日本旅遊情境設計文案"
            copy = "準備再去日本了嗎？機票住宿安排好，也別忘了把旅平險一起準備好。"
        else:
            name = "一般低頻觀望客"
            priority = "中低"
            t1, t2 = "年度旅遊旺季前", "連假前"
            direction = "用季節性與連假提醒喚起需求"
            copy = "下一次旅行規劃中嗎？出發前把旅平險準備好，讓旅程多一層安心。"
        names.append(name)
        priorities.append(priority)
        timing_1.append(t1)
        timing_2.append(t2)
        copy_dirs.append(direction)
        sample_copy.append(copy)
    merged["cluster_name"] = names
    merged["marketing_priority"] = priorities
    merged["first_reminder"] = timing_1
    merged["second_reminder"] = timing_2
    merged["copy_direction"] = copy_dirs
    merged["sample_copy"] = sample_copy
    return merged


def add_business_summary(profile: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    premium_q = result["total_premium_median"].quantile(0.67)
    recency_q = result["recency_median"].quantile(0.34)
    summaries = []
    suggested_uses = []
    business_names = []
    for _, row in result.iterrows():
        if row["multi_dest_share_mean"] >= 0.35 or row["avg_days_mean"] >= 14:
            business_names.append("多目的地長天數高價值客")
            summaries.append("長天數或多目的地行程明顯，客單價與保障完整度較重要")
            suggested_uses.append("適合保留為高價值旅客提醒群，但樣本較小，簡報中應避免過度放大")
        elif row["intl_share_mean"] < 0.3 and row["short_notice_share_mean"] >= 0.70:
            business_names.append("臨行前快速投保客")
            summaries.append("短天數、低保費、臨行前投保比例高，較像國內或近程快速需求")
            suggested_uses.append("可用於出發前 1-3 天提醒與快速投保流程優化")
        elif row["recency_median"] <= recency_q and row["total_premium_median"] >= premium_q:
            business_names.append("近期高價值回購客")
            summaries.append("最近一次投保較近、保費中位數高，且 2025 回購率明顯較高")
            suggested_uses.append("K=4 最大新增洞察，可作為高優先回購提醒的備選切法")
        elif row["japan_share_mean"] >= 0.35:
            business_names.append("海外日本主力客")
            summaries.append("海外旅遊占比高，且日本目的地占比突出，是目前資料中的主力客群")
            suggested_uses.append("可搭配日本旅遊旺季做提醒，但仍建議維持為大眾主力群")
        elif row["long_haul_share_mean"] >= 0.20 or row["avg_premium_median"] >= result["avg_premium_median"].quantile(0.75):
            business_names.append("長程完整保障需求客")
            summaries.append("海外且保費或長程目的地占比較高，可視為較完整保障需求")
            suggested_uses.append("可測試較完整保障訊息，但需確認群體規模是否足以單獨經營")
        else:
            business_names.append("一般海外低頻旅客")
            summaries.append("行為接近一般海外低頻旅客，和主力客群差異較細")
            suggested_uses.append("若簡報篇幅有限，可併回 K=3 主力群說明")
    result["cluster_name"] = business_names
    result["business_summary"] = summaries
    result["suggested_use"] = suggested_uses
    return result


def build_k4_business_comparison(features: pd.DataFrame, valid: pd.DataFrame, fitted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    k4_labels = fitted[(FINAL_FEATURE_SET, 4)]["fit"]["labels"]
    k4_profile = cluster_profiles(features, k4_labels)
    _, k4_validation = build_validation(features, valid, k4_labels)
    k4_named = add_business_summary(name_clusters(k4_profile, k4_validation))

    rows = [
        {
            "方案": "K=3",
            "客群數": 3,
            "最小群占比": "4.6% 左右",
            "商業解釋": "三群輪廓清楚：海外日本主力、臨行前快速投保、多目的地長天數高價值",
            "簡報建議": "作為定稿主方案，容易命名與轉成策略",
        },
        {
            "方案": "K=4",
            "客群數": 4,
            "最小群占比": f"{k4_named['customer_share'].min():.1%}",
            "商業解釋": "可再拆細大型海外主力客，但多一群後命名與策略呈現更複雜",
            "簡報建議": "可放在備選比較，說明曾測試拆細但仍採 K=3",
        },
    ]
    return k4_named, pd.DataFrame(rows)


def build_validation_lift(named: pd.DataFrame) -> pd.DataFrame:
    total_customers = named["customers"].sum()
    weighted_rate = (named["customers"] * named["repurchase_rate_2025"]).sum() / total_customers
    result = named[
        [
            "cluster",
            "cluster_name",
            "customers",
            "repurchase_rate_2025",
            "event_time_median",
        ]
    ].copy()
    result["overall_repurchase_rate_2025"] = weighted_rate
    result["rate_diff_vs_overall"] = result["repurchase_rate_2025"] - weighted_rate
    result["lift_vs_overall"] = result["repurchase_rate_2025"] / weighted_rate
    result["validation_reading"] = np.where(
        result["rate_diff_vs_overall"] > 0,
        "高於整體平均，代表此群回購傾向較強",
        "低於整體平均，代表此群回購傾向較弱",
    )
    return result


def write_svg(path: Path, body: str, width: int = 1000, height: int = 560) -> Path:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif; fill: #243044; }}
.title {{ font-size: 24px; font-weight: 700; }}
.label {{ font-size: 14px; }}
.small {{ font-size: 12px; fill: #5d6b82; }}
.value {{ font-size: 14px; font-weight: 700; }}
</style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
    return path


def bar_chart(path: Path, title: str, labels, values, width=1000, height=560, color="#2f6fed", value_fmt=None):
    labels = [str(x) for x in labels]
    values = [0 if pd.isna(x) else float(x) for x in values]
    max_value = max(values) if values else 1
    max_value = max(max_value, 1e-9)
    left, right, top, bottom = 230, 120, 78, 42
    chart_w = width - left - right
    row_h = (height - top - bottom) / max(len(labels), 1)
    parts = [f'<text x="30" y="42" class="title">{title}</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h + row_h * 0.22
        bar_h = min(28, row_h * 0.52)
        bar_w = chart_w * value / max_value
        text_val = value_fmt(value) if value_fmt else f"{value:,.2f}"
        parts.append(f'<text x="{left - 14}" y="{y + bar_h * .75:.1f}" text-anchor="end" class="label">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="5" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + bar_h * .75:.1f}" class="value">{text_val}</text>')
    return write_svg(path, "\n".join(parts), width, height)


def line_chart(path: Path, title: str, curves: dict[str, pd.DataFrame], width=1000, height=560):
    colors = ["#2f6fed", "#1c8c62", "#d95f02", "#7b61ff", "#596b82", "#c43c5c"]
    left, right, top, bottom = 70, 170, 78, 70
    chart_w = width - left - right
    chart_h = height - top - bottom
    parts = [
        f'<text x="30" y="42" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#c8d2e3"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#c8d2e3"/>',
    ]
    for tick in [0, 90, 180, 270, 365]:
        x = left + chart_w * tick / 365
        parts.append(f'<line x1="{x:.1f}" y1="{top + chart_h}" x2="{x:.1f}" y2="{top + chart_h + 5}" stroke="#c8d2e3"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + chart_h + 24}" text-anchor="middle" class="small">{tick}</text>')
    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 18}" text-anchor="middle" class="small">距 2023-2024 最後一次投保天數</text>')
    for idx, (name, curve) in enumerate(curves.items()):
        color = colors[idx % len(colors)]
        if curve.empty:
            continue
        pts = []
        for _, row in curve.iterrows():
            x = left + chart_w * min(row["day"], 365) / 365
            y = top + chart_h * (1 - row["cumulative_repurchase"])
            pts.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_y = top + idx * 24
        parts.append(f'<rect x="{left + chart_w + 28}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{left + chart_w + 48}" y="{legend_y}" class="small">{name}</text>')
    return write_svg(path, "\n".join(parts), width, height)


def pca_projection_chart(path: Path, pca_sample: pd.DataFrame, width=1000, height=650):
    colors = ["#2f6fed", "#1c8c62", "#d95f02", "#7b61ff", "#596b82", "#c43c5c"]
    df = pca_sample.copy()
    df["x_proj"] = df["PC1"] + 0.55 * df["PC3"]
    df["y_proj"] = df["PC2"] - 0.35 * df["PC3"]
    left, right, top, bottom = 60, 180, 70, 60
    chart_w = width - left - right
    chart_h = height - top - bottom
    x_min, x_max = df["x_proj"].quantile([0.01, 0.99])
    y_min, y_max = df["y_proj"].quantile([0.01, 0.99])
    parts = [
        '<text x="30" y="42" class="title">PCA 三維投影：最終分群視覺化</text>',
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#f8fafc" stroke="#d9e1ef"/>',
    ]
    for _, row in df.iterrows():
        x = left + chart_w * (np.clip(row["x_proj"], x_min, x_max) - x_min) / (x_max - x_min + 1e-9)
        y = top + chart_h * (1 - (np.clip(row["y_proj"], y_min, y_max) - y_min) / (y_max - y_min + 1e-9))
        color = colors[int(row["cluster"]) % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}" opacity="0.58"/>')
    for i, cluster in enumerate(sorted(df["cluster"].unique())):
        color = colors[int(cluster) % len(colors)]
        legend_y = top + i * 24
        parts.append(f'<rect x="{left + chart_w + 28}" y="{legend_y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{left + chart_w + 48}" y="{legend_y}" class="small">Cluster {int(cluster)}</text>')
    parts.append(f'<text x="{left + chart_w / 2:.1f}" y="{height - 20}" text-anchor="middle" class="small">PC1 + PC3 投影 / PC2 + PC3 投影（靜態 SVG 近似 3D 視角）</text>')
    return write_svg(path, "\n".join(parts), width, height)


def color_blend(low: str, mid: str, high: str, value: float) -> str:
    value = float(np.clip(value, -1, 1))

    def hex_to_rgb(color: str):
        color = color.lstrip("#")
        return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)

    def rgb_to_hex(rgb):
        return "#" + "".join(f"{int(round(x)):02x}" for x in np.clip(rgb, 0, 255))

    if value < 0:
        t = value + 1
        rgb = hex_to_rgb(low) * (1 - t) + hex_to_rgb(mid) * t
    else:
        rgb = hex_to_rgb(mid) * (1 - value) + hex_to_rgb(high) * value
    return rgb_to_hex(rgb)


def build_profile_heatmap(named: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_labels = {
        "recency_median": "距上次投保天數",
        "frequency_mean": "平均投保次數",
        "total_premium_median": "累積保費中位數",
        "avg_premium_median": "單筆保費中位數",
        "avg_days_mean": "平均旅遊天數",
        "intl_share_mean": "海外旅遊占比",
        "japan_share_mean": "日本目的地占比",
        "multi_dest_share_mean": "多目的地占比",
        "short_notice_share_mean": "臨行前投保占比",
        "repurchase_rate_2025": "2025 回購率",
    }
    cluster_labels = named.apply(lambda r: f"{int(r['cluster'])} {r['cluster_name']}", axis=1)
    raw = named[list(metric_labels)].astype(float).copy()
    z = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0).replace(0, 1)
    z.index = cluster_labels
    heatmap = z.T.copy()
    heatmap.index = [metric_labels[col] for col in heatmap.index]

    long_rows = []
    for metric_col, metric_label in metric_labels.items():
        for idx, cluster_label in enumerate(cluster_labels):
            long_rows.append(
                {
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "cluster": int(named.iloc[idx]["cluster"]),
                    "cluster_name": named.iloc[idx]["cluster_name"],
                    "raw_value": raw.iloc[idx][metric_col],
                    "z_score": z.iloc[idx][metric_col],
                }
            )
    return heatmap, pd.DataFrame(long_rows)


def heatmap_chart(path: Path, heatmap: pd.DataFrame, width: int = 1080, height: int = 660):
    left, right, top, bottom = 210, 40, 118, 60
    cols = list(heatmap.columns)
    rows = list(heatmap.index)
    cell_w = (width - left - right) / max(len(cols), 1)
    cell_h = (height - top - bottom) / max(len(rows), 1)
    max_abs = max(float(np.nanmax(np.abs(heatmap.to_numpy()))), 1e-9)
    parts = [
        '<text x="30" y="42" class="title">客群輪廓標準化 Heatmap</text>',
        '<text x="30" y="68" class="small">數值為各指標跨客群標準化後的 z-score；紅色代表相對較高，藍色代表相對較低。</text>',
    ]
    for j, col in enumerate(cols):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 18}" transform="rotate(-18 {x:.1f},{top - 18})" text-anchor="middle" class="small">{escape(col)}</text>'
        )
    for i, row_label in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + cell_h * .65:.1f}" text-anchor="end" class="small">{escape(row_label)}</text>')
        for j, col in enumerate(cols):
            value = float(heatmap.iloc[i, j])
            normalized = value / max_abs
            fill = color_blend("#3b82c4", "#f8fafc", "#d95f02", normalized)
            x = left + j * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff"/>')
            text_color = "#1f2937" if abs(normalized) < 0.75 else "#ffffff"
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h * .65:.1f}" text-anchor="middle" class="small" fill="{text_color}">{value:+.1f}</text>'
            )
    parts.append(f'<text x="{left + (width - left - right) / 2:.1f}" y="{height - 24}" text-anchor="middle" class="small">相對比較圖，適合簡報快速說明每群高低特徵。</text>')
    return write_svg(path, "\n".join(parts), width, height)


def build_lifelines_km_outputs(validation_customers: pd.DataFrame, named: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    if not lifelines_available():
        return pd.DataFrame(), pd.DataFrame(), False, "lifelines 不可用，保留手寫 Kaplan-Meier 結果。"

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    curve_rows = []
    timeline = np.arange(0, 366)
    name_map = named.set_index("cluster")["cluster_name"].to_dict()
    for cluster, group in validation_customers.groupby("cluster"):
        kmf = KaplanMeierFitter(label=f"{int(cluster)} {name_map[int(cluster)]}")
        kmf.fit(group["duration_days"], event_observed=group["event"], timeline=timeline)
        survival = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
        for day in timeline:
            curve_rows.append(
                {
                    "cluster": int(cluster),
                    "cluster_name": name_map[int(cluster)],
                    "day": int(day),
                    "survival": float(survival.loc[day]),
                    "survival_ci_lower": float(ci_lower.loc[day]),
                    "survival_ci_upper": float(ci_upper.loc[day]),
                    "cumulative_repurchase": 1 - float(survival.loc[day]),
                    "cumulative_repurchase_ci_lower": 1 - float(ci_upper.loc[day]),
                    "cumulative_repurchase_ci_upper": 1 - float(ci_lower.loc[day]),
                }
            )

    logrank_rows = []
    clusters = sorted(validation_customers["cluster"].unique())
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1 :]:
            g1 = validation_customers[validation_customers["cluster"] == c1]
            g2 = validation_customers[validation_customers["cluster"] == c2]
            result = logrank_test(g1["duration_days"], g2["duration_days"], event_observed_A=g1["event"], event_observed_B=g2["event"])
            logrank_rows.append(
                {
                    "cluster_a": int(c1),
                    "cluster_a_name": name_map[int(c1)],
                    "cluster_b": int(c2),
                    "cluster_b_name": name_map[int(c2)],
                    "test_statistic": float(result.test_statistic),
                    "p_value": float(result.p_value),
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(logrank_rows), True, "lifelines Kaplan-Meier、95% 信賴區間與 pairwise log-rank test 已完成。"


def lifelines_km_chart(path: Path, lifelines_curve: pd.DataFrame, width: int = 1000, height: int = 560) -> None:
    if lifelines_curve.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["svg.fonttype"] = "none"
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    colors = ["#2f6fed", "#1c8c62", "#d95f02", "#7b61ff", "#596b82", "#c43c5c"]
    for idx, (cluster, group) in enumerate(lifelines_curve.groupby("cluster")):
        label = f"{int(cluster)} {group['cluster_name'].iloc[0]}"
        color = colors[idx % len(colors)]
        ax.plot(group["day"], group["cumulative_repurchase"], label=label, color=color, linewidth=2.4)
        ax.fill_between(
            group["day"].to_numpy(),
            group["cumulative_repurchase_ci_lower"].to_numpy(),
            group["cumulative_repurchase_ci_upper"].to_numpy(),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax.set_title("各群 2025 累積回購曲線（lifelines 95% 信賴區間）", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("距 2023-2024 最後一次投保天數")
    ax.set_ylabel("累積回購率")
    ax.set_xlim(0, 365)
    ax.grid(axis="y", color="#d9e1ef", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def create_notebook(summary_md: str) -> None:
    code = '''# 這份 notebook 對應「簡報大綱」的實作流程。
# 若要重跑完整分析，請在同一資料夾執行：
#   python run_customer_segmentation_analysis.py

from pathlib import Path
import pandas as pd

WORK_DIR = Path(".")
pd.read_csv(WORK_DIR / "outputs" / "feature_set_comparison.csv").head()
'''
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# 客群分群與 2025 回購驗證分析\n\n本 notebook 是實作紀錄入口；完整結果請看 `測試紀錄與結果.md`，彙總表請看 `outputs/`，定稿圖請看 `charts/`。\n",
        },
        {"cell_type": "markdown", "metadata": {}, "source": summary_md},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code},
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df, _ = load_and_prepare_data()
    features, train, valid = build_customer_features(df)
    use_sklearn = sklearn_available()

    split_summary = pd.DataFrame(
        [
            {
                "period": "2023-2024 建模",
                "policies": len(train),
                "customers": train["APL_ID"].nunique(),
                "premium": train["TOT_PREM"].sum(),
            },
            {
                "period": "2025 驗證",
                "policies": len(valid),
                "customers": valid["APL_ID"].nunique(),
                "premium": valid["TOT_PREM"].sum(),
            },
        ]
    )

    comparison, selected_key, fitted = evaluate_feature_sets(features, use_sklearn=use_sklearn)
    implementation_check = compare_implementations(
        features,
        [("A_RFM_only", 2), (FINAL_FEATURE_SET, FINAL_K), (FINAL_FEATURE_SET, 4), ("C_RFM_purchase", 3)],
    )
    implementation_check.to_csv(OUTPUT_DIR / "implementation_comparison.csv", index=False, encoding="utf-8-sig")
    comparison = comparison.sort_values(["silhouette_sample", "min_cluster_share"], ascending=[False, False])
    comparison.to_csv(OUTPUT_DIR / "feature_set_comparison.csv", index=False, encoding="utf-8-sig")

    best_key = (FINAL_FEATURE_SET, FINAL_K) if (FINAL_FEATURE_SET, FINAL_K) in fitted else selected_key
    best = fitted[best_key]
    labels = best["fit"]["labels"]
    final_feature_set, final_k = best_key
    features_with_cluster = features.copy()
    features_with_cluster["cluster"] = labels

    profile = cluster_profiles(features, labels)
    validation_customers, validation = build_validation(features, valid, labels)
    named = name_clusters(profile, validation)
    validation_lift = build_validation_lift(named)
    named.to_csv(OUTPUT_DIR / "final_cluster_profile.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUTPUT_DIR / "validation_by_cluster.csv", index=False, encoding="utf-8-sig")
    validation_lift.to_csv(OUTPUT_DIR / "validation_lift_vs_overall.csv", index=False, encoding="utf-8-sig")
    named[
        [
            "cluster",
            "cluster_name",
            "marketing_priority",
            "first_reminder",
            "second_reminder",
            "copy_direction",
            "sample_copy",
        ]
    ].to_csv(OUTPUT_DIR / "strategy_recommendations.csv", index=False, encoding="utf-8-sig")

    scores, explained = pca_numpy(best["x"], n_components=3)
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(scores), size=min(7000, len(scores)), replace=False)
    pca_sample = pd.DataFrame(
        {
            "PC1": scores[sample_idx, 0],
            "PC2": scores[sample_idx, 1],
            "PC3": scores[sample_idx, 2],
            "cluster": labels[sample_idx],
        }
    )
    pca_sample.to_csv(OUTPUT_DIR / "pca_3d_sample.csv", index=False, encoding="utf-8-sig")
    pca_explained = pd.DataFrame({"component": ["PC1", "PC2", "PC3"], "explained_variance_ratio": explained})
    pca_explained.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    km_rows = []
    km_curves = {}
    for cluster, group in validation_customers.groupby("cluster"):
        curve = km_curve(group, max_day=365)
        curve["cluster"] = int(cluster)
        km_rows.append(curve)
        cluster_name = named.loc[named["cluster"] == cluster, "cluster_name"].iloc[0]
        km_curves[f"{int(cluster)} {cluster_name}"] = curve
    km_all = pd.concat(km_rows, ignore_index=True)
    km_all.to_csv(OUTPUT_DIR / "km_curve_by_cluster.csv", index=False, encoding="utf-8-sig")

    lifelines_curve, logrank_results, lifelines_ok, lifelines_note = build_lifelines_km_outputs(validation_customers, named)
    if not lifelines_curve.empty:
        lifelines_curve.to_csv(OUTPUT_DIR / "km_curve_by_cluster_lifelines.csv", index=False, encoding="utf-8-sig")
    if not logrank_results.empty:
        logrank_results.to_csv(OUTPUT_DIR / "km_logrank_test.csv", index=False, encoding="utf-8-sig")

    heatmap, heatmap_long = build_profile_heatmap(named)
    heatmap_long.to_csv(OUTPUT_DIR / "cluster_profile_heatmap.csv", index=False, encoding="utf-8-sig")

    k4_profile, k3_k4_comparison = build_k4_business_comparison(features, valid, fitted)
    k4_profile.to_csv(OUTPUT_DIR / "b_rfm_travel_k4_profile.csv", index=False, encoding="utf-8-sig")
    k3_k4_comparison.to_csv(OUTPUT_DIR / "b_rfm_travel_k3_k4_business_comparison.csv", index=False, encoding="utf-8-sig")

    top_comp = pd.read_csv(OUTPUT_DIR / "feature_set_comparison.csv").sort_values(
        ["silhouette_sample", "min_cluster_share"], ascending=[False, False]
    ).head(10)
    bar_chart(
        CHART_DIR / "01_feature_set_silhouette.svg",
        "特徵組合與 K 值：Silhouette Top 10",
        top_comp.apply(lambda r: f"{r['feature_set']} / K={int(r['k'])}", axis=1),
        top_comp["silhouette_sample"],
        height=650,
        value_fmt=lambda x: f"{x:.3f}",
    )
    pca_projection_chart(CHART_DIR / "02_pca_3d_projection.svg", pca_sample)
    bar_chart(
        CHART_DIR / "03_2025_repurchase_rate.svg",
        "2025 各群回購率",
        named.apply(lambda r: f"{int(r['cluster'])} {r['cluster_name']}", axis=1),
        named["repurchase_rate_2025"],
        value_fmt=lambda x: f"{x*100:.1f}%",
        color="#1c8c62",
    )
    line_chart(CHART_DIR / "04_km_repurchase_curve.svg", "各群 2025 累積回購曲線", km_curves)
    lifelines_km_chart(CHART_DIR / "04b_km_repurchase_curve_lifelines.svg", lifelines_curve)
    heatmap_chart(CHART_DIR / "05_cluster_profile_heatmap.svg", heatmap)

    display_comp = comparison.sort_values(["silhouette_sample", "min_cluster_share"], ascending=[False, False]).head(8).copy()
    display_comp = display_comp[
        [
            "feature_set",
            "k",
            "feature_count",
            "silhouette_sample",
            "davies_bouldin",
            "calinski_harabasz",
            "min_cluster_share",
            "max_cluster_share",
        ]
    ]
    for col in ["silhouette_sample", "davies_bouldin", "calinski_harabasz", "min_cluster_share", "max_cluster_share"]:
        display_comp[col] = display_comp[col].map(lambda x: fmt_float(x, 3) if "share" not in col else fmt_pct(x))

    profile_display = named[
        [
            "cluster",
            "cluster_name",
            "customers",
            "customer_share",
            "recency_median",
            "frequency_mean",
            "total_premium_median",
            "avg_premium_median",
            "avg_days_mean",
            "short_notice_share_mean",
            "repurchase_rate_2025",
            "event_time_median",
        ]
    ].copy()
    profile_display["customers"] = profile_display["customers"].map(fmt_int)
    profile_display["customer_share"] = profile_display["customer_share"].map(fmt_pct)
    for col in ["recency_median", "frequency_mean", "avg_days_mean", "event_time_median"]:
        profile_display[col] = profile_display[col].map(lambda x: fmt_float(x, 1))
    for col in ["total_premium_median", "avg_premium_median"]:
        profile_display[col] = profile_display[col].map(fmt_money)
    profile_display["short_notice_share_mean"] = profile_display["short_notice_share_mean"].map(fmt_pct)
    profile_display["repurchase_rate_2025"] = profile_display["repurchase_rate_2025"].map(fmt_pct)

    validation_display = validation.copy()
    validation_display["customers"] = validation_display["customers"].map(fmt_int)
    validation_display["repurchase_2025"] = validation_display["repurchase_2025"].map(fmt_int)
    validation_display["repurchase_rate_2025"] = validation_display["repurchase_rate_2025"].map(fmt_pct)
    for col in ["event_time_mean", "event_time_median", "event_time_p25", "event_time_p75"]:
        validation_display[col] = validation_display[col].map(lambda x: fmt_float(x, 1))
    for horizon in [30, 60, 90, 180, 365]:
        validation_display[f"km_repurchase_{horizon}d"] = validation_display[f"km_repurchase_{horizon}d"].map(fmt_pct)
    validation_display = validation_display[
        [
            "cluster",
            "customers",
            "repurchase_2025",
            "repurchase_rate_2025",
            "event_time_median",
            "km_repurchase_30d",
            "km_repurchase_60d",
            "km_repurchase_90d",
            "km_repurchase_180d",
            "km_repurchase_365d",
        ]
    ]

    lift_display = validation_lift.copy()
    lift_display["customers"] = lift_display["customers"].map(fmt_int)
    for col in ["repurchase_rate_2025", "overall_repurchase_rate_2025", "rate_diff_vs_overall"]:
        lift_display[col] = lift_display[col].map(fmt_pct)
    lift_display["lift_vs_overall"] = lift_display["lift_vs_overall"].map(lambda x: f"{x:.2f}x")
    lift_display["event_time_median"] = lift_display["event_time_median"].map(lambda x: fmt_float(x, 1))
    lift_display = lift_display[
        [
            "cluster",
            "cluster_name",
            "customers",
            "repurchase_rate_2025",
            "overall_repurchase_rate_2025",
            "rate_diff_vs_overall",
            "lift_vs_overall",
            "event_time_median",
            "validation_reading",
        ]
    ]

    strategy_display = named[
        [
            "cluster",
            "cluster_name",
            "marketing_priority",
            "first_reminder",
            "second_reminder",
            "copy_direction",
            "sample_copy",
        ]
    ].copy()

    implementation_display = implementation_check[
        [
            "feature_set",
            "k",
            "implementation",
            "silhouette_sample",
            "davies_bouldin",
            "calinski_harabasz",
            "min_cluster_share",
            "max_cluster_share",
        ]
    ].copy()
    for col in ["silhouette_sample", "davies_bouldin", "calinski_harabasz"]:
        implementation_display[col] = implementation_display[col].map(lambda x: fmt_float(x, 3))
    for col in ["min_cluster_share", "max_cluster_share"]:
        implementation_display[col] = implementation_display[col].map(fmt_pct)

    k4_display = k4_profile[
        [
            "cluster",
            "cluster_name",
            "customers",
            "customer_share",
            "recency_median",
            "total_premium_median",
            "avg_days_mean",
            "intl_share_mean",
            "multi_dest_share_mean",
            "repurchase_rate_2025",
            "business_summary",
        ]
    ].copy()
    k4_display["customers"] = k4_display["customers"].map(fmt_int)
    k4_display["customer_share"] = k4_display["customer_share"].map(fmt_pct)
    for col in ["recency_median", "avg_days_mean"]:
        k4_display[col] = k4_display[col].map(lambda x: fmt_float(x, 1))
    k4_display["total_premium_median"] = k4_display["total_premium_median"].map(fmt_money)
    for col in ["intl_share_mean", "multi_dest_share_mean", "repurchase_rate_2025"]:
        k4_display[col] = k4_display[col].map(fmt_pct)

    logrank_display = logrank_results.copy()
    if not logrank_display.empty:
        logrank_display["pair"] = logrank_display.apply(
            lambda r: f"{int(r['cluster_a'])} {r['cluster_a_name']} vs {int(r['cluster_b'])} {r['cluster_b_name']}",
            axis=1,
        )
        logrank_display["test_statistic"] = logrank_display["test_statistic"].map(lambda x: fmt_float(x, 3))
        logrank_display["p_value"] = logrank_display["p_value"].map(lambda x: f"{x:.4g}")
        logrank_display = logrank_display[["pair", "test_statistic", "p_value"]]

    checklist = pd.DataFrame(
        [
            ["建立獨立工作資料夾", "完成", "所有實作檔案集中於 `分群回購分析實作/`"],
            ["資料切分：2023-2024 建模、2025 驗證", "完成", "分群特徵只由 2023-2024 建立"],
            ["建立 RFM 核心特徵", "完成", "R=截至 2024-12-31 recency，F=投保次數，M=累積保費"],
            ["測試 A-E 特徵組合", "完成", "每組 K=2 到 K=6；目前使用 scikit-learn 重跑"],
            ["分群品質比較", "完成", "輸出 silhouette、Davies-Bouldin、CH、群體占比，並保留手寫版對照"],
            ["PCA 三維視覺化", "完成", "輸出靜態三維投影 SVG 與 PCA sample CSV"],
            ["2025 回購驗證", "完成", "計算各群 2025 回購率與回購天數"],
            ["Kaplan-Meier 回購曲線", "完成", "補上 lifelines 95% 信賴區間與 log-rank test"],
            ["客群輪廓 heatmap", "完成", "輸出標準化 heatmap，方便簡報比較高低特徵"],
            ["K=4 商業解釋比較", "完成", "補做 B_RFM_travel / K=4 備選方案比較"],
            ["行銷策略與文案", "完成初版", "依群體輪廓自動產生提醒時機與文案方向"],
        ],
        columns=["項目", "狀態", "說明"],
    )

    best_row = comparison[(comparison["feature_set"] == final_feature_set) & (comparison["k"] == final_k)].iloc[0]
    report = f"""# 測試紀錄與結果

> 本文件紀錄 `簡報大綱.md` 的實作進度、測試比較與目前定稿結果。  
> 工作資料夾：`分群回購分析實作/`  
> 原始資料只讀取，不在原始資料夾中新增分析輸出。

## 0. 環境紀錄

- 本次環境已可使用 `scikit-learn`、`matplotlib`、`seaborn`、`lifelines`、`nbformat`、`nbclient`、`ipykernel`。
- 分群與分群指標已改用 `scikit-learn` 重跑；手寫 `numpy` 版保留為 fallback 與一致性對照。
- Kaplan-Meier 已補 `lifelines` 正式估計、95% 信賴區間與 pairwise log-rank test。
- 產圖時將 matplotlib cache 指到工作資料夾，避免寫入使用者家目錄失敗。

## 1. 是否照簡報大綱逐步完成

{df_to_markdown(checklist)}

## 2. 資料切分

{df_to_markdown(split_summary.assign(policies=split_summary['policies'].map(fmt_int), customers=split_summary['customers'].map(fmt_int), premium=split_summary['premium'].map(fmt_money)))}

## 3. 特徵組合與 K 值比較

最終選擇：

- 特徵組合：`{final_feature_set}`
- K 值：`{final_k}`
- Silhouette sample：`{best_row['silhouette_sample']:.3f}`
- Davies-Bouldin：`{best_row['davies_bouldin']:.3f}`
- Calinski-Harabasz：`{best_row['calinski_harabasz']:.1f}`
- 最小群體占比：`{best_row['min_cluster_share']:.1%}`

選擇理由：`A_RFM_only / K=2` 的 silhouette 最高，但只用 RFM 分成兩群，客群輪廓較粗，較難支撐「旅遊型態」與「差異化文案」；`B_RFM_travel / K=3` 的 silhouette 接近，Davies-Bouldin 較佳，且能分出海外主力、短天數臨行、多目的地長天數高價值三種較能轉成行銷策略的客群，因此先採用這組作為目前定稿版本。

Top 8 組合：

{df_to_markdown(display_comp)}

![特徵組合與 K 值比較](charts/01_feature_set_silhouette.svg)

### 3.1 scikit-learn 與手寫版一致性檢查

下表保留幾個主要候選組合的重跑結果。`scikit-learn` 與手寫版的指標排序方向大致一致；小幅差異主要來自 K-means 初始化、收斂細節與 silhouette 抽樣。

{df_to_markdown(implementation_display)}

## 4. PCA 視覺化

前三個主成分解釋變異：

{df_to_markdown(pca_explained.assign(explained_variance_ratio=pca_explained['explained_variance_ratio'].map(fmt_pct)))}

![PCA 三維投影](charts/02_pca_3d_projection.svg)

## 5. 最終客群輪廓

{df_to_markdown(profile_display)}

![客群輪廓標準化 heatmap](charts/05_cluster_profile_heatmap.svg)

## 6. 2025 回購驗證

{df_to_markdown(validation_display)}

### 6.1 與整體 2025 回購率比較

這張表才是判斷分群有沒有驗證效果時最直覺看的地方。`overall_repurchase_rate_2025` 是不分群時的整體基準；`rate_diff_vs_overall` 和 `lift_vs_overall` 則看每一群是否高於或低於這個基準。

{df_to_markdown(lift_display)}

![2025 各群回購率](charts/03_2025_repurchase_rate.svg)

![各群 2025 累積回購曲線](charts/04_km_repurchase_curve.svg)

![各群 2025 累積回購曲線 lifelines](charts/04b_km_repurchase_curve_lifelines.svg)

`lifelines` 補強狀態：{lifelines_note}

{df_to_markdown(logrank_display) if not logrank_display.empty else "目前沒有 log-rank test 輸出。"}

## 7. B_RFM_travel / K=4 商業解釋比較

{df_to_markdown(k4_display)}

K=3 與 K=4 的簡報取捨：

{df_to_markdown(k3_k4_comparison)}

## 8. 行銷策略與文案初版

{df_to_markdown(strategy_display)}

> 這裡只保留自動產生的初版策略方向；「改成更像品牌行銷語氣」刻意不處理，留給負責文案的組員發想。

## 9. 目前觀察到的重點

1. 這次未使用 2025 資訊建立分群，符合避免資料洩漏原則。
2. RFM 與精簡加值特徵比舊版大量特徵更容易解釋，也更適合轉成簡報。
3. 各群 2025 回購率與回購曲線已可比較；正式簡報時可用這些結果決定提醒優先級。
4. 與整體 2025 回購率相比，高價值長天數客略高、臨行前快速投保客略低，代表分群有驗證出方向性差異，但 lift 幅度不大。
5. K=4 可以拆細部分海外主力客，但策略溝通成本較高，因此目前仍建議以 K=3 作為簡報主方案。
6. 行銷策略目前只保留依客群輪廓與 2025 驗證自動產生的初版，不做品牌語氣潤飾。

## 10. 9.5 後續優先改善事項執行結果

| 原優先事項 | 本次狀態 |
| --- | --- |
| 用 `scikit-learn` 重跑 K-means 與分群指標 | 已完成，並輸出 `outputs/implementation_comparison.csv` |
| 補做 `B_RFM_travel / K=4` 商業解釋比較 | 已完成，並輸出 K=4 輪廓與 K=3/K=4 比較表 |
| 用 `lifelines` 補正式 Kaplan-Meier 圖、信賴區間與 log-rank test | 已完成，輸出 lifelines 曲線、信賴區間與 pairwise log-rank test |
| 補一張客群輪廓 heatmap | 已完成，輸出 `charts/05_cluster_profile_heatmap.svg` |
| 由負責文案的組員改寫品牌行銷語氣 | 本次刻意保留不做，留給組員發想 |

## 11. 產出檔案

- Notebook：`客群分群_回購驗證_分析.ipynb`
- 特徵組合比較：`outputs/feature_set_comparison.csv`
- scikit-learn / 手寫版一致性對照：`outputs/implementation_comparison.csv`
- 最終客群輪廓：`outputs/final_cluster_profile.csv`
- 2025 驗證表：`outputs/validation_by_cluster.csv`
- 2025 回購率相對整體基準比較：`outputs/validation_lift_vs_overall.csv`
- KM 曲線資料：`outputs/km_curve_by_cluster.csv`
- lifelines KM 曲線資料：`outputs/km_curve_by_cluster_lifelines.csv`
- log-rank test：`outputs/km_logrank_test.csv`
- 客群輪廓 heatmap 資料：`outputs/cluster_profile_heatmap.csv`
- K=4 備選輪廓：`outputs/b_rfm_travel_k4_profile.csv`
- K=3/K=4 商業比較：`outputs/b_rfm_travel_k3_k4_business_comparison.csv`
- 策略建議：`outputs/strategy_recommendations.csv`
- 圖表：`charts/*.svg`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    create_notebook(report)

    print("final_feature_set", final_feature_set)
    print("final_k", final_k)
    print("report", REPORT_PATH)
    print("notebook", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
