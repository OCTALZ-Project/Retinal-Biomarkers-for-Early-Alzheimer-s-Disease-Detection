import datetime as _datetime
from datetime import timezone

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = timezone.utc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, NelsonAalenFitter
from lifelines.plotting import add_at_risk_counts
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

# --- Publication figure style (Springer/GeroScience): sans-serif lettering
# (Arial/Helvetica) + TrueType/Type-42 embedding (matplotlib's default is
# Type-3, which the journal disallows). Set after sns.set_style so it wins. ---
plt.rcParams.update(
    {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        # Arial first; the rest are Arial-metric / sans-serif fallbacks (see
        # requirements.txt for the Arial/mscorefonts install command).
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Helvetica",
            "Nimbus Sans",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "axes.labelweight": "normal",
        "axes.titleweight": "bold",
        "xtick.labelsize": 18,  # larger tick labels for readability
        "ytick.labelsize": 18,
    }
)


def combine_cox_summaries_rubins(summary_dfs):
    """
    Combine Cox PH summaries across folds using Rubin's rules.
    Each df in summary_dfs should be a cph.summary DataFrame with 'feature' column.
    """
    combined = pd.concat(summary_dfs)
    m = len(summary_dfs)

    grouped = combined.groupby("feature")

    # Mean coefficient across folds
    coef_mean = grouped["coef"].mean()

    # Within-fold variance (mean of squared SEs)
    within_var = grouped["se(coef)"].apply(lambda x: (x**2).mean())

    # Between-fold variance of coefficients
    between_var = grouped["coef"].var(ddof=1)

    # Rubin's combined variance
    combined_var = within_var + (1 + 1 / m) * between_var
    combined_se = np.sqrt(combined_var)

    # Recompute z and p from combined estimates
    z = coef_mean / combined_se
    p = 2 * (1 - scipy_stats.norm.cdf(np.abs(z)))

    # Recompute confidence intervals
    coef_lower = coef_mean - 1.96 * combined_se
    coef_upper = coef_mean + 1.96 * combined_se

    result = pd.DataFrame(
        {
            "coef": coef_mean,
            "exp(coef)": np.exp(coef_mean),
            "se(coef)": combined_se,
            "coef lower 95%": coef_lower,
            "coef upper 95%": coef_upper,
            "exp(coef) lower 95%": np.exp(coef_lower),
            "exp(coef) upper 95%": np.exp(coef_upper),
            "z": z,
            "p": p,
        }
    )

    result.index.name = "feature"
    return result


def fit_cox_rubins(
    ldf, duration_col, event_col, categorical_features, drop_cols=None, penalizer=0.01
):
    """
    Fit Cox PH across multiple imputed datasets and combine with Rubin's rules.
    Numerical features are standardized (not binarized) to preserve statistical power.
    Categorical features (binary or one-hot dummy columns) are passed as-is; do
    any one-hot encoding upstream before calling this function.
    """
    if drop_cols is None:
        drop_cols = ["eid", "Date of death", "examination year"]

    numerical_features = ldf[0].columns.difference(
        categorical_features + [event_col, duration_col] + drop_cols
    )

    summary_dfs = []
    for df in ldf:
        df_cox = df.copy()
        df_cox.drop(columns=[c for c in drop_cols if c in df_cox.columns], inplace=True)

        scaler = StandardScaler()
        df_cox[numerical_features] = scaler.fit_transform(df_cox[numerical_features])

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df_cox, duration_col, event_col=event_col)

        summary = cph.summary.copy()
        summary["feature"] = summary.index
        summary_dfs.append(summary)

    agg_df = combine_cox_summaries_rubins(summary_dfs)
    agg_df["-log2(p)"] = -np.log2(agg_df["p"].replace(0, np.nextafter(0, 1)))
    agg_df = agg_df.round(4)
    return agg_df


def plot_cox_forest(agg_df, save_path=None, figsize=(20, 25)):
    """
    Plot a forest plot of Cox PH coefficients with 95% CIs.
    Significant features (p < 0.05) are highlighted in red.
    """
    plot_df = agg_df.sort_values("coef")
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        x=plot_df["coef"],
        y=y_pos,
        xerr=[
            plot_df["coef"] - plot_df["coef lower 95%"],
            plot_df["coef upper 95%"] - plot_df["coef"],
        ],
        fmt="none",
        ecolor="darkgray",
        elinewidth=2.5,
        capsize=3,
    )

    sig_mask = plot_df["p"] < 0.05
    ax.scatter(
        plot_df.loc[sig_mask, "coef"],
        y_pos[sig_mask],
        marker="D",
        s=80,
        color="tab:red",
        label="p < 0.05",
    )

    ax.axvline(x=0, color="black", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index, fontsize=18, fontweight="bold")
    ax.set_xlabel("log(HR) (mean. +/- 95% CI)", fontsize=19)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return fig, ax


# Function to calculate median survival time from a Cox model
def get_important_cox_features(summary_df, p_threshold=0.05):
    summary_df["is_significant"] = (summary_df["p"] < p_threshold) & ~(
        (summary_df["coef lower 95%"] < 0) & (summary_df["coef upper 95%"] > 0)
    )
    important_features = summary_df[summary_df["is_significant"]].index.tolist()
    return important_features


# Determine dynamic max time where at least N subjects are still at risk
def find_dynamic_xlim(naf, min_at_risk=50):
    at_risk_counts = naf.event_table["at_risk"]
    valid_times = at_risk_counts[at_risk_counts >= min_at_risk].index
    if len(valid_times) > 0:
        return valid_times[-1]  # last time point where at_risk >= min_at_risk
    else:
        return naf.cumulative_hazard_.index.max()  # fallback: full range


# Function to plot cumulative hazard curves with a risk table
def plot_cumulative_hazard_with_risk_table(
    T,
    E,
    group_mask,
    show_ci=True,
    style="ax",
    group_labels=("Thin", "Thick"),
    colors=("red", "blue"),
    title="Cumulative hazard plot",
    xlabel="Years",
    ylabel="Cumulative hazard",
    figsize=(10, 6),
):

    naf_1 = NelsonAalenFitter()
    naf_2 = NelsonAalenFitter()

    fig, ax = plt.subplots(figsize=figsize)

    if style == "original":
        naf_1.fit(T[group_mask], event_observed=E[group_mask], label=group_labels[0])
        naf_1.plot_cumulative_hazard(
            ax=ax,
            ci_show=show_ci,
            color=colors[0],
            linestyle="-",
            marker="|",
            markersize=6,
        )
        naf_2.fit(T[~group_mask], event_observed=E[~group_mask], label=group_labels[1])
        naf_2.plot_cumulative_hazard(
            ax=ax,
            ci_show=show_ci,
            color=colors[1],
            linestyle="-",
            marker="|",
            markersize=6,
        )

    else:
        # Group 1
        naf_1.fit(T[group_mask], event_observed=E[group_mask], label=group_labels[0])
        (line_1,) = ax.plot(
            naf_1.cumulative_hazard_.index,
            naf_1.cumulative_hazard_[group_labels[0]],
            label=group_labels[0],
            color=colors[0],
            linestyle="-",
            marker="|",
            markersize=6,
        )

        # Group 2
        naf_2.fit(T[~group_mask], event_observed=E[~group_mask], label=group_labels[1])
        (line_2,) = ax.plot(
            naf_2.cumulative_hazard_.index,
            naf_2.cumulative_hazard_[group_labels[1]],
            label=group_labels[1],
            color=colors[1],
            linestyle="-",
            marker="|",
            markersize=6,
        )

        xmin = min(
            naf_1.cumulative_hazard_.index.min(), naf_2.cumulative_hazard_.index.min()
        )
        # xlim1 = find_dynamic_xlim(naf_1, min_at_risk=10)
        # xlim2 = find_dynamic_xlim(naf_2, min_at_risk=10)
        xlim_max = 13
        xmin_rounded = np.floor(xmin / 2.5) * 2.5

        ax.set_xlim(xmin_rounded, xlim_max)
        xticks = np.arange(xmin_rounded, xlim_max, 2.5)
        ax.set_xticks(xticks)

        handles = [line_1, line_2]
        labels = [line.get_label() for line in handles]
        plt.legend(
            handles,
            labels,
            title="Group",
            loc="upper left",
            fontsize=10,
            title_fontsize=10,
        )

    # Risk table
    add_at_risk_counts(naf_1, naf_2, ax=ax, labels=group_labels)

    # Title and axes
    plt.title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.show()


# Function to plot cumulative hazard curves for all features and save as separate files
def plot_and_save_separate(
    d_knn,
    important_features,
    categorical_features,
    curve_path="cumulative_hazard.pdf",
    risk_path="risk_table.pdf",
    task="ad",
):
    base_colors = [
        "green",
        "blue",
        "purple",
        "orange",
        "red",
        "yellow",
        "brown",
        "pink",
        "cyan",
        "magenta",
    ]
    markers = ["d", "|", "p", "s", "o", "^", "v", "<", ">", "x"]

    duration_col = "ad_after" if task == "ad" else "d_after"
    T, E = d_knn[duration_col], d_knn["Status"]
    xticks = np.array([0, 3, 5, 7, 9, 11, 13])

    curve_fig, curve_ax = plt.subplots(figsize=(14, 8))
    fitters, labels = [], []

    for i, feature in enumerate(important_features):
        if feature in categorical_features:
            continue
        thin_mask = d_knn[feature] < d_knn[feature].median()
        thick_mask = ~thin_mask

        naf_thin = NelsonAalenFitter()
        naf_thick = NelsonAalenFitter()

        lbl_thin = f"{feature} - thin"
        lbl_thick = f"{feature} - thick"

        naf_thin.fit(T[thin_mask], event_observed=E[thin_mask], label=lbl_thin)
        naf_thick.fit(T[thick_mask], event_observed=E[thick_mask], label=lbl_thick)

        color = base_colors[i % len(base_colors)]
        marker = markers[i % len(markers)]

        curve_ax.plot(
            naf_thin.cumulative_hazard_.index,
            naf_thin.cumulative_hazard_[lbl_thin],
            linestyle="--",
            marker=marker,
            markersize=5,
            color=color,
            label=lbl_thin,
        )

        curve_ax.plot(
            naf_thick.cumulative_hazard_.index,
            naf_thick.cumulative_hazard_[lbl_thick],
            linestyle="-",
            marker=marker,
            markersize=5,
            color=color,
            label=lbl_thick,
        )

        for naf, lbl in [(naf_thin, lbl_thin), (naf_thick, lbl_thick)]:
            curve_ax.text(
                naf.cumulative_hazard_.index[-1] + 0.1,
                naf.cumulative_hazard_[lbl].iloc[-1],
                lbl,
                fontsize=13,
                color=color,
                va="center",
                ha="left",
            )

        fitters.extend([naf_thin, naf_thick])
        labels.extend([lbl_thin, lbl_thick])

    curve_ax.set_xlim(0, 19)
    curve_ax.set_xticks(xticks)
    curve_ax.set_xlabel("Years", fontsize=15)
    curve_ax.set_ylabel("Cumulative Hazard", fontsize=15)
    curve_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    curve_ax.tick_params(axis="x", length=0)
    curve_fig.tight_layout()
    curve_fig.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(curve_fig)

    risk_fig, risk_ax = plt.subplots(figsize=(14, 5))

    risk_ax.set_xlim(0, 14)
    risk_ax.set_xticks(
        xticks,
        labels=[f"After {t} Years" for t in xticks],
        fontsize=13,
        fontweight="bold",
    )

    add_at_risk_counts(
        *fitters,
        ax=risk_ax,
        labels=labels,
        fontsize=13,
        rows_to_show=["At risk", "Events"],
    )

    risk_ax.tick_params(axis="x", which="both", length=0)
    risk_ax.set_yticks([])
    for spine in risk_ax.spines.values():
        spine.set_visible(False)

    for ax in risk_fig.axes:
        if ax is not risk_ax:
            ax.tick_params(axis="x", which="both", length=0)
            ax.grid(False)

    plt.setp(risk_ax.get_xticklabels(), rotation=0, ha="center")
    risk_fig.tight_layout()
    risk_fig.savefig(risk_path, dpi=300, bbox_inches="tight")
    plt.close(risk_fig)


def process_ad(df, death_info, ad_years, cohort_years):
    # filter the dataframe for AD
    ad_df = df.copy()
    filter = ad_df["Status"] == "dementia"
    ad_df.drop(ad_df[filter].index, inplace=True)
    ad_df["Status"] = ad_df["Status"].map({"healthy": 0, "AD": 1})

    # fill the NaN values in the date of death with the end date of the study
    ad_df = ad_df.merge(death_info, on="eid", how="left")
    ad_df["Date of death"] = ad_df["Date of death"].fillna(pd.Timestamp("2023-07-01"))
    ad_df = ad_df.merge(ad_years, on="eid", how="left")
    ad_df = ad_df.merge(cohort_years, on="eid", how="left")

    # calculate ad after for non-ad participants
    ad_filter = ad_df["ad_after"].isna()
    ad_df.loc[ad_filter, "ad_after"] = (
        ad_df.loc[ad_filter, "Date of death"] - ad_df.loc[ad_filter, "examination year"]
    ).dt.days // 365

    # sayı azaldığı için 13'ten büyük olanları atıyoruz
    ad_filter = (ad_df["ad_after"] >= 0) & (ad_df["ad_after"] <= 13)
    ad_df = ad_df[ad_filter]

    return ad_df


def process_d(df, death_info, d_years, cohort_years):

    # create the dementia
    d_df = df.copy()
    d_df["Status"] = d_df["Status"].map({"healthy": 0, "dementia": 1, "AD": 1})
    d_df = d_df.merge(death_info, on="eid", how="left")

    # fill the NaN values in the date of death with the end date of the study
    d_df["Date of death"] = d_df["Date of death"].fillna(pd.Timestamp("2023-07-01"))
    d_df = d_df.merge(d_years, on="eid", how="left")
    d_df = d_df.merge(cohort_years, on="eid", how="left")

    d_filter = d_df["d_after"].isna()
    # calculate d after for non-dementia participants
    d_df.loc[d_filter, "d_after"] = (
        d_df.loc[d_filter, "Date of death"] - d_df.loc[d_filter, "examination year"]
    ).dt.days // 365

    # sayı azaldığı için 13'ten büyük olanları atıyoruz
    d_filter = d_df["d_after"] <= 13
    d_df = d_df[d_filter]

    d_filter = d_df["d_after"] >= 0
    d_df = d_df[d_filter]

    return d_df


class MeanFitter:
    """
    add_at_risk_counts fonksiyonuna 'ortalama' risk/olay sayıları
    verebilmek için, yalnızca .event_table ve .name özelliklerini
    sunan küçük bir sarmalayıcı.
    """

    def __init__(self, name, durations, at_risk, events):
        self._name = name
        self._event_table = pd.DataFrame(
            {
                "at_risk": at_risk,
                "observed": events,
                "censored": 0,
                "removed": events,
                "entrance": 0,
            },
            index=durations,
        )

    @property
    def event_table(self):
        return self._event_table

    @property
    def name(self):
        return self._name


# Function to plot cumulative hazard curves for all features on a single figure
def plot_all_features_on_single_figure(
    ldf_ad, important_features, categorical_features, output_path="./", task="ad"
):
    base_colors = [
        "green",
        "blue",
        "brown",
        "purple",
        "red",
        "magenta",
        "brown",
        "orange",
        "yellow",
        "pink",
    ]
    markers = ["d", "|", "p", "s", "o", "^", "v", "<", ">", "x"]

    curve_fig, curve_ax = plt.subplots(figsize=(16, 10))
    xticks = np.array([0, 3, 5, 7, 9, 11, 13])

    stats = {}
    feature_groups = {}
    labels_to_plot = []

    for i, feature in enumerate(important_features):
        if feature == "Sex":
            low_name, high_name = "Women", "Men"
        elif feature in categorical_features:
            low_name, high_name = "false", "true"
        else:
            low_name, high_name = "thin", "thick"
        feature_groups[feature] = (low_name, high_name)

        stats.setdefault(
            feature, {low_name: {"at": [], "ev": []}, high_name: {"at": [], "ev": []}}
        )
        low_curves, high_curves = [], []
        lbl_low = f"{feature} - {low_name}"
        lbl_high = f"{feature} - {high_name}"

        for df in ldf_ad:
            if task == "dementia":
                T, E = df["d_after"], df["Status"]
            elif task == "ad":
                T, E = df["ad_after"], df["Status"]
            if feature in categorical_features:
                low_mask = df[feature] == 0
            else:
                low_mask = df[feature] < df[feature].median()
            high_mask = ~low_mask

            naf_low = NelsonAalenFitter()
            naf_high = NelsonAalenFitter()

            naf_low.fit(T[low_mask], event_observed=E[low_mask], label=lbl_low)
            naf_high.fit(T[high_mask], event_observed=E[high_mask], label=lbl_high)

            low_curves.append(naf_low.cumulative_hazard_.iloc[:, 0])
            high_curves.append(naf_high.cumulative_hazard_.iloc[:, 0])

            stats[feature][low_name]["at"].append(
                [(T[low_mask] >= t).sum() for t in xticks]
            )
            stats[feature][low_name]["ev"].append(
                [((T[low_mask] <= t) & (E[low_mask] == 1)).sum() for t in xticks]
            )
            stats[feature][high_name]["at"].append(
                [(T[high_mask] >= t).sum() for t in xticks]
            )
            stats[feature][high_name]["ev"].append(
                [((T[high_mask] <= t) & (E[high_mask] == 1)).sum() for t in xticks]
            )

        # Align curves to a common timeline via forward-fill, then average
        low_df = pd.concat(low_curves, axis=1).sort_index().ffill().fillna(0)
        high_df = pd.concat(high_curves, axis=1).sort_index().ffill().fillna(0)
        mean_low = low_df.mean(axis=1)
        mean_high = high_df.mean(axis=1)

        color = base_colors[i % len(base_colors)]
        marker = markers[i % len(markers)]

        curve_ax.plot(
            mean_low.index,
            mean_low.values,
            linestyle="--",
            marker=marker,
            markersize=5,
            color=color,
            label=lbl_low,
        )

        curve_ax.plot(
            mean_high.index,
            mean_high.values,
            linestyle="-",
            marker=marker,
            markersize=5,
            color=color,
            label=lbl_high,
        )

        x_pos = max(mean_low.index[-1], mean_high.index[-1]) + 0.1
        labels_to_plot.append(
            {
                "x": x_pos,
                "y": mean_low.iloc[-1],
                "text": lbl_low,
                "color": color,
                "fontsize": 13,
            }
        )
        labels_to_plot.append(
            {
                "x": x_pos,
                "y": mean_high.iloc[-1],
                "text": lbl_high,
                "color": color,
                "fontsize": 12,
            }
        )

    if labels_to_plot:
        labels_to_plot.sort(key=lambda item: item["y"])

        MIN_VERTICAL_SEPARATION = (
            curve_ax.get_yticks().max() - curve_ax.get_yticks().min()
        ) / 80
        print(curve_ax.get_yticks().max(), curve_ax.get_yticks().min())
        print(MIN_VERTICAL_SEPARATION)
        for i in range(1, len(labels_to_plot)):
            prev_label = labels_to_plot[i - 1]
            curr_label = labels_to_plot[i]

            if (curr_label["y"] - prev_label["y"]) < MIN_VERTICAL_SEPARATION:
                curr_label["y"] = curr_label["y"] + MIN_VERTICAL_SEPARATION / 2
                prev_label["y"] = prev_label["y"] - MIN_VERTICAL_SEPARATION / 2

    for label_info in labels_to_plot:
        curve_ax.text(
            label_info["x"],
            label_info["y"],
            label_info["text"],
            fontsize=label_info["fontsize"],
            color=label_info["color"],
            va="center",
            ha="left",
        )

    mean_fitters = []
    for feature in important_features:
        low_name, high_name = feature_groups[feature]
        low_at = np.round(np.mean(stats[feature][low_name]["at"], axis=0)).astype(int)
        low_ev = np.round(np.mean(stats[feature][low_name]["ev"], axis=0)).astype(int)
        high_at = np.round(np.mean(stats[feature][high_name]["at"], axis=0)).astype(int)
        high_ev = np.round(np.mean(stats[feature][high_name]["ev"], axis=0)).astype(int)

        mean_fitters.append(
            MeanFitter(f"{feature} - {low_name}", xticks, low_at, low_ev)
        )
        mean_fitters.append(
            MeanFitter(f"{feature} - {high_name}", xticks, high_at, high_ev)
        )

    curve_ax.set_xlim(0, 21)
    curve_ax.set_xticks(xticks)
    curve_ax.set_xlabel("Years", fontsize=19)
    curve_ax.set_ylabel("Mean Cumulative Hazard", fontsize=19)
    curve_ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    curve_ax.tick_params(axis="x", length=0)
    curve_fig.tight_layout()
    curve_fig.savefig(output_path + "cumulative_hazard.pdf", bbox_inches="tight")
    plt.show()
    plt.close(curve_fig)

    risk_fig, risk_ax = plt.subplots(figsize=(14, 5))
    risk_ax.set_xlim(0, 14)
    risk_ax.set_xticks(
        xticks,
        labels=[f"After {t} Years" for t in xticks],
        fontsize=13,
        fontweight="bold",
    )

    add_at_risk_counts(
        *mean_fitters,
        ax=risk_ax,
        labels=[mf.name for mf in mean_fitters],
        fontsize=13,
        rows_to_show=["At risk", "Events"],
    )

    risk_ax.tick_params(axis="x", which="both", length=0)
    risk_ax.set_yticks([])
    for spine in risk_ax.spines.values():
        spine.set_visible(False)

    for ax in risk_fig.axes:
        if ax is not risk_ax:
            ax.tick_params(axis="x", which="both", length=0)
            ax.grid(False)

    plt.setp(risk_ax.get_xticklabels(), rotation=0, ha="center")
    risk_fig.tight_layout()
    risk_fig.savefig(output_path + "risk_table.pdf", bbox_inches="tight")
    plt.close(risk_fig)
