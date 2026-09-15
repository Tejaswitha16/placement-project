import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data


# ============================================================
# CHART DIRECTORY
# ============================================================

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


# ============================================================
# RUN EDA
# ============================================================

def run_eda():

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = load_data()

    charts = []

    sns.set_style("whitegrid")


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    duplicates = int(
        data.duplicated().sum()
    )


    # ========================================================
    # NUMERIC DISTRIBUTIONS
    # ========================================================

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    # Keep only columns that actually exist

    hist_cols = [
        c for c in hist_cols
        if c in data.columns
    ]

    # Remove duplicate column names

    hist_cols = list(
        dict.fromkeys(hist_cols)
    )


    if hist_cols:

        filename = "numeric_distributions.png"

        numeric_data = data[hist_cols].copy()

        # Convert to numeric safely

        for col in hist_cols:
            numeric_data[col] = pd.to_numeric(
                numeric_data[col],
                errors="coerce"
            )

        # Create figure

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9)
        )

        axes = axes.flatten()

        for i, col in enumerate(hist_cols):

            axes[i].hist(
                numeric_data[col].dropna(),
                bins=20
            )

            axes[i].set_title(
                col
            )

            axes[i].set_xlabel(
                col
            )

            axes[i].set_ylabel(
                "Frequency"
            )

        # Hide unused axes

        for i in range(
            len(hist_cols),
            len(axes)
        ):
            axes[i].set_visible(False)

        fig.suptitle(
            "Numeric Feature Distributions",
            fontsize=16
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # CGPA DISTRIBUTION
    # ========================================================

    if "CGPA" in data.columns:

        filename = "cgpa_distribution.png"

        cgpa = pd.to_numeric(
            data["CGPA"],
            errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            cgpa,
            kde=True,
            ax=ax
        )

        if len(cgpa) > 0:

            ax.axvline(
                cgpa.mean(),
                linestyle="--",
                label="Mean"
            )

        ax.set_title(
            "CGPA Distribution"
        )

        ax.set_xlabel(
            "CGPA"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # BOXPLOTS
    # ========================================================

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "SoftSkillRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package"
    ]

    box_cols = [
        c for c in box_cols
        if c in data.columns
    ]

    box_cols = list(
        dict.fromkeys(box_cols)
    )


    for col in box_cols:

        filename = (
            f"boxplot_{col.replace(' ', '_')}.png"
        )

        values = pd.to_numeric(
            data[col],
            errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        sns.boxplot(
            x=values,
            ax=ax
        )

        ax.set_title(
            f"Boxplot - {col}"
        )

        ax.set_xlabel(
            col
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    filename = "correlation_heatmap.png"

    numeric_df = data.select_dtypes(
        include="number"
    ).copy()

    # Limit rows only for speed

    if len(numeric_df) > 2000:

        numeric_sample = numeric_df.sample(
            2000,
            random_state=42
        )

    else:

        numeric_sample = numeric_df


    corr = numeric_sample.corr()

    fig, ax = plt.subplots(
        figsize=(15, 11)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    fig.tight_layout()

    fig.savefig(
        _chart_path(filename),
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    charts.append(filename)


    # ========================================================
    # CGPA VS SALARY PACKAGE
    # ========================================================

    if (
        "CGPA" in data.columns
        and
        "Salary Package" in data.columns
    ):

        filename = "cgpa_salary.png"

        plot_data = data[
            [
                "CGPA",
                "Salary Package"
            ]
        ].copy()

        plot_data["CGPA"] = pd.to_numeric(
            plot_data["CGPA"],
            errors="coerce"
        )

        plot_data["Salary Package"] = pd.to_numeric(
            plot_data["Salary Package"],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        if len(plot_data) > 5000:

            plot_data = plot_data.sample(
                5000,
                random_state=42
            )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=plot_data,
            scatter_kws={
                "s": 20,
                "alpha": 0.5
            },
            ax=ax
        )

        ax.set_title(
            "CGPA vs Salary Package"
        )

        ax.set_xlabel(
            "CGPA"
        )

        ax.set_ylabel(
            "Salary Package"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # CODING TEST VS APTITUDE TEST
    # ========================================================

    if (
        "CodingTestScore" in data.columns
        and
        "AptitudeTestScore" in data.columns
    ):

        filename = "coding_aptitude.png"

        plot_data = data[
            [
                "CodingTestScore",
                "AptitudeTestScore"
            ]
        ].copy()

        plot_data["CodingTestScore"] = pd.to_numeric(
            plot_data["CodingTestScore"],
            errors="coerce"
        )

        plot_data["AptitudeTestScore"] = pd.to_numeric(
            plot_data["AptitudeTestScore"],
            errors="coerce"
        )

        plot_data = plot_data.dropna()

        if len(plot_data) > 5000:

            plot_data = plot_data.sample(
                5000,
                random_state=42
            )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.regplot(
            x="CodingTestScore",
            y="AptitudeTestScore",
            data=plot_data,
            scatter_kws={
                "s": 20,
                "alpha": 0.5
            },
            ax=ax
        )

        ax.set_title(
            "Coding Test Score vs Aptitude Test Score"
        )

        ax.set_xlabel(
            "Coding Test Score"
        )

        ax.set_ylabel(
            "Aptitude Test Score"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # CATEGORICAL COUNTS
    # ========================================================

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [
        c for c in cat_cols
        if c in data.columns
    ]


    for col in cat_cols:

        filename = (
            f"{col.lower()}_count.png"
        )

        counts = (
            data[col]
            .astype(str)
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax
        )

        ax.set_title(
            f"{col} Count"
        )

        ax.set_xlabel(
            col
        )

        ax.set_ylabel(
            "Count"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # GENDER VS PLACEMENT
    # ========================================================

    if (
        "Gender" in data.columns
        and
        "PlacementStatus" in data.columns
    ):

        filename = "gender_vs_placement.png"

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.countplot(
            x="Gender",
            hue="PlacementStatus",
            data=data,
            ax=ax
        )

        ax.set_title(
            "Placement Status by Gender"
        )

        ax.set_xlabel(
            "Gender"
        )

        ax.set_ylabel(
            "Count"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # COLLEGE TIER VS PLACEMENT
    # ========================================================

    if (
        "CollegeTier" in data.columns
        and
        "PlacementStatus" in data.columns
    ):

        filename = (
            "college_tier_vs_placement.png"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.countplot(
            x="CollegeTier",
            hue="PlacementStatus",
            data=data,
            ax=ax
        )

        ax.set_title(
            "Placement Status by College Tier"
        )

        ax.set_xlabel(
            "College Tier"
        )

        ax.set_ylabel(
            "Count"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # SGPA TREND
    # ========================================================

    sgpa_cols = [
        f"SGPA_Sem{i}"
        for i in range(1, 9)
        if f"SGPA_Sem{i}" in data.columns
    ]


    if sgpa_cols:

        filename = "sgpa_trend.png"

        sgpa_data = data[
            sgpa_cols
        ].copy()

        for col in sgpa_cols:

            sgpa_data[col] = pd.to_numeric(
                sgpa_data[col],
                errors="coerce"
            )

        avg_sgpa = (
            sgpa_data
            .mean()
        )

        semesters = [
            f"Sem{i}"
            for i in range(
                1,
                len(avg_sgpa) + 1
            )
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            semesters,
            avg_sgpa.values,
            marker="o",
            linewidth=2
        )

        ax.set_title(
            "Average SGPA Across Semesters"
        )

        ax.set_xlabel(
            "Semester"
        )

        ax.set_ylabel(
            "Average SGPA"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # SALARY DISTRIBUTION
    # ========================================================

    if "Salary Package" in data.columns:

        filename = "salary_distribution.png"

        salary = pd.to_numeric(
            data["Salary Package"],
            errors="coerce"
        ).dropna()

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            salary,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Salary Package Distribution"
        )

        ax.set_xlabel(
            "Salary Package"
        )

        ax.set_ylabel(
            "Frequency"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


        # ====================================================
        # SALARY BOXPLOT
        # ====================================================

        filename = "salary_boxplot.png"

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        sns.boxplot(
            x=salary,
            ax=ax
        )

        ax.set_title(
            "Salary Package Boxplot"
        )

        ax.set_xlabel(
            "Salary Package"
        )

        fig.tight_layout()

        fig.savefig(
            _chart_path(filename),
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        charts.append(filename)


    # ========================================================
    # PAIRPLOT
    # ========================================================

    pair_cols = [
        "CGPA",
        "AttendancePercent",
        "PlacementStatus"
    ]

    pair_cols = [
        c for c in pair_cols
        if c in data.columns
    ]


    if len(pair_cols) >= 3:

        filename = "pairplot.png"

        pair_data = data[
            pair_cols
        ].copy()

        pair_data["CGPA"] = pd.to_numeric(
            pair_data["CGPA"],
            errors="coerce"
        )

        pair_data["AttendancePercent"] = pd.to_numeric(
            pair_data["AttendancePercent"],
            errors="coerce"
        )

        pair_data = pair_data.dropna()

        if len(pair_data) > 500:

            pair_data = pair_data.sample(
                500,
                random_state=42
            )

        if len(pair_data) > 0:

            g = sns.pairplot(
                pair_data,
                hue="PlacementStatus"
            )

            g.fig.suptitle(
                "Relationship Between CGPA, Attendance and Placement",
                y=1.02
            )

            g.savefig(
                _chart_path(filename),
                dpi=120,
                bbox_inches="tight"
            )

            plt.close(
                g.fig
            )

            charts.append(filename)


    # ========================================================
    # REMOVE OLD UNUSED CHARTS
    # ========================================================

    old_unused_charts = [
        "missing_values.png",
        "missing_heatmap.png",
        "placement_status.png"
    ]

    for filename in old_unused_charts:

        filepath = _chart_path(
            filename
        )

        if os.path.exists(filepath):

            try:
                os.remove(filepath)
            except Exception:
                pass


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "rows": len(data),

        "columns": len(data.columns),

        "duplicates": duplicates,

        "charts": charts

    }