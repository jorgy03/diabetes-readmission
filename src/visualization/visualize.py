import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import FIGURES_DIR

ACCENT_COLOR = "#2f6690"


def plot_target_balance(df, target_column, save_as=None):
    fig, ax = plt.subplots()
    counts = df[target_column].value_counts().sort_index()
    labels = ["No (0)", "Readmitted <30d (1)"]
    ax.bar(labels, counts.values, color=ACCENT_COLOR)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylabel("Encounters")
    ax.set_title("Readmission (30-day) class balance")
    sns.despine(ax=ax)
    _save(fig, save_as)


def plot_missingness(df, save_as=None, top_n=15):
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(missing))))
    ax.barh(missing.index[::-1], missing.values[::-1] * 100, color=ACCENT_COLOR)
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    sns.despine(ax=ax)
    _save(fig, save_as)


def plot_readmission_rate_by(df, group_column, target_column, save_as=None):
    rate = df.groupby(group_column)[target_column].mean().sort_index()
    fig, ax = plt.subplots()
    ax.bar(rate.index.astype(str), rate.values * 100, color=ACCENT_COLOR)
    ax.set_ylabel("Readmitted <30d rate (%)")
    ax.set_xlabel(group_column)
    ax.set_title(f"Readmission rate by {group_column}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine(ax=ax)
    _save(fig, save_as)


def plot_feature_importance(importances_df, save_as=None):
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(importances_df))))
    ordered = importances_df.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"], color=ACCENT_COLOR)
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances")
    sns.despine(ax=ax)
    _save(fig, save_as)


def plot_confusion_matrix(model, X_test, y_test, save_as=None):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    _save(fig, save_as)


def plot_roc_curve(model, X_test, y_test, save_as=None):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    _save(fig, save_as)


def _save(fig, save_as):
    if save_as:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / save_as, bbox_inches="tight")
