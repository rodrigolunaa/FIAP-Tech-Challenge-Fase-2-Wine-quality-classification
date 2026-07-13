"""
Pipeline completo - Classificação de Qualidade de Vinho
Gera todos os gráficos e métricas usados no notebook e na apresentação executiva.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, RocCurveDisplay
)
from sklearn.inspection import permutation_importance

RESULTS = "results"
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 1. CARGA E LIMPEZA
# ---------------------------------------------------------------------------
df = pd.read_csv("data/WineQT.csv")
df = df.drop(columns=["Id"])  # identificador da amostra, sem valor preditivo

feature_cols = [c for c in df.columns if c != "quality"]

missing = df.isnull().sum().sum()
dup_rows = df.duplicated().sum()

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------

# 2.1 Distribuição de cada variável
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#7B2D26")
    axes[i].set_title(col, fontsize=11)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{RESULTS}/01_distribuicoes.png", bbox_inches="tight")
plt.close()

# 2.2 Boxplots (outliers)
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#C9A227")
    axes[i].set_title(col, fontsize=11)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{RESULTS}/02_boxplots_outliers.png", bbox_inches="tight")
plt.close()

# outlier count via IQR
outlier_summary = {}
for col in feature_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    outlier_summary[col] = {"n_outliers": n_out, "pct": round(100 * n_out / len(df), 2),
                             "low": round(low, 3), "high": round(high, 3)}

# 2.3 Correlação
corr = df.corr(numeric_only=True)
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlação")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_correlacao.png", bbox_inches="tight")
plt.close()

corr_with_target = corr["quality"].drop("quality").sort_values(ascending=False)

# 2.4 Classe alvo binária e balanceamento
df["high_quality"] = (df["quality"] >= 7).astype(int)
class_counts = df["high_quality"].value_counts().sort_index()
class_pct = (class_counts / len(df) * 100).round(2)

plt.figure(figsize=(6, 4.5))
ax = sns.countplot(x="high_quality", data=df, palette=["#7B2D26", "#C9A227"])
ax.set_xticklabels(["Baixa/Média (<7)", "Alta (>=7)"])
ax.set_xlabel("")
ax.set_ylabel("Nº de amostras")
ax.set_title("Balanceamento das classes")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.savefig(f"{RESULTS}/04_balanceamento_classes.png", bbox_inches="tight")
plt.close()

# 2.5 Boxplots das top variáveis vs classe alvo
top_feats = corr_with_target.abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(top_feats):
    sns.boxplot(x="high_quality", y=col, data=df, ax=axes[i], palette=["#7B2D26", "#C9A227"])
    axes[i].set_xticklabels(["Baixa/Média", "Alta"])
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_top_features_vs_target.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------
df_clean = df.copy()

# Tratamento de outliers: CAPPING (winsorização) via IQR, não remoção.
# Motivo: dataset pequeno (1143 amostras) e a classe "Alta qualidade" já é
# minoritária (~13,9%). Remover linhas poderia eliminar justamente vinhos
# de qualidade alta/baixa que são naturalmente mais extremos em algumas
# variáveis físico-químicas, enviesando o modelo. O capping preserva o
# volume amostral e reduz a influência de valores extremos sem descartá-los.
for col in feature_cols:
    q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df_clean[col] = df_clean[col].clip(lower=low, upper=high)

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# Justificativas de domínio (enologia):
# - acid_balance: equilíbrio entre acidez fixa e volátil; vinhos com alta
#   acidez volátil relativa tendem a ter defeitos sensoriais (gosto de vinagre).
# - free_sulfur_ratio: fração do SO2 total que permanece livre (ativo como
#   conservante/antioxidante) — mais informativo que os valores absolutos.
# - alcohol_density: álcool e densidade são fisicamente relacionados
#   (mais álcool -> menor densidade); a razão captura esse efeito conjunto.
# - total_acidity_alcohol: interação entre acidez fixa e álcool, dois dos
#   fatores mais associados a vinhos bem avaliados.
df_clean["acid_balance"] = df_clean["fixed acidity"] / (df_clean["volatile acidity"] + 1e-6)
df_clean["free_sulfur_ratio"] = df_clean["free sulfur dioxide"] / (df_clean["total sulfur dioxide"] + 1e-6)
df_clean["alcohol_density"] = df_clean["alcohol"] / df_clean["density"]
df_clean["acidity_alcohol_interaction"] = df_clean["fixed acidity"] * df_clean["alcohol"]

engineered_feats = ["acid_balance", "free_sulfur_ratio", "alcohol_density", "acidity_alcohol_interaction"]
all_feature_cols = feature_cols + engineered_feats

# ---------------------------------------------------------------------------
# 5. SPLIT + ESCALONAMENTO
# ---------------------------------------------------------------------------
X = df_clean[all_feature_cols]
y = df_clean["high_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------------
# 6. MODELAGEM - RANDOM FOREST
# ---------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 8, 12],
    "min_samples_leaf": [1, 3],
}
rf_base = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_grid = GridSearchCV(rf_base, rf_param_grid, scoring="f1", cv=cv, n_jobs=-1)
rf_grid.fit(X_train, y_train)  # RF não exige escalonamento
rf_best = rf_grid.best_estimator_

# ---------------------------------------------------------------------------
# 7. MODELAGEM - SVM
# ---------------------------------------------------------------------------
svm_param_grid = {
    "C": [1, 5, 10],
    "gamma": ["scale", 0.01, 0.1],
    "kernel": ["rbf"],
}
svm_base = SVC(class_weight="balanced", probability=True, random_state=42)
svm_grid = GridSearchCV(svm_base, svm_param_grid, scoring="f1", cv=cv, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)  # SVM exige escalonamento
svm_best = svm_grid.best_estimator_

# ---------------------------------------------------------------------------
# 8. AVALIAÇÃO
# ---------------------------------------------------------------------------
def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    cv_scores = cross_validate(
        model, X_tr, y_tr, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
        "cv_f1_mean": cv_scores["test_f1"].mean(),
        "cv_f1_std": cv_scores["test_f1"].std(),
        "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
        "cv_roc_auc_std": cv_scores["test_roc_auc"].std(),
    }
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=["Baixa/Média", "Alta"])
    return metrics, cm, report, y_pred, y_proba

rf_metrics, rf_cm, rf_report, rf_pred, rf_proba = evaluate(
    rf_best, X_train, y_train, X_test, y_test, "Random Forest"
)
svm_metrics, svm_cm, svm_report, svm_pred, svm_proba = evaluate(
    svm_best, X_train_scaled, y_train, X_test_scaled, y_test, "SVM"
)

# Matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, title in zip(axes, [rf_cm, svm_cm], ["Random Forest", "SVM (RBF)"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r", cbar=False, ax=ax,
                xticklabels=["Baixa/Média", "Alta"], yticklabels=["Baixa/Média", "Alta"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
plt.tight_layout()
plt.savefig(f"{RESULTS}/06_matrizes_confusao.png", bbox_inches="tight")
plt.close()

# Curvas ROC
plt.figure(figsize=(6.5, 5.5))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_metrics['roc_auc']:.3f})", color="#7B2D26", lw=2)
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={svm_metrics['roc_auc']:.3f})", color="#C9A227", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
plt.xlabel("Taxa de Falso Positivo")
plt.ylabel("Taxa de Verdadeiro Positivo")
plt.title("Curvas ROC - Comparação dos modelos")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/07_curvas_roc.png", bbox_inches="tight")
plt.close()

# Comparação de métricas (barras)
metrics_df = pd.DataFrame({
    "Random Forest": {k: rf_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    "SVM (RBF)": {k: svm_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
}).T

plt.figure(figsize=(9, 5))
metrics_df.plot(kind="bar", ax=plt.gca(), color=["#7B2D26", "#A8442A", "#C9752E", "#C9A227", "#8C8C56"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Comparação de métricas entre modelos")
plt.legend(loc="lower right", ncol=3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{RESULTS}/08_comparacao_metricas.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. INTERPRETAÇÃO - IMPORTÂNCIA DE VARIÁVEIS
# ---------------------------------------------------------------------------
rf_importances = pd.Series(rf_best.feature_importances_, index=all_feature_cols).sort_values(ascending=False)

perm = permutation_importance(svm_best, X_test_scaled, y_test, n_repeats=20, random_state=42, scoring="f1")
svm_importances = pd.Series(perm.importances_mean, index=all_feature_cols).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
rf_importances.head(10).sort_values().plot(kind="barh", ax=axes[0], color="#7B2D26")
axes[0].set_title("Random Forest - Importância (Gini)")
svm_importances.head(10).sort_values().plot(kind="barh", ax=axes[1], color="#C9A227")
axes[1].set_title("SVM - Importância por permutação")
plt.tight_layout()
plt.savefig(f"{RESULTS}/09_importancia_variaveis.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 10. SALVAR RESULTADOS EM JSON/CSV
# ---------------------------------------------------------------------------
results = {
    "dataset": {
        "n_amostras": len(df),
        "n_features_originais": len(feature_cols),
        "n_features_apos_engenharia": len(all_feature_cols),
        "valores_faltantes": int(missing),
        "linhas_duplicadas_completas": int(dup_rows),
        "classe_alta_qualidade_pct": float(class_pct.get(1, 0)),
        "classe_baixa_media_pct": float(class_pct.get(0, 0)),
    },
    "outliers_iqr": outlier_summary,
    "correlacao_com_quality": corr_with_target.round(3).to_dict(),
    "melhores_hiperparametros": {
        "random_forest": rf_grid.best_params_,
        "svm": svm_grid.best_params_,
    },
    "metricas": {
        "random_forest": {k: round(float(v), 4) for k, v in rf_metrics.items()},
        "svm": {k: round(float(v), 4) for k, v in svm_metrics.items()},
    },
    "importancia_variaveis": {
        "random_forest": rf_importances.round(4).to_dict(),
        "svm_permutation": svm_importances.round(4).to_dict(),
    },
}

with open(f"{RESULTS}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(f"{RESULTS}/classification_report_rf.txt", "w") as f:
    f.write(rf_report)
with open(f"{RESULTS}/classification_report_svm.txt", "w") as f:
    f.write(svm_report)

print("Pipeline concluído com sucesso.")
print(json.dumps(results["metricas"], indent=2))
print("Melhores hiperparâmetros:", results["melhores_hiperparametros"])
