# Classificando a Qualidade de Vinhos com Machine Learning
## Entregáveis
- 🎥 Vídeo executivo: [[link do YouTube]](https://youtu.be/Sl257ZFhSvk)
- 📊 Apresentação executiva: [[apresentacao_executiva.pdf]](https://github.com/rodrigolunaa/FIAP-Tech-Challenge-Fase-2-Wine-quality-classification/blob/main/wine-quality-classification/wine-quality-classification/results/apresentacao_executiva.pdf)
- 📓 Notebook: [[notebooks/wine_quality_classification.ipynb]](https://github.com/rodrigolunaa/FIAP-Tech-Challenge-Fase-2-Wine-quality-classification/blob/main/wine-quality-classification/wine-quality-classification/notebooks/wine_quality_classification.ipynb)

Tech Challenge — Fase 2 (POSTECH). Pipeline de classificação binária para prever se um vinho é de **Alta Qualidade** (nota ≥ 7) ou **Baixa/Média Qualidade** (nota < 7), a partir de suas características físico-químicas.

## Contexto

A avaliação sensorial de vinhos por especialistas é subjetiva e demorada. Este projeto usa dados físico-químicos objetivos (acidez, teor alcoólico, densidade, dióxido de enxofre etc.) para treinar modelos preditivos que apoiem decisões de produção e padronização de qualidade.

**Fonte dos dados:** [Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — 1.143 amostras.

## Estrutura do repositório

```
wine-quality-classification/
│
├── data/                              # Base de dados utilizada (WineQT.csv)
├── notebooks/
│   └── wine_quality_classification.ipynb   # Notebook com toda a análise e modelagem
├── src/
│   └── run_pipeline.py                # Script standalone que reproduz todo o pipeline
├── results/                            # Gráficos, métricas (metrics.json) e classification reports
├── requirements.txt
└── README.md
```

## Como reproduzir

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Opção 1: notebook interativo
jupyter notebook notebooks/wine_quality_classification.ipynb

# Opção 2: script direto (gera tudo em results/)
cd wine-quality-classification
python src/run_pipeline.py
```

## Resumo da abordagem

1. **EDA:** distribuição das variáveis, análise de outliers (IQR), matriz de correlação, balanceamento de classes (~13,9% de vinhos de Alta Qualidade — dataset desbalanceado).
2. **Pré-processamento:** tratamento de outliers por *capping* (winsorização), sem remoção de linhas; padronização (`StandardScaler`) para o SVM.
3. **Feature engineering:** 4 variáveis derivadas com justificativa enológica (`acid_balance`, `free_sulfur_ratio`, `alcohol_density`, `acidity_alcohol_interaction`).
4. **Modelagem:** Random Forest e SVM (kernel RBF), ambos com `class_weight="balanced"` e hiperparâmetros otimizados via `GridSearchCV` (validação cruzada estratificada, 5 folds).
5. **Avaliação:** acurácia, precisão, recall, F1-score, ROC-AUC, matriz de confusão e validação cruzada.
6. **Interpretação:** importância de variáveis (nativa no Random Forest; por permutação no SVM).

## Principais resultados

| Métrica (teste) | Random Forest | SVM (RBF) |
|---|---|---|
| Acurácia | 0.895 | 0.891 |
| Precisão | 0.625 | 0.621 |
| Recall | 0.625 | 0.563 |
| F1-score | 0.625 | 0.590 |
| ROC-AUC | 0.893 | 0.873 |

**Random Forest apresentou o melhor desempenho geral**, com maior estabilidade em validação cruzada. As variáveis mais influentes em ambos os modelos foram `alcohol` e `volatile acidity`, coerente com a literatura enológica.

Detalhes completos, gráficos e discussão estão no notebook (`notebooks/wine_quality_classification.ipynb`) e na apresentação executiva (`results/apresentacao_executiva.pdf`).

## Autoria

Projeto desenvolvido como parte do Tech Challenge — Fase 2 (POSTECH — Data Analytics).
