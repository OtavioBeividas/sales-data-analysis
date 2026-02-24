import os
from textwrap import dedent

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st

from arquivo import Config, generate_dataset, load_from_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Dashboard - Treino de Análise de Dados", layout="wide")


def load_dataset(out_path: str = "output/dataset_sample.csv"):
    if out_path and os.path.exists(out_path):
        return __import__("pandas").read_csv(out_path)
    cfg = Config()
    return generate_dataset(cfg)


def plot_correlation_plotly(df):
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig = ff.create_annotated_heatmap(
        z=corr.values.round(2).tolist(),
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        showscale=True,
    )
    fig.update_layout(height=600, width=700)
    return fig


def plot_confusion_plotly(cm):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(height=350, width=400)
    return fig


def main():
    st.title("Dashboard — Treino de Modelo (Casos Reais)")

    st.sidebar.header("Carregar dados")
    uploaded = st.sidebar.file_uploader("Faça upload de um CSV", type=["csv"])
    use_dataset = st.sidebar.selectbox("Ou usar dataset salvo", ("output/dataset_sample.csv", "Gerar novo"))

    if uploaded is not None:
        import pandas as pd

        df = pd.read_csv(uploaded)
        st.sidebar.success("CSV carregado")
    elif use_dataset == "output/dataset_sample.csv" and os.path.exists("output/dataset_sample.csv"):
        df = load_dataset("output/dataset_sample.csv")
    else:
        df = generate_dataset(Config())

    st.sidebar.subheader("Colunas / Target")
    cols = list(df.columns)
    target_col = st.sidebar.selectbox("Coluna target", options=[c for c in cols if df[c].dtype != 'O'] + [None])
    if target_col is None and "target" in df.columns:
        target_col = "target"

    st.subheader("Visualização do Dataset")
    num_display = st.slider("Linhas exibidas", 5, 500, 50)
    st.dataframe(df.head(num_display).round(4))

    st.markdown("**Estatísticas descritivas (numéricas)**")
    st.dataframe(df.select_dtypes(include=["number"]).describe().T.round(4))

    st.subheader("Informação de valores faltantes")
    missing = df.isnull().sum().sort_values(ascending=False)
    st.bar_chart(missing[missing > 0])

    st.subheader("Matriz de Correlação")
    corr_fig = plot_correlation_plotly(df)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Não há colunas numéricas suficientes para correlação.")

    st.sidebar.subheader("Treinar modelo")
    n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100)
    test_size = st.sidebar.slider("test_size", 0.1, 0.5, 0.25)
    random_state = st.sidebar.number_input("random_state", value=42)
    retrain = st.sidebar.button("Treinar")

    if retrain:
        if target_col is None:
            st.error("Selecione uma coluna target válida (numérica/categórica).")
            return

        X_df = df.drop(columns=[target_col]).select_dtypes(include=["number"]).fillna(0)
        y = df[target_col].fillna(0)

        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y if len(set(y))>1 else None
        )

        model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and X_test.shape[0]>0 and len(set(y))>1 else None

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        st.subheader("Métricas do Modelo")
        st.write(f"Acurácia: {acc:.4f}")
        if roc is not None:
            st.write(f"ROC AUC: {roc:.4f}")

        st.subheader("Relatório de Classificação")
        cr = classification_report(y_test, y_pred, output_dict=True)
        import pandas as pd

        st.dataframe(pd.DataFrame(cr).T.round(4))

        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(plot_confusion_plotly(cm))

        try:
            fi = model.feature_importances_
            fi_df = pd.DataFrame({"feature": X_df.columns, "importance": fi}).sort_values("importance", ascending=False)
            st.subheader("Importância das features")
            st.plotly_chart(px.bar(fi_df, x="importance", y="feature", orientation="h"))
        except Exception:
            pass

        st.success("Treinamento concluído.")


if __name__ == "__main__":
    main()
