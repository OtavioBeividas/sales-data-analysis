
import argparse
import os
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
														 confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
plt.rcParams.update({"font.size": 12})


@dataclass
class Config:
	n_samples: int = 1000
	n_features: int = 8
	n_informative: int = 4
	random_state: int = 42
	test_size: float = 0.25


def generate_dataset(cfg: Config) -> pd.DataFrame:
	X, y = make_classification(
		n_samples=cfg.n_samples,
		n_features=cfg.n_features,
		n_informative=cfg.n_informative,
		n_redundant=0,
		n_repeated=0,
		n_classes=2,
		flip_y=0.02,
		class_sep=1.0,
		random_state=cfg.random_state,
	)

	cols = [f"feature_{i+1}" for i in range(cfg.n_features)]
	df = pd.DataFrame(X, columns=cols)
	df["target"] = y
	return df


def load_from_csv(path: str, target: str = None) -> pd.DataFrame:
	df = pd.read_csv(path)
	if target is not None and target not in df.columns:
		raise ValueError(f"Coluna target '{target}' não encontrada no CSV")
	if target is None:
		for cand in ("target", "label", "classe", "class"):
			if cand in df.columns:
				target = cand
				break
	if target is not None:
		if target not in df.columns:
			raise ValueError(f"Target detectado não existe: {target}")
	else:
		df["target"] = 0
		target = "target"
	return df


def basic_eda(df: pd.DataFrame) -> None:
	print("\n--- Cabeçalho (5 primeiras linhas) ---")
	print(df.head().to_string())

	print("\n--- Estatísticas descritivas ---")
	print(df.describe().T)

	print("\n--- Balanceamento de classes ---")
	if "target" in df.columns:
		print(df["target"].value_counts(normalize=True))
	else:
		print("Coluna 'target' não encontrada no dataset.")


def plot_correlations(df: pd.DataFrame, out_dir: str) -> None:
	numeric = df.select_dtypes(include=[np.number])
	corr = numeric.corr()
	mask = np.triu(np.ones_like(corr, dtype=bool))
	plt.figure(figsize=(10, 8))
	plt.title("Matriz de Correlação")
	sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	path = os.path.join(out_dir, "correlation_matrix.png")
	plt.savefig(path, dpi=150)
	plt.close()
	print(f"Matriz de correlação salva em: {path}")


def prepare_features(df: pd.DataFrame):
	if "target" not in df.columns:
		raise ValueError("DataFrame não contém coluna 'target'.")
	X = df.drop(columns=["target"]).select_dtypes(include=[np.number]).values
	y = df["target"].values
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	return X_scaled, y, scaler


def train_and_evaluate(X, y, cfg: Config, out_dir: str) -> None:
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
	)

	model = RandomForestClassifier(n_estimators=100, random_state=cfg.random_state)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]

	acc = accuracy_score(y_test, y_pred)
	roc = roc_auc_score(y_test, y_proba)

	print("\n--- Métricas do modelo ---")
	print(f"Acurácia: {acc:.4f}")
	print(f"ROC AUC: {roc:.4f}")
	print("\nClassificação detalhada:")
	print(classification_report(y_test, y_pred))

	cm = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
	plt.title("Matriz de Confusão")
	plt.xlabel("Predito")
	plt.ylabel("Verdadeiro")
	plt.tight_layout()
	path = os.path.join(out_dir, "confusion_matrix.png")
	plt.savefig(path, dpi=150)
	plt.close()
	print(f"Matriz de confusão salva em: {path}")


def save_feature_importance(model, feature_names, out_dir: str):
	try:
		importances = model.feature_importances_
	except Exception:
		return
	idx = np.argsort(importances)[::-1]
	plt.figure(figsize=(8, 6))
	sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
	plt.title("Importância das Features")
	plt.tight_layout()
	path = os.path.join(out_dir, "feature_importance.png")
	plt.savefig(path, dpi=150)
	plt.close()
	print(f"Feature importance salva em: {path}")


def ensure_out_dir(path: str = "output") -> str:
	os.makedirs(path, exist_ok=True)
	return path


def main():
	warnings.simplefilter(action="ignore", category=FutureWarning)
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", help="Caminho para CSV de entrada (opcional)")
	parser.add_argument("--target", help="Nome da coluna target no CSV (opcional)")
	args = parser.parse_args()

	cfg = Config()
	out_dir = ensure_out_dir()

	if args.input:
		df = load_from_csv(args.input, args.target)
	else:
		df = generate_dataset(cfg)

	csv_path = os.path.join(out_dir, "dataset_sample.csv")
	df.to_csv(csv_path, index=False)
	print(f"Dataset salvo em: {csv_path}")

	basic_eda(df)
	plot_correlations(df, out_dir)

	try:
		X, y, scaler = prepare_features(df)
		model = train_and_evaluate(X, y, cfg, out_dir)
		if model is not None:
			feature_names = df.select_dtypes(include=[np.number]).drop(columns=["target"]).columns
			save_feature_importance(model, feature_names, out_dir)
	except Exception as e:
		print(f"Pular treino/avaliação: {e}")


if __name__ == "__main__":
	main()

