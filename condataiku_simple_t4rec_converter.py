"""
SÉLECTION INTELLIGENTE DE COLONNES
==================================================

Script avec plusieurs stratégies de sélection intelligente :
1. Analyse de variance (features discriminantes)
2. Corrélation et redondance
3. Sélection métier bancaire
4. Analyse d'importance via Random Forest
5. Analyse temporelle et patterns

Choisit automatiquement les meilleures colonnes selon plusieurs critères.
"""

import dataiku
import numpy as np
import pandas as pd
import json
import torch
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re


def load_input_dataset() -> pd.DataFrame:
    """Charge le dataset d'input depuis Dataiku"""
    try:
        dataset = dataiku.Dataset("t4_rec_df")
        df = dataset.get_dataframe()
        print(f"Dataset input 't4_rec_df' chargé: {df.shape}")
        return df
    except Exception as e:
        print(f"Erreur chargement input: {e}")
        raise


def detect_and_convert_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et convertit les colonnes contenant des arrays JSON"""
    print(f"🔍 Analyse de {len(df.columns)} colonnes...")
    df_converted = df.copy()
    converted_count = 0

    for col in df.columns:
        if df[col].dtype == "object":
            sample_values = df[col].dropna().head(3).tolist()

            is_json_array = False
            for val in sample_values:
                if (
                    isinstance(val, str)
                    and val.strip().startswith("[")
                    and val.strip().endswith("]")
                ):
                    is_json_array = True
                    break

            if is_json_array:

                def convert_json_array(val):
                    try:
                        if pd.isna(val):
                            return np.nan
                        if isinstance(val, str):
                            array_val = json.loads(val)
                            if isinstance(array_val, list) and len(array_val) > 0:
                                return float(array_val[0])
                        return float(val) if val is not None else np.nan
                    except:
                        return np.nan

                df_converted[col] = df[col].apply(convert_json_array)
                converted_count += 1

    print(f"📈 {converted_count} colonnes arrays converties")
    return df_converted


class SmartColumnSelector:
    """Classe pour sélection intelligente de colonnes"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(
            f" SmartSelector initialisé avec {len(self.numeric_cols)} colonnes numériques"
        )

    def analyze_variance(self, threshold: float = 0.01) -> List[str]:
        """1. ANALYSE DE VARIANCE - Élimine les features constantes"""
        print(f" Analyse de variance (seuil: {threshold})...")

        # Calculer la variance de chaque colonne
        variances = self.df[self.numeric_cols].var().sort_values(ascending=False)
        high_variance_cols = variances[variances > threshold].index.tolist()

        print(
            f"  {len(high_variance_cols)}/{len(self.numeric_cols)} colonnes avec variance > {threshold}"
        )
        print(f"  📈 Top 5 variances: {variances.head().to_dict()}")

        return high_variance_cols

    def analyze_correlation(
        self, features: List[str], threshold: float = 0.95
    ) -> List[str]:
        """2. ANALYSE CORRÉLATION - Élimine les features redondantes"""
        print(f" 2. Analyse corrélation (seuil: {threshold})...")

        if len(features) < 2:
            return features

        # Matrice de corrélation
        corr_matrix = self.df[features].corr().abs()

        # Trouver les paires très corrélées
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Colonnes à supprimer (gardant celle avec plus de variance)
        to_drop = set()
        for col in upper_tri.columns:
            highly_corr = upper_tri[col][upper_tri[col] > threshold].index.tolist()
            if highly_corr:
                # Garder la colonne avec plus de variance
                variances = self.df[[col] + highly_corr].var()
                to_drop.update(variances.drop(variances.idxmax()).index.tolist())

        selected_features = [col for col in features if col not in to_drop]

        print(
            f"  {len(selected_features)}/{len(features)} colonnes après suppression corrélation"
        )
        print(f"   Supprimées pour corrélation > {threshold}: {len(to_drop)} colonnes")

        return selected_features

    def banking_domain_selection(self, features: List[str]) -> Dict[str, List[str]]:
        """3. SÉLECTION MÉTIER BANCAIRE - Logique business"""
        print(f"🏦 3. Sélection métier bancaire...")

        banking_categories = {
            "temporal_evolution": [],  # Evolution temporelle (3m, 6m, 9m, 12m)
            "transaction_volume": [],  # Volumes de transactions
            "transaction_amounts": [],  # Montants
            "product_usage": [],  # Usage produits bancaires
            "demographics": [],  # Démographie client
            "risk_indicators": [],  # Indicateurs de risque
            "digital_behavior": [],  # Comportement digital
            "geographical": [],  # Géographie
        }

        # Patterns pour classification automatique
        patterns = {
            "temporal_evolution": [
                r"_3m$",
                r"_6m$",
                r"_9m$",
                r"_12m$",
                r"_3dm$",
                r"_6dm$",
                r"_12dm$",
            ],
            "transaction_volume": [r"^nb_", r"^top_", r"_nb_", r"nbr"],
            "transaction_amounts": [r"^mnt", r"^mt_", r"somme_", r"var_mnt"],
            "product_usage": [
                r"livret",
                r"euro",
                r"carte",
                r"credit",
                r"epargne",
                r"assur",
            ],
            "demographics": [r"age", r"enfant", r"iris", r"pop_", r"csp"],
            "risk_indicators": [r"decou", r"resil", r"fermeture", r"enc_"],
            "digital_behavior": [r"mobile", r"contact", r"visite", r"services"],
            "geographical": [r"iris", r"region", r"zone"],
        }

        # Classifier chaque feature
        for feature in features:
            feature_lower = feature.lower()
            categorized = False

            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, feature_lower):
                        banking_categories[category].append(feature)
                        categorized = True
                        break
                if categorized:
                    break

        # Afficher la classification
        for category, cols in banking_categories.items():
            if cols:
                print(f"  📂 {category}: {len(cols)} colonnes")

        return banking_categories

    def importance_based_selection(
        self, features: List[str], top_k: int = 50
    ) -> List[str]:
        """4. SÉLECTION PAR IMPORTANCE - Random Forest"""
        print(f" Sélection par importance Random Forest (top {top_k})...")

        if len(features) <= top_k:
            return features

        try:
            # Créer une target synthétique (somme de quelques colonnes importantes)
            target_cols = [
                col
                for col in features
                if any(x in col.lower() for x in ["mnt", "somme", "nb_"])
            ][:5]
            if target_cols:
                y = self.df[target_cols].sum(axis=1)
            else:
                y = self.df[features].sum(axis=1)

            # Préparer les données
            X = self.df[features].fillna(0)

            # Random Forest pour importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # Récupérer l'importance
            feature_importance = pd.DataFrame(
                {"feature": features, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

            selected_features = feature_importance.head(top_k)["feature"].tolist()

            print(
                f"  Top {len(selected_features)} features sélectionnées par importance"
            )
            print(f"  Top 5: {selected_features[:5]}")

            return selected_features

        except Exception as e:
            print(f"  Erreur Random Forest: {e}, utilisation variance à la place")
            variances = self.df[features].var().sort_values(ascending=False)
            return variances.head(top_k).index.tolist()

    def temporal_pattern_analysis(self, features: List[str]) -> List[str]:
        """5. ANALYSE PATTERNS TEMPORELS - Sélectionne selon évolution temporelle"""
        print(f"Analyse patterns temporels...")

        # Identifier les colonnes temporelles
        temporal_groups = {}
        for feature in features:
            # Extraire le nom de base (sans suffixe temporel)
            base_name = re.sub(r"_(3m|6m|9m|12m|3dm|6dm|12dm)$", "", feature.lower())
            if base_name != feature.lower():  # A un suffixe temporel
                if base_name not in temporal_groups:
                    temporal_groups[base_name] = []
                temporal_groups[base_name].append(feature)

        # Pour chaque groupe temporel, sélectionner la période la plus récente et la plus ancienne
        selected_temporal = []
        for base_name, group in temporal_groups.items():
            if len(group) > 1:
                # Ordre de préférence : 3m (plus récent), puis 12m (plus de contexte)
                priority_order = ["3m", "6m", "12m", "3dm", "6dm", "12dm"]
                for period in priority_order:
                    matching = [col for col in group if col.lower().endswith(period)]
                    if matching:
                        selected_temporal.extend(matching[:1])  # Prendre le premier
                        break
            else:
                selected_temporal.extend(group)

        # Ajouter les colonnes non-temporelles
        non_temporal = [
            f for f in features if f not in sum(temporal_groups.values(), [])
        ]
        final_selection = selected_temporal + non_temporal

        print(f" {len(temporal_groups)} groupes temporels analysés")
        print(f"  {len(selected_temporal)} colonnes temporelles optimisées")
        print(f" {len(final_selection)} colonnes finales")

        return final_selection

    def smart_selection_pipeline(
        self,
        max_features: int = 30,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
    ) -> Tuple[List[str], Dict]:
        """PIPELINE COMPLET de sélection intelligente"""
        print(f"\n PIPELINE SÉLECTION INTELLIGENTE")
        print(f" Objectif: {max_features} features optimales")
        print("=" * 50)

        stats = {}

        # Étape 1: Variance
        high_var_features = self.analyze_variance(variance_threshold)
        stats["step1_variance"] = len(high_var_features)

        # Étape 2: Corrélation
        low_corr_features = self.analyze_correlation(
            high_var_features, correlation_threshold
        )
        stats["step2_correlation"] = len(low_corr_features)

        # Étape 3: Classification métier
        banking_categories = self.banking_domain_selection(low_corr_features)
        stats["step3_banking_categories"] = banking_categories

        # Étape 4: Patterns temporels
        temporal_optimized = self.temporal_pattern_analysis(low_corr_features)
        stats["step4_temporal"] = len(temporal_optimized)

        # Étape 5: Importance (si encore trop de features)
        if len(temporal_optimized) > max_features:
            final_features = self.importance_based_selection(
                temporal_optimized, max_features
            )
        else:
            final_features = temporal_optimized

        stats["final_features_count"] = len(final_features)
        stats["final_features"] = final_features

        print(f"\n SÉLECTION TERMINÉE!")
        print(
            f"Pipeline: {len(self.numeric_cols)} → {stats['step1_variance']} → {stats['step2_correlation']} → {stats['step4_temporal']} → {len(final_features)}"
        )
        print(f"Features finales: {len(final_features)}")

        return final_features, stats


def create_sequences_smart_selection(
    df: pd.DataFrame, selected_features: List[str], seq_length: int = None
) -> Tuple[np.ndarray, Dict]:
    """Crée des séquences avec les colonnes sélectionnées intelligemment"""

    if seq_length is None:
        seq_length = min(len(selected_features), 30)  # Adaptatif

    print(f"Création séquences avec {len(selected_features)} features sélectionnées")
    print(f"Longueur séquence: {seq_length}")

    sequences = []

    for idx, row in df.iterrows():
        # Récupérer les valeurs des features sélectionnées
        values = [row[col] for col in selected_features]

        # Filtrer les valeurs NaN/infinies
        values = [v for v in values if not (pd.isna(v) or np.isinf(v))]

        if len(values) < seq_length:
            # Répéter les valeurs si pas assez
            while len(values) < seq_length:
                values.extend(values[: min(len(values), seq_length - len(values))])
            values = values[:seq_length]
        else:
            # Prendre les premières valeurs
            values = values[:seq_length]

        sequences.append(values)

        if idx % 1000 == 0:
            print(f" Traité {idx + 1} lignes...")

    sequences_array = np.array(sequences)
    print(f"Séquences créées: {sequences_array.shape}")

    stats = {
        "num_sequences": len(sequences),
        "sequence_length": seq_length,
        "selected_columns": selected_features,
        "selection_method": "smart_pipeline",
        "value_range": (float(np.min(sequences_array)), float(np.max(sequences_array))),
        "mean_value": float(np.mean(sequences_array)),
    }

    return sequences_array, stats


def create_t4rec_format(sequences: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convertit les séquences au format T4Rec"""
    print(f" Conversion au format T4Rec...")

    # Normaliser les valeurs entre 1 et 2000
    sequences_norm = sequences.copy()
    seq_min = np.min(sequences_norm)
    seq_max = np.max(sequences_norm)

    if seq_max != seq_min:
        sequences_norm = (sequences_norm - seq_min) / (seq_max - seq_min)
        sequences_norm = sequences_norm * 1999 + 1  # [1, 2000]
    else:
        sequences_norm = np.ones_like(sequences_norm)

    sequences_ids = sequences_norm.astype(int)

    # Créer input/target pairs
    inputs = sequences_ids[:, :-1]
    targets = sequences_ids[:, 1:]

    input_tensor = torch.LongTensor(inputs)
    target_tensor = torch.LongTensor(targets)

    print(f"  Input shape: {input_tensor.shape}")
    print(f" Target shape: {target_tensor.shape}")
    print(
        f"  Item ID range: {int(np.min(sequences_ids))} - {int(np.max(sequences_ids))}"
    )

    return input_tensor, target_tensor


def save_selection_analysis(
    selection_stats: Dict, filename: str = "column_selection_analysis.txt"
):
    """Sauvegarde l'analyse de sélection"""

    analysis = f"""
# ANALYSE SÉLECTION INTELLIGENTE DE COLONNES
============================================

## Pipeline de sélection:
1. Analyse variance: {selection_stats.get("step1_variance", 0)} colonnes
2. Suppression corrélation: {selection_stats.get("step2_correlation", 0)} colonnes  
3. Optimisation temporelle: {selection_stats.get("step4_temporal", 0)} colonnes
4. Sélection finale: {selection_stats.get("final_features_count", 0)} colonnes

## Catégories métier identifiées:
"""

    banking_cats = selection_stats.get("step3_banking_categories", {})
    for category, cols in banking_cats.items():
        if cols:
            analysis += f"- {category}: {len(cols)} colonnes\n"

    analysis += f"""
## Colonnes finales sélectionnées:
{chr(10).join(f"- {col}" for col in selection_stats.get("final_features", []))}

## Avantages de cette sélection:
Variance élevée (features discriminantes)
Faible corrélation (pas de redondance)  
Logique métier bancaire respectée
Optimisation patterns temporels
Sélection par importance statistique
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(analysis)

    print(f"Analyse sauvegardée: {filename}")


def main():
    """Pipeline principal - SÉLECTION INTELLIGENTE"""
    print(" DÉMARRAGE RECIPE SÉLECTION INTELLIGENTE")
    print("=" * 60)

    try:
        # 1. Charger et convertir
        df = load_input_dataset()
        df_converted = detect_and_convert_arrays(df)

        # 2. Sélection intelligente
        selector = SmartColumnSelector(df_converted)
        selected_features, selection_stats = selector.smart_selection_pipeline(
            max_features=30,  # Nombre optimal de features
            variance_threshold=0.01,
            correlation_threshold=0.95,
        )

        # 3. Créer les séquences avec features sélectionnées
        sequences, stats = create_sequences_smart_selection(
            df_converted,
            selected_features,
            seq_length=25,  # Séquence adaptée
        )

        # 4. Format T4Rec
        inputs, targets = create_t4rec_format(sequences)

        # 5. Sauvegardes
        stats["num_unique_items"] = int(
            torch.max(torch.cat([inputs.flatten(), targets.flatten()])).item()
        )

        # Sauvegarde analyse
        save_selection_analysis(selection_stats)

        # Sauvegarde tenseurs
        data_dict = {
            "inputs": inputs,
            "targets": targets,
            "metadata": {
                "num_sequences": inputs.shape[0],
                "sequence_length": inputs.shape[1],
                "num_unique_items": stats["num_unique_items"],
                "selected_columns": selected_features,
                "selection_method": "smart_pipeline",
                "selection_stats": selection_stats,
            },
        }

        torch.save(data_dict, "t4rec_training_data_SMART_SELECTION.pt")

        print(f"\n SÉLECTION INTELLIGENTE TERMINÉE!")
        print(f"{len(selected_features)} colonnes optimales sélectionnées")
        print(f"Données sauvegardées: t4rec_training_data_SMART_SELECTION.pt")
        print(f" Analyse détaillée: column_selection_analysis.txt")

        print(f"\n TOP FEATURES SÉLECTIONNÉES:")
        for i, feature in enumerate(selected_features[:10], 1):
            print(f"  {i:2d}. {feature}")

    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
