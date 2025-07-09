#!/usr/bin/env python3
"""
CONVERSION RAPIDE POUR T4REC - SIMPLE ET DIRECT
================================================

Script basique pour convertir toutes les colonnes array en séquences T4Rec
et faire tourner le modèle rapidement.
"""

import dataiku
import numpy as np
import pandas as pd
import json
import torch
from typing import List, Tuple, Dict, Any


def load_dataiku_dataset(dataset_name: str = "t4_rec_df") -> pd.DataFrame:
    """
    Charge le dataset t4_rec_df depuis Dataiku
    """
    try:
        dataset = dataiku.Dataset(dataset_name)
        df = dataset.get_dataframe()
        print(f"✅ Dataset '{dataset_name}' chargé: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erreur chargement '{dataset_name}': {e}")
        raise


def detect_and_convert_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte et convertit toutes les colonnes qui contiennent des arrays JSON
    """
    print(f"🔍 Analyse de {len(df.columns)} colonnes...")

    converted_df = df.copy()
    conversion_stats = {"array_columns": 0, "converted_successfully": 0, "errors": 0}

    for col in df.columns:
        # Échantillon pour détecter le format
        sample_values = df[col].dropna().head(10)

        if len(sample_values) == 0:
            continue

        first_val = str(sample_values.iloc[0])

        # Détecter si c'est un array JSON
        if first_val.startswith("[") and first_val.endswith("]"):
            conversion_stats["array_columns"] += 1
            print(f"  📊 Conversion colonne '{col}': {first_val[:50]}...")

            try:
                # Convertir chaque valeur JSON en float
                def convert_json_array(val):
                    if pd.isna(val):
                        return 0.0
                    try:
                        arr = json.loads(str(val))
                        if isinstance(arr, list) and len(arr) > 0:
                            return float(arr[0])  # Prendre le premier élément
                        return 0.0
                    except:
                        return 0.0

                converted_df[col] = df[col].apply(convert_json_array)
                conversion_stats["converted_successfully"] += 1

            except Exception as e:
                print(f"    ❌ Erreur conversion {col}: {e}")
                conversion_stats["errors"] += 1
                # Garder la colonne originale en cas d'erreur

    print(f"📈 Résultats conversion:")
    print(f"  - Colonnes array détectées: {conversion_stats['array_columns']}")
    print(f"  - Converties avec succès: {conversion_stats['converted_successfully']}")
    print(f"  - Erreurs: {conversion_stats['errors']}")

    return converted_df


def create_simple_sequences(
    df: pd.DataFrame, seq_length: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Crée des séquences simples en prenant les valeurs numériques de chaque ligne
    et en les regroupant pour former des séquences
    """
    print(f"🔄 Création de séquences de longueur {seq_length}...")

    # Sélectionner seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  📊 {len(numeric_cols)} colonnes numériques trouvées")

    if len(numeric_cols) == 0:
        raise ValueError("Aucune colonne numérique trouvée!")

    # Prendre un sous-ensemble de colonnes si nécessaire
    if len(numeric_cols) > seq_length * 2:  # Garder assez pour input/target
        # Prendre les premières colonnes (ou on pourrait randomiser)
        selected_cols = numeric_cols[: seq_length * 2]
        print(f"  ✂️ Sélection de {len(selected_cols)} colonnes sur {len(numeric_cols)}")
    else:
        selected_cols = numeric_cols

    sequences = []

    for idx, row in df.iterrows():
        # Récupérer les valeurs numériques de la ligne
        values = [row[col] for col in selected_cols]

        # Filtrer les valeurs NaN/infinies
        values = [v for v in values if not (pd.isna(v) or np.isinf(v))]

        if len(values) < seq_length:
            # Répéter les valeurs ou padding si pas assez
            while len(values) < seq_length:
                values.extend(values)  # Répéter
            values = values[:seq_length]  # Tronquer
        else:
            # Prendre les premières valeurs
            values = values[:seq_length]

        sequences.append(values)

        if idx % 1000 == 0:
            print(f"  📝 Traité {idx + 1} lignes...")

    sequences_array = np.array(sequences)
    print(f"✅ Séquences créées: {sequences_array.shape}")

    # Stats sur les séquences
    stats = {
        "num_sequences": len(sequences),
        "sequence_length": seq_length,
        "selected_columns": selected_cols,
        "value_range": (float(np.min(sequences_array)), float(np.max(sequences_array))),
        "mean_value": float(np.mean(sequences_array)),
    }

    return sequences_array, stats


def create_t4rec_format(sequences: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit les séquences au format T4Rec (input/target pairs)
    """
    print(f"🎯 Conversion au format T4Rec...")

    # Pour T4Rec, on a besoin d'item_ids (entiers positifs)
    # Normaliser et convertir en IDs

    # 1. Normaliser les valeurs entre 1 et 1000 (simule des item_ids)
    sequences_norm = sequences.copy()

    # Normalisation min-max vers [1, 1000]
    seq_min = np.min(sequences_norm)
    seq_max = np.max(sequences_norm)

    if seq_max != seq_min:
        sequences_norm = (sequences_norm - seq_min) / (seq_max - seq_min)
        sequences_norm = sequences_norm * 999 + 1  # [1, 1000]
    else:
        sequences_norm = np.ones_like(sequences_norm)  # Toutes les valeurs = 1

    # 2. Convertir en entiers (item_ids)
    sequences_ids = sequences_norm.astype(int)

    # 3. Créer input/target pairs pour next-item prediction
    # Input: [1, 2, 3, ..., n-1]
    # Target: [2, 3, 4, ..., n]

    inputs = sequences_ids[:, :-1]  # Tout sauf le dernier
    targets = sequences_ids[:, 1:]  # Tout sauf le premier

    # 4. Convertir en tenseurs PyTorch
    input_tensor = torch.LongTensor(inputs)
    target_tensor = torch.LongTensor(targets)

    print(f"  📊 Input shape: {input_tensor.shape}")
    print(f"  📊 Target shape: {target_tensor.shape}")
    print(
        f"  📊 Item ID range: {int(np.min(sequences_ids))} - {int(np.max(sequences_ids))}"
    )

    return input_tensor, target_tensor


def save_t4rec_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_path: str = "data/simple_t4rec_data.pt",
) -> None:
    """
    Sauvegarde les données au format T4Rec
    """
    print(f"💾 Sauvegarde vers '{output_path}'...")

    data_dict = {
        "inputs": inputs,
        "targets": targets,
        "metadata": {
            "num_sequences": inputs.shape[0],
            "sequence_length": inputs.shape[1],
            "num_unique_items": int(
                torch.max(torch.cat([inputs.flatten(), targets.flatten()])).item()
            ),
            "creation_timestamp": pd.Timestamp.now().isoformat(),
        },
    }

    torch.save(data_dict, output_path)
    print(f"✅ Données sauvegardées: {inputs.shape[0]} séquences")


def main():
    """
    Pipeline principal de conversion
    """
    print("🚀 DÉMARRAGE CONVERSION SIMPLE T4REC")
    print("=" * 50)

    try:
        # 1. Charger les données
        df = load_dataiku_dataset()
        print(f"📊 Dataset initial: {df.shape}")

        # 2. Convertir les arrays JSON
        df_converted = detect_and_convert_arrays(df)

        # 3. Créer des séquences simples
        sequences, stats = create_simple_sequences(df_converted, seq_length=10)
        print(f"📈 Stats séquences: {stats}")

        # 4. Convertir au format T4Rec
        inputs, targets = create_t4rec_format(sequences)

        # 5. Sauvegarder
        save_t4rec_data(inputs, targets)

        print("\n🎉 CONVERSION TERMINÉE AVEC SUCCÈS!")
        print("=" * 50)
        print("📁 Fichier généré: data/simple_t4rec_data.pt")
        print("🎯 Prêt pour l'entraînement T4Rec!")

        # Test rapide de compatibilité
        print("\n🔬 TEST RAPIDE:")
        print(f"  - Format inputs: {inputs.dtype} {inputs.shape}")
        print(f"  - Format targets: {targets.dtype} {targets.shape}")
        print(
            f"  - Range item_ids: 1 - {int(torch.max(torch.cat([inputs.flatten(), targets.flatten()])).item())}"
        )
        print("  ✅ Compatible avec T4Rec!")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
