import dataiku
import numpy as np
import pandas as pd
import json
import torch
import os
from typing import List, Tuple, Dict, Any


def load_input_dataset() -> pd.DataFrame:
    """
    Charge le dataset d'input depuis Dataiku
    """
    try:
        dataset = dataiku.Dataset("t4_rec_df")
        df = dataset.get_dataframe()
        print(f"✅ Dataset input 't4_rec_df' chargé: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Erreur chargement input: {e}")
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


def create_output_dataframe(
    inputs: torch.Tensor, targets: torch.Tensor, original_df: pd.DataFrame, stats: Dict
) -> pd.DataFrame:
    """
    Crée le DataFrame de sortie pour Dataiku (VERSION CSV)
    """
    print(f"📋 Création du DataFrame CSV pour Dataiku...")

    # Convertir les tenseurs en numpy pour le DataFrame
    inputs_np = inputs.numpy()
    targets_np = targets.numpy()

    # Créer le DataFrame de sortie
    output_data = []

    for i in range(len(inputs_np)):
        # Créer une ligne par séquence
        row_data = {
            "sequence_id": i,
            "sequence_length": len(inputs_np[i]),
            "num_unique_items": stats.get("num_unique_items", 1000),
        }

        # Ajouter les inputs individuels comme colonnes
        for j, val in enumerate(inputs_np[i]):
            row_data[f"input_{j + 1}"] = int(val)

        # Ajouter les targets individuels comme colonnes
        for j, val in enumerate(targets_np[i]):
            row_data[f"target_{j + 1}"] = int(val)

        output_data.append(row_data)

    output_df = pd.DataFrame(output_data)

    print(f"✅ DataFrame CSV créé: {output_df.shape}")
    print(
        f"  📊 Colonnes: {list(output_df.columns[:5])}... (+ {len(output_df.columns) - 5} autres)"
    )

    return output_df


def save_pytorch_tensors(
    inputs: torch.Tensor, targets: torch.Tensor, stats: Dict
) -> str:
    """
    Sauvegarde les tenseurs PyTorch dans un fichier .pt prêt pour l'entraînement
    """
    print(f"💾 Sauvegarde des tenseurs PyTorch...")

    # Créer le dictionnaire avec toutes les données nécessaires pour l'entraînement
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
            "selected_columns": stats.get("selected_columns", []),
            "value_range": stats.get("value_range", (0, 1000)),
            "mean_value": stats.get("mean_value", 0.0),
        },
    }

    # Nom du fichier simple
    filename = "t4rec_training_data.pt"

    # Sauvegarde
    torch.save(data_dict, filename)
    print(f"✅ Tenseurs PyTorch sauvegardés: {filename}")
    print(f"  📦 Contenu: inputs {inputs.shape}, targets {targets.shape}")

    return filename


def save_to_output_dataset(df: pd.DataFrame) -> None:
    """
    Sauvegarde le DataFrame vers le dataset de sortie Dataiku
    """
    try:
        output_dataset = dataiku.Dataset("t4_rec_df_clean")
        output_dataset.write_with_schema(df)
        print(f"✅ Dataset CSV 't4_rec_df_clean' créé avec {len(df)} lignes")
    except Exception as e:
        print(f"❌ Erreur sauvegarde output: {e}")
        raise


def create_usage_guide(pytorch_file: str, stats: Dict) -> None:
    """
    Crée un guide d'utilisation pour charger les données dans le modèle
    """
    guide = f"""
# GUIDE D'UTILISATION T4REC
========================

## Fichiers générés:
1. Dataset Dataiku: 't4_rec_df_clean' (visualisation/debug)
2. Fichier PyTorch: '{pytorch_file}' (entraînement direct)

## Charger les données pour l'entraînement:
```python
import torch

# Charger les données T4Rec
data = torch.load('{pytorch_file}')
inputs = data['inputs']      # torch.Size([{stats["num_sequences"]}, {stats["sequence_length"] - 1}])
targets = data['targets']    # torch.Size([{stats["num_sequences"]}, {stats["sequence_length"] - 1}])
metadata = data['metadata']

# Utiliser avec ton modèle T4Rec
from src.transformer_model import TransformerRecommendationModel
from src.model_trainer import ModelTrainer

model = TransformerRecommendationModel(
    num_items=metadata['num_unique_items'],  # {stats.get("num_unique_items", 1000)}
    embedding_dim=64,
    seq_length={stats["sequence_length"]}
)

trainer = ModelTrainer(model)
trainer.train(inputs, targets, num_epochs=5)
```

## Données disponibles:
- Séquences d'origine: {stats["num_sequences"]}
- Longueur de séquence: {stats["sequence_length"]}
- Range item_ids: 1 - {stats.get("num_unique_items", 1000)}
- Colonnes utilisées: {len(stats.get("selected_columns", []))}
"""

    with open("t4rec_usage_guide.txt", "w", encoding="utf-8") as f:
        f.write(guide)

    print(f"📚 Guide d'utilisation créé: t4rec_usage_guide.txt")


def main():
    """
    Pipeline principal du recipe Dataiku
    """
    print("🚀 DÉMARRAGE RECIPE DATAIKU T4REC")
    print("=" * 50)

    try:
        # 1. Charger les données d'input
        df = load_input_dataset()
        print(f"📊 Dataset initial: {df.shape}")

        # 2. Convertir les arrays JSON
        df_converted = detect_and_convert_arrays(df)

        # 3. Créer des séquences simples
        sequences, stats = create_simple_sequences(df_converted, seq_length=10)
        print(f"📈 Stats séquences: {stats}")

        # 4. Convertir au format T4Rec
        inputs, targets = create_t4rec_format(sequences)

        # Ajouter les stats pour le DataFrame de sortie
        stats["num_unique_items"] = int(
            torch.max(torch.cat([inputs.flatten(), targets.flatten()])).item()
        )

        # 5. DOUBLE SAUVEGARDE

        # 5a. Créer le DataFrame CSV pour Dataiku
        output_df = create_output_dataframe(inputs, targets, df, stats)
        save_to_output_dataset(output_df)

        # 5b. Sauvegarder les tenseurs PyTorch pour l'entraînement
        pytorch_file = save_pytorch_tensors(inputs, targets, stats)

        # 6. Créer le guide d'utilisation
        create_usage_guide(pytorch_file, stats)

        print("\n🎉 RECIPE TERMINÉ AVEC SUCCÈS!")
        print("=" * 50)
        print("📋 DOUBLE SORTIE CRÉÉE:")
        print(f"  1. Dataset CSV Dataiku: 't4_rec_df_clean' ({len(output_df)} lignes)")
        print(f"  2. Tenseurs PyTorch: '{pytorch_file}'")
        print("  3. Guide d'usage: 't4rec_usage_guide.txt'")
        print("\n🎯 PRÊT POUR L'ENTRAÎNEMENT!")

        # Test rapide de compatibilité
        print("\n🔬 RÉSUMÉ TECHNIQUE:")
        print(f"  - Séquences d'origine: {len(df)} → {len(output_df)} séquences T4Rec")
        print(f"  - Format inputs: {inputs.dtype} {inputs.shape}")
        print(f"  - Format targets: {targets.dtype} {targets.shape}")
        print(f"  - Item ID range: 1 - {stats['num_unique_items']}")
        print("  ✅ Compatible avec ton modèle T4Rec!")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
