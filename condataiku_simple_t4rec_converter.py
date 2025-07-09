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
        print(f"Dataset input 't4_rec_df' chargé: {df.shape}")
        return df
    except Exception as e:
        print(f"Erreur chargement input: {e}")
        raise


def detect_and_convert_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte et convertit les colonnes contenant des arrays JSON
    """
    print(f"🔍 Analyse de {len(df.columns)} colonnes...")
    df_converted = df.copy()

    converted_count = 0
    error_count = 0

    for col in df.columns:
        if df[col].dtype == "object":
            sample_values = df[col].dropna().head(3).tolist()

            # Vérifier si c'est des arrays JSON
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
                print(f"  Conversion colonne '{col}': {sample_values}")

                def convert_json_array(val):
                    try:
                        if pd.isna(val):
                            return np.nan
                        if isinstance(val, str):
                            array_val = json.loads(val)
                            if isinstance(array_val, list) and len(array_val) > 0:
                                return float(array_val[0])  # Prendre la première valeur
                        return float(val) if val is not None else np.nan
                    except:
                        return np.nan

                df_converted[col] = df[col].apply(convert_json_array)
                converted_count += 1

    print(f"📈 Résultats conversion:")
    print(f"  - Colonnes array détectées: {converted_count}")
    print(f"  - Converties avec succès: {converted_count}")
    print(f"  - Erreurs: {error_count}")

    return df_converted


def create_sequences_all_columns(
    df: pd.DataFrame,
    seq_length: int = 50,  # Augmenté pour accommoder plus de colonnes
    max_columns: int = None,  # Si None, prend toutes les colonnes
) -> Tuple[np.ndarray, Dict]:
    """
    Crée des séquences en utilisant TOUTES les colonnes numériques disponibles
    """
    print(f"Création de séquences avec TOUTES LES COLONNES...")

    # Sélectionner seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  {len(numeric_cols)} colonnes numériques trouvées")

    if len(numeric_cols) == 0:
        raise ValueError("Aucune colonne numérique trouvée!")

    # MODIFICATION PRINCIPALE : Prendre TOUTES les colonnes ou limite max
    if max_columns is not None and len(numeric_cols) > max_columns:
        selected_cols = numeric_cols[:max_columns]
        print(
            f"  Limite appliquée: {len(selected_cols)} colonnes sur {len(numeric_cols)}"
        )
    else:
        selected_cols = numeric_cols
        print(f"  TOUTES les colonnes sélectionnées: {len(selected_cols)}")

    # Ajuster la longueur de séquence selon le nombre de colonnes
    actual_seq_length = min(seq_length, len(selected_cols))
    print(f"  📏 Longueur séquence ajustée: {actual_seq_length}")

    sequences = []

    for idx, row in df.iterrows():
        # Récupérer les valeurs numériques de la ligne
        values = [row[col] for col in selected_cols]

        # Filtrer les valeurs NaN/infinies
        values = [v for v in values if not (pd.isna(v) or np.isinf(v))]

        if len(values) < actual_seq_length:
            # Répéter les valeurs si pas assez
            while len(values) < actual_seq_length:
                values.extend(
                    values[: min(len(values), actual_seq_length - len(values))]
                )
            values = values[:actual_seq_length]
        else:
            # Prendre les premières valeurs
            values = values[:actual_seq_length]

        sequences.append(values)

        if idx % 1000 == 0:
            print(f"  Traité {idx + 1} lignes...")

    sequences_array = np.array(sequences)
    print(f"✅ Séquences créées: {sequences_array.shape}")

    # Stats sur les séquences
    stats = {
        "num_sequences": len(sequences),
        "sequence_length": actual_seq_length,
        "selected_columns": selected_cols,
        "total_available_columns": len(numeric_cols),
        "columns_used": len(selected_cols),
        "value_range": (float(np.min(sequences_array)), float(np.max(sequences_array))),
        "mean_value": float(np.mean(sequences_array)),
    }

    return sequences_array, stats


def create_t4rec_format(sequences: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convertit les séquences au format T4Rec (input/target pairs)
    """
    print(f"🎯 Conversion au format T4Rec...")

    # Normaliser les valeurs entre 1 et 5000 (plus d'espace pour plus de colonnes)
    sequences_norm = sequences.copy()

    seq_min = np.min(sequences_norm)
    seq_max = np.max(sequences_norm)

    if seq_max != seq_min:
        sequences_norm = (sequences_norm - seq_min) / (seq_max - seq_min)
        sequences_norm = sequences_norm * 4999 + 1  # [1, 5000]
    else:
        sequences_norm = np.ones_like(sequences_norm)

    # Convertir en entiers (item_ids)
    sequences_ids = sequences_norm.astype(int)

    # Créer input/target pairs pour next-item prediction
    inputs = sequences_ids[:, :-1]  # Tout sauf le dernier
    targets = sequences_ids[:, 1:]  # Tout sauf le premier

    # Convertir en tenseurs PyTorch
    input_tensor = torch.LongTensor(inputs)
    target_tensor = torch.LongTensor(targets)

    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Target shape: {target_tensor.shape}")
    print(
        f" Item ID range: {int(np.min(sequences_ids))} - {int(np.max(sequences_ids))}"
    )

    return input_tensor, target_tensor


def create_output_dataframe(
    inputs: torch.Tensor, targets: torch.Tensor, original_df: pd.DataFrame, stats: Dict
) -> pd.DataFrame:
    """
    Crée le DataFrame de sortie pour Dataiku (VERSION CSV)
    """
    print(f" Création du DataFrame CSV pour Dataiku...")

    inputs_np = inputs.numpy()
    targets_np = targets.numpy()

    output_data = []

    for i in range(len(inputs_np)):
        row_data = {
            "sequence_id": i,
            "sequence_length": len(inputs_np[i]),
            "num_unique_items": stats.get("num_unique_items", 5000),
            "total_columns_available": stats.get("total_available_columns", 0),
            "columns_used": stats.get("columns_used", 0),
        }

        # Ajouter les inputs individuels comme colonnes
        for j, val in enumerate(inputs_np[i]):
            row_data[f"input_{j + 1}"] = int(val)

        # Ajouter les targets individuels comme colonnes
        for j, val in enumerate(targets_np[i]):
            row_data[f"target_{j + 1}"] = int(val)

        output_data.append(row_data)

    output_df = pd.DataFrame(output_data)

    print(f"DataFrame CSV créé: {output_df.shape}")
    print(
        f"  Colonnes: {list(output_df.columns[:7])}... (+ {len(output_df.columns) - 7} autres)"
    )

    return output_df


def save_pytorch_tensors(
    inputs: torch.Tensor, targets: torch.Tensor, stats: Dict
) -> str:
    """
    Sauvegarde les tenseurs PyTorch dans un fichier .pt prêt pour l'entraînement
    """
    print(f" Sauvegarde des tenseurs PyTorch...")

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
            "total_available_columns": stats.get("total_available_columns", 0),
            "columns_used": stats.get("columns_used", 0),
            "value_range": stats.get("value_range", (0, 5000)),
            "mean_value": stats.get("mean_value", 0.0),
        },
    }

    filename = "t4rec_training_data_ALL_COLUMNS.pt"
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
        print(f"Dataset CSV 't4_rec_df_clean' créé avec {len(df)} lignes")
    except Exception as e:
        print(f" Erreur sauvegarde output: {e}")
        raise


def main():
    """
    Pipeline principal du recipe Dataiku - VERSION TOUTES COLONNES
    """
    print("DÉMARRAGE RECIPE DATAIKU T4REC - TOUTES COLONNES")
    print("=" * 60)

    try:
        # 1. Charger les données d'input
        df = load_input_dataset()
        print(f" Dataset initial: {df.shape}")

        # 2. Convertir les arrays JSON
        df_converted = detect_and_convert_arrays(df)

        # 3. Créer des séquences avec TOUTES les colonnes
        # Options :
        # - max_columns=None : TOUTES les colonnes
        # - max_columns=100 : Limite à 100 colonnes
        sequences, stats = create_sequences_all_columns(
            df_converted,
            seq_length=50,  # Séquence plus longue
            max_columns=None,  # TOUTES les colonnes
        )
        print(f"📈 Stats séquences: {stats}")

        # 4. Convertir au format T4Rec
        inputs, targets = create_t4rec_format(sequences)

        # Ajouter les stats pour le DataFrame de sortie
        stats["num_unique_items"] = int(
            torch.max(torch.cat([inputs.flatten(), targets.flatten()])).item()
        )

        # 5. DOUBLE SAUVEGARDE
        output_df = create_output_dataframe(inputs, targets, df, stats)
        save_to_output_dataset(output_df)

        pytorch_file = save_pytorch_tensors(inputs, targets, stats)

        print("\n🎉 RECIPE TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print("📋 RÉSULTATS - VERSION TOUTES COLONNES:")
        print(f"  Colonnes totales disponibles: {stats['total_available_columns']}")
        print(f"  Colonnes utilisées: {stats['columns_used']}")
        print(f"  Longueur séquence: {stats['sequence_length']}")
        print(f"  Dataset CSV: 't4_rec_df_clean' ({len(output_df)} lignes)")
        print(f"  Tenseurs PyTorch: '{pytorch_file}'")

        print(f"\n RÉSUMÉ TECHNIQUE:")
        print(f"  - Format inputs: {inputs.dtype} {inputs.shape}")
        print(f"  - Format targets: {targets.dtype} {targets.shape}")
        print(f"  - Item ID range: 1 - {stats['num_unique_items']}")
        print("  Compatible avec modèle T4Rec étendu!")

    except Exception as e:
        print(f"\n ERREUR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

