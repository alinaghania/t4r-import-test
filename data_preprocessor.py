"""
MODULE: PREPROCESSEUR DE DONNÉES
==============================

Ce module transforme les données de transactions brutes en séquences
prêtes pour l'entraînement du modèle Transformer.

Auteur: Assistant AI & Alina Ghani
Date: Juillet 2025
"""

import os
import pandas as pd
import numpy as np
import torch
import logging
from typing import Tuple, List

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDataPreprocessor:
    """
    Preprocesseur pour transformer les données de transactions en séquences.

    Cette classe gère:
    - La conversion des sessions en séquences
    - Le padding/troncature des séquences
    - La création des paires input/target pour l'entraînement
    """

    def __init__(self, max_seq_length=10):
        """
        Initialise le preprocesseur.

        Args:
            max_seq_length (int): Longueur maximale des séquences
        """
        self.max_seq_length = max_seq_length

        logger.info(f"Preprocesseur initialisé:")
        logger.info(f"  • Longueur max séquences: {max_seq_length}")

    def load_data(self, data_path):
        """
        Charge les données de transactions depuis un fichier.

        Args:
            data_path (str): Chemin vers le fichier de données

        Returns:
            pd.DataFrame: Données chargées
        """
        logger.info(f"📂 Chargement des données depuis {data_path}...")

        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {data_path}")

        logger.info(f"✅ Données chargées: {df.shape[0]:,} lignes")

        return df

    def create_sequences(self, df):
        """
        Convertit les sessions en séquences d'items.

        Args:
            df (pd.DataFrame): DataFrame avec colonnes session_id, item_id, timestamp

        Returns:
            np.ndarray: Array de séquences [n_sessions, max_seq_length]
        """
        logger.info("🔄 Conversion des sessions en séquences...")

        sequences = []
        session_stats = {"lengths": [], "total_sessions": 0}

        # Grouper par session et créer les séquences
        for session_id, session_data in df.groupby("session_id"):
            session_stats["total_sessions"] += 1

            # Trier par timestamp pour respecter l'ordre chronologique
            session_data = session_data.sort_values("timestamp")
            item_seq = session_data["item_id"].tolist()

            # Stocker la longueur originale
            session_stats["lengths"].append(len(item_seq))

            # Padding et troncature pour uniformiser les longueurs
            if len(item_seq) > self.max_seq_length:
                # Garder les derniers items (plus récents)
                item_seq = item_seq[-self.max_seq_length :]
            else:
                # Ajouter du padding (0) à la fin
                item_seq = item_seq + [0] * (self.max_seq_length - len(item_seq))

            sequences.append(item_seq)

            # Progression
            if len(sequences) % 1000 == 0:
                logger.info(f"  Séquences traitées: {len(sequences)}")

        sequences = np.array(sequences)

        # Statistiques
        lengths = np.array(session_stats["lengths"])
        logger.info("✅ Séquences créées !")
        logger.info(f"  • Nombre de séquences: {sequences.shape[0]:,}")
        logger.info(f"  • Forme finale: {sequences.shape}")
        logger.info(f"  • Longueur moyenne originale: {lengths.mean():.1f}")
        logger.info(f"  • Longueur médiane originale: {np.median(lengths):.1f}")
        logger.info(f"  • Sessions tronquées: {(lengths > self.max_seq_length).sum()}")
        logger.info(f"  • Sessions paddées: {(lengths < self.max_seq_length).sum()}")

        return sequences

    def create_training_pairs(self, sequences):
        """
        Crée les paires input/target pour l'entraînement next-item prediction.

        Args:
            sequences (np.ndarray): Séquences d'items [n_sessions, seq_length]

        Returns:
            tuple: (inputs, targets) pour l'entraînement
        """
        logger.info("🎯 Création des paires input/target...")

        # Convertir en tenseurs PyTorch
        sequences_tensor = torch.LongTensor(sequences)

        # Pour next-item prediction:
        # - Input: tous les items sauf le dernier
        # - Target: tous les items sauf le premier (décalage d'une position)
        inputs = sequences_tensor[:, :-1]  # [batch, seq_len-1]
        targets = sequences_tensor[:, 1:]  # [batch, seq_len-1]

        logger.info("✅ Paires créées !")
        logger.info(f"  • Inputs shape: {inputs.shape}")
        logger.info(f"  • Targets shape: {targets.shape}")

        # Exemples
        logger.info("📋 Exemples:")
        for i in range(min(3, len(inputs))):
            logger.info(f"  Séquence {i + 1}:")
            logger.info(f"    Input:  {inputs[i].numpy()}")
            logger.info(f"    Target: {targets[i].numpy()}")

        return inputs, targets

    def save_sequences(self, sequences, output_path="data/processed_sequences.pt"):
        """
        Sauvegarde les séquences preprocessées.

        Args:
            sequences (np.ndarray): Séquences à sauvegarder
            output_path (str): Chemin de sauvegarde
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sauvegarder en format PyTorch
        torch.save(
            {
                "sequences": sequences,
                "max_seq_length": self.max_seq_length,
                "shape": sequences.shape,
                "preprocessing_info": {
                    "num_sequences": len(sequences),
                    "seq_length": self.max_seq_length,
                    "unique_items": len(np.unique(sequences[sequences > 0])),
                },
            },
            output_path,
        )

        logger.info(f"💾 Séquences sauvegardées: {output_path}")

        return output_path

    def get_data_stats(self, sequences):
        """
        Calcule les statistiques des données preprocessées.

        Args:
            sequences (np.ndarray): Séquences preprocessées

        Returns:
            dict: Statistiques des données
        """
        # Items uniques (sans padding)
        non_zero_items = sequences[sequences > 0]
        unique_items = np.unique(non_zero_items)

        # Statistiques par séquence
        seq_lengths = []
        for seq in sequences:
            # Compter les items non-padding
            length = np.sum(seq > 0)
            seq_lengths.append(length)

        stats = {
            "num_sequences": len(sequences),
            "max_seq_length": self.max_seq_length,
            "unique_items": len(unique_items),
            "item_range": (unique_items.min(), unique_items.max()),
            "avg_seq_length": np.mean(seq_lengths),
            "median_seq_length": np.median(seq_lengths),
            "total_interactions": len(non_zero_items),
            "padding_ratio": np.sum(sequences == 0) / sequences.size,
        }

        return stats

    def full_preprocessing_pipeline(
        self,
        data_path,
        output_sequences_path="data/processed_sequences.pt",
        output_training_path="data/training_data.pt",
    ):
        """
        Pipeline complet de preprocessing.

        Args:
            data_path (str): Chemin vers les données brutes
            output_sequences_path (str): Chemin de sauvegarde des séquences
            output_training_path (str): Chemin de sauvegarde des données d'entraînement

        Returns:
            dict: Résultats du preprocessing
        """
        # 1. Charger les données
        df = self.load_data(data_path)

        # 2. Créer les séquences
        sequences = self.create_sequences(df)

        # 3. Créer les paires d'entraînement
        inputs, targets = self.create_training_pairs(sequences)

        # 4. Sauvegarder les séquences
        self.save_sequences(sequences, output_sequences_path)

        # 5. Sauvegarder les données d'entraînement
        os.makedirs(os.path.dirname(output_training_path), exist_ok=True)
        torch.save(
            {
                "inputs": inputs,
                "targets": targets,
                "sequences": sequences,
                "preprocessing_config": {"max_seq_length": self.max_seq_length},
            },
            output_training_path,
        )

        # 6. Calculer les statistiques
        stats = self.get_data_stats(sequences)

        logger.info(f"💾 Données d'entraînement sauvegardées: {output_training_path}")
        logger.info("📊 Statistiques finales:")
        for key, value in stats.items():
            logger.info(f"  • {key}: {value}")

        return {
            "sequences": sequences,
            "inputs": inputs,
            "targets": targets,
            "stats": stats,
            "paths": {
                "sequences": output_sequences_path,
                "training": output_training_path,
            },
        }


def main():
    """Fonction principale pour test du module."""
    print("=" * 60)
    print("⚙️ PREPROCESSING DES DONNÉES")
    print("=" * 60)

    # Créer le preprocesseur
    preprocessor = SequenceDataPreprocessor(max_seq_length=10)

    # Pipeline complet (en supposant que les données existent)
    try:
        results = preprocessor.full_preprocessing_pipeline(
            data_path="data/raw_transactions.parquet",
            output_sequences_path="data/processed_sequences.pt",
            output_training_path="data/training_data.pt",
        )

        print(f"\n✅ Preprocessing terminé avec succès!")
        print(f"📁 Séquences: {results['paths']['sequences']}")
        print(f"📁 Entraînement: {results['paths']['training']}")

    except FileNotFoundError:
        print("Fichier de données non trouvé.")
        print("Exécutez d'abord data_generator.py pour créer les données.")


if __name__ == "__main__":
    main()
