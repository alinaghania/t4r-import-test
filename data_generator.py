"""
MODULE: GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES
==========================================

Ce module génère des données de transactions synthétiques pour l'entraînement
du modèle de recommandation.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Générateur de données de transactions synthétiques.

    Cette classe crée des données réalistes d'interactions client-produit
    avec des sessions, produits, et montants de transactions.
    """

    def __init__(
        self, n_customers=1000, n_products=50, n_sessions=5000, random_seed=42
    ):
        """
        Initialise le générateur de données.

        Args:
            n_customers (int): Nombre de clients uniques
            n_products (int): Nombre de produits
            n_sessions (int): Nombre de sessions à générer
            random_seed (int): Graine pour la reproductibilité
        """
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_sessions = n_sessions
        self.random_seed = random_seed

        # Configuration reproductible
        np.random.seed(random_seed)

        logger.info(f"Générateur initialisé:")
        logger.info(f"  • Clients: {n_customers}")
        logger.info(f"  • Produits: {n_products}")
        logger.info(f"  • Sessions: {n_sessions}")

    def generate_transaction_data(self):
        """
        Génère les données de transactions synthétiques.

        Returns:
            pd.DataFrame: DataFrame avec les colonnes:
                - customer_id: ID du client
                - item_id: ID du produit
                - timestamp: Horodatage de l'interaction
                - session_id: ID de la session
                - amount: Montant de la transaction
        """
        logger.info("🔄 Génération des données de transactions en cours...")

        data = []

        for session_id in range(self.n_sessions):
            # Progression
            if session_id % 1000 == 0:
                logger.info(f"  Session {session_id}/{self.n_sessions}...")

            # Client aléatoire pour cette session
            customer_id = np.random.randint(1, self.n_customers + 1)

            # Longueur variable de session (2-9 interactions)
            session_length = np.random.randint(2, 10)

            # Timestamp de base (année 2024)
            base_timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=np.random.randint(0, 365)
            )

            # Générer les interactions de la session
            for i in range(session_length):
                # Produit aléatoire
                item_id = np.random.randint(1, self.n_products + 1)

                # Timestamp avec progression temporelle
                timestamp = base_timestamp + pd.Timedelta(minutes=i * 5)

                # Montant réaliste (distribution log-normale)
                amount = np.random.lognormal(mean=5, sigma=1)  # €10-€1000+
                amount = max(10, min(10000, amount))  # Borner entre 10€ et 10k€

                data.append(
                    {
                        "customer_id": customer_id,
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "session_id": session_id,
                        "amount": round(amount, 2),
                    }
                )

        # Créer le DataFrame
        df = pd.DataFrame(data)

        # Trier par session et timestamp
        df = df.sort_values(["session_id", "timestamp"])

        # Optimiser les types
        df["session_id"] = df["session_id"].astype(np.int64)
        df["customer_id"] = df["customer_id"].astype(np.int64)
        df["item_id"] = df["item_id"].astype(np.int64)

        logger.info("✅ Génération terminée !")
        logger.info(f"  • Total transactions: {len(df):,}")
        logger.info(f"  • Sessions uniques: {df['session_id'].nunique():,}")
        logger.info(f"  • Clients uniques: {df['customer_id'].nunique():,}")
        logger.info(f"  • Produits uniques: {df['item_id'].nunique()}")

        return df

    def save_data(self, df, output_path="data/raw_transactions.parquet"):
        """
        Sauvegarde les données générées.

        Args:
            df (pd.DataFrame): Données à sauvegarder
            output_path (str): Chemin de sauvegarde
        """
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sauvegarder en Parquet (format optimisé)
        df.to_parquet(output_path, index=False)

        logger.info(f"💾 Données sauvegardées: {output_path}")
        logger.info(f"  • Taille: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

        return output_path

    def generate_and_save(self, output_path="data/raw_transactions.parquet"):
        """
        Pipeline complet: génération + sauvegarde.

        Args:
            output_path (str): Chemin de sauvegarde

        Returns:
            tuple: (df, output_path)
        """
        # Générer les données
        df = self.generate_transaction_data()

        # Sauvegarder
        saved_path = self.save_data(df, output_path)

        return df, saved_path


def main():
    """Fonction principale pour test du module."""
    print("=" * 60)
    print("🔄 GÉNÉRATION DE DONNÉES SYNTHÉTIQUES")
    print("=" * 60)

    # Créer le générateur
    generator = SyntheticDataGenerator(
        n_customers=1000, n_products=50, n_sessions=5000, random_seed=42
    )

    # Générer et sauvegarder
    df, output_path = generator.generate_and_save()

    # Aperçu des données
    print("\n📊 Aperçu des données générées:")
    print(df.head(10))

    print("\n📈 Statistiques:")
    print(f"  • Période: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  • Montant moyen: {df['amount'].mean():.2f}€")
    print(f"  • Montant médian: {df['amount'].median():.2f}€")
    print(
        f"  • Session moyenne: {df.groupby('session_id').size().mean():.1f} interactions"
    )

    print(f"\n✅ Génération terminée avec succès!")
    print(f"📁 Fichier sauvegardé: {output_path}")


if __name__ == "__main__":
    main()
