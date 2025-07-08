"""
PIPELINE PRINCIPAL: SYSTÈME DE RECOMMANDATION TRANSFORMER
=========================================================

Ce script orchestre le pipeline complet de recommandation de produits:
1. Génération de données synthétiques
2. Preprocessing des séquences
3. Entraînement du modèle Transformer
4. Démonstration d'inférence

Usage:
    python main.py [mode]

Modes disponibles:
    - full: Pipeline complet (défaut)
    - data: Génération de données seulement
    - train: Entraînement seulement
    - inference: Inférence seulement
    - demo: Démonstration interactive

"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Import des modules du système de recommandation
from src import (
    SyntheticDataGenerator,
    SequenceDataPreprocessor,
    TransformerRecommendationModel,
    ModelTrainer,
    RecommendationEngine,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """
    Pipeline principal pour le système de recommandation.

    Gère l'orchestration complète du système:
    - Génération/chargement des données
    - Preprocessing des séquences
    - Entraînement du modèle
    - Inférence et recommandations
    """

    def __init__(
        self,
        data_dir="data",
        models_dir="models",
        n_customers=1000,
        n_products=50,
        n_sessions=5000,
    ):
        """
        Initialise le pipeline.

        Args:
            data_dir: Répertoire des données
            models_dir: Répertoire des modèles
            n_customers: Nombre de clients pour la génération
            n_products: Nombre de produits
            n_sessions: Nombre de sessions
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # Configuration des données
        self.n_customers = n_customers
        self.n_products = n_products
        self.n_sessions = n_sessions

        # Chemins des fichiers
        self.raw_data_path = self.data_dir / "raw_transactions.parquet"
        self.training_data_path = self.data_dir / "training_data.pt"
        self.model_path = self.models_dir / "transformer_recommendation_model.pt"

        # Créer les répertoires
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        logger.info("🏗️ Pipeline initialisé:")
        logger.info(f"  • Répertoire données: {self.data_dir}")
        logger.info(f"  • Répertoire modèles: {self.models_dir}")
        logger.info(f"  • Configuration: {n_customers} clients, {n_products} produits")

    def run_data_generation(self):
        """Génère les données synthétiques."""
        logger.info("=" * 60)
        logger.info("📊 ÉTAPE 1: GÉNÉRATION DES DONNÉES")
        logger.info("=" * 60)

        # Créer le générateur
        generator = SyntheticDataGenerator(
            n_customers=self.n_customers,
            n_products=self.n_products,
            n_sessions=self.n_sessions,
            random_seed=42,
        )

        # Générer et sauvegarder
        df, _ = generator.generate_and_save(str(self.raw_data_path))

        logger.info(f"✅ Données générées: {len(df):,} transactions")

        return df

    def run_preprocessing(self):
        """Preprocesse les données en séquences."""
        logger.info("=" * 60)
        logger.info("⚙️ ÉTAPE 2: PREPROCESSING DES DONNÉES")
        logger.info("=" * 60)

        # Créer le preprocesseur
        preprocessor = SequenceDataPreprocessor(max_seq_length=10)

        # Pipeline complet de preprocessing
        results = preprocessor.full_preprocessing_pipeline(
            data_path=str(self.raw_data_path),
            output_training_path=str(self.training_data_path),
        )

        logger.info(
            f"✅ Preprocessing terminé: {results['stats']['num_sequences']:,} séquences"
        )

        return results

    def run_training(self):
        """Entraîne le modèle Transformer."""
        logger.info("=" * 60)
        logger.info("🚀 ÉTAPE 3: ENTRAÎNEMENT DU MODÈLE")
        logger.info("=" * 60)

        # Charger les données d'entraînement
        from src.model_trainer import load_training_data

        inputs, targets = load_training_data(str(self.training_data_path))

        # Créer le modèle
        model = TransformerRecommendationModel(
            num_items=self.n_products,
            embedding_dim=64,
            seq_length=10,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

        # Créer l'entraîneur
        trainer = ModelTrainer(
            model=model, device="auto", learning_rate=1e-3, batch_size=64
        )

        # Entraîner le modèle
        history = trainer.train(
            inputs=inputs,
            targets=targets,
            num_epochs=5,
            val_split=0.2,
            save_path=str(self.model_path),
            save_best=True,
            verbose=True,
        )

        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]

        logger.info("✅ Entraînement terminé:")
        logger.info(f"  • Loss finale train: {final_train_loss:.4f}")
        logger.info(f"  • Loss finale validation: {final_val_loss:.4f}")

        return history

    def run_inference_demo(self):
        """Démonstration d'inférence avec le modèle entraîné."""
        logger.info("=" * 60)
        logger.info("🔮 ÉTAPE 4: DÉMONSTRATION D'INFÉRENCE")
        logger.info("=" * 60)

        # Créer le moteur de recommandation
        engine = RecommendationEngine(model_path=str(self.model_path), device="auto")

        # Exemples de démonstration
        demo_cases = [
            {
                "name": "Client jeune actif",
                "description": "Jeune professionnel qui débute",
                "sequence": [1, 15, 23],
            },
            {
                "name": "Famille en croissance",
                "description": "Famille qui investit dans l'immobilier",
                "sequence": [5, 12, 34, 8],
            },
            {
                "name": "Investisseur expérimenté",
                "description": "Client fortuné avec portefeuille diversifié",
                "sequence": [2, 45, 18, 31, 7],
            },
            {
                "name": "Retraité prudent",
                "description": "Client senior sécurisant son patrimoine",
                "sequence": [8, 22, 41],
            },
        ]

        logger.info("🔍 Démonstration de prédictions:")

        for i, case in enumerate(demo_cases, 1):
            logger.info(f"\n📊 Exemple {i}: {case['name']}")
            logger.info(f"   Description: {case['description']}")
            logger.info(f"   Historique: {case['sequence']}")

            # Faire la prédiction
            prediction = engine.predict_single(
                sequence=case["sequence"], top_k=5, return_probabilities=True
            )

            # Afficher les résultats
            logger.info(f"   🎯 Prédictions Top-5:")
            for j, (item, prob) in enumerate(
                zip(prediction["predicted_items"], prediction["probabilities"]), 1
            ):
                logger.info(f"      {j}. Produit {item} | Probabilité: {prob:.1%}")

        # Explication détaillée pour le premier cas
        first_case = demo_cases[0]
        explanation = engine.explain_prediction(first_case["sequence"], top_k=5)

        logger.info(f"\n📋 Explication détaillée pour {first_case['name']}:")
        logger.info(
            f"   • Séquence préprocessée: {explanation['prediction_summary']['preprocessed_sequence']}"
        )
        logger.info(
            f"   • Items uniques: {explanation['sequence_analysis']['unique_items']}"
        )
        logger.info(
            f"   • Confiance modèle: {explanation['prediction_analysis']['confidence_level']:.3f}"
        )

        logger.info(f"\n✅ Démonstration terminée!")

        return engine

    def run_full_pipeline(self):
        """Exécute le pipeline complet."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("🎯 DÉMARRAGE DU PIPELINE COMPLET")
        logger.info("=" * 60)

        try:
            # 1. Génération des données
            self.run_data_generation()

            # 2. Preprocessing
            self.run_preprocessing()

            # 3. Entraînement
            self.run_training()

            # 4. Démonstration
            self.run_inference_demo()

            # Temps total
            total_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS! 🎉")
            logger.info("=" * 60)
            logger.info(f"⏱️  Temps total: {total_time:.1f}s")
            logger.info(f"📁 Modèle sauvegardé: {self.model_path}")
            logger.info("🚀 Le système de recommandation est prêt pour la production!")

        except Exception as e:
            logger.error(f"❌ Erreur dans le pipeline: {e}")
            raise


def main():
    """Fonction principale avec gestion des arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline de recommandation Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  python main.py                    # Pipeline complet
  python main.py --mode data        # Génération de données seulement
  python main.py --mode train       # Entraînement seulement
  python main.py --mode inference   # Inférence seulement
  python main.py --customers 2000   # 2000 clients au lieu de 1000
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "data", "preprocess", "train", "inference", "demo"],
        default="full",
        help="Mode d'exécution (défaut: full)",
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=1000,
        help="Nombre de clients pour la génération (défaut: 1000)",
    )

    parser.add_argument(
        "--products", type=int, default=50, help="Nombre de produits (défaut: 50)"
    )

    parser.add_argument(
        "--sessions", type=int, default=5000, help="Nombre de sessions (défaut: 5000)"
    )

    parser.add_argument(
        "--data-dir", default="data", help="Répertoire des données (défaut: data)"
    )

    parser.add_argument(
        "--models-dir", default="models", help="Répertoire des modèles (défaut: models)"
    )

    args = parser.parse_args()

    # Créer le pipeline
    pipeline = RecommendationPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        n_customers=args.customers,
        n_products=args.products,
        n_sessions=args.sessions,
    )

    # Exécuter selon le mode
    try:
        if args.mode == "full":
            pipeline.run_full_pipeline()
        elif args.mode == "data":
            pipeline.run_data_generation()
        elif args.mode == "preprocess":
            pipeline.run_preprocessing()
        elif args.mode == "train":
            pipeline.run_training()
        elif args.mode in ["inference", "demo"]:
            pipeline.run_inference_demo()

    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
