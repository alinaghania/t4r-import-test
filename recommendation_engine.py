"""
MODULE: MOTEUR DE RECOMMANDATION
===============================

Ce module gère l'inférence et les prédictions du modèle Transformer
pour fournir des recommandations de produits en temps réel.

Auteur: Assistant AI & Alina Ghani
Date: Juillet 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

from .transformer_model import TransformerRecommendationModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Moteur de recommandation basé sur le modèle Transformer.

    Cette classe gère:
    - Le chargement du modèle entraîné
    - L'inférence en temps réel
    - L'évaluation des performances
    - L'explication des prédictions
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        model: TransformerRecommendationModel = None,
    ):
        """
        Initialise le moteur de recommandation.

        Args:
            model_path: Chemin vers le modèle sauvegardé
            device: Device d'inférence ('auto', 'cpu', 'cuda')
            model: Modèle pré-chargé (optionnel)
        """
        # Configuration du device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Charger le modèle
        if model is not None:
            self.model = model
            self.model.to(self.device)
        elif model_path:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Vous devez fournir soit model_path soit model")

        # Mode évaluation
        self.model.eval()

        logger.info("🔮 Prédicteur initialisé:")
        logger.info(f"  • Device: {self.device}")
        logger.info(f"  • Items: {self.model.num_items}")
        logger.info(f"  • Séquence max: {self.model.seq_length}")

    def _load_model(self, model_path: str) -> TransformerRecommendationModel:
        """
        Charge le modèle depuis un fichier.

        Args:
            model_path: Chemin vers le modèle

        Returns:
            TransformerRecommendationModel: Modèle chargé
        """
        logger.info(f"📂 Chargement du modèle: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint["model_config"]

        # Créer le modèle avec la configuration sauvegardée
        model = TransformerRecommendationModel(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

        logger.info("✅ Modèle chargé avec succès!")

        return model

    def _preprocess_sequence(self, sequence: List[int]) -> torch.Tensor:
        """
        Préprocesse une séquence pour l'inférence.

        Args:
            sequence: Liste d'IDs de produits

        Returns:
            torch.Tensor: Séquence preprocessée [1, seq_length]
        """
        # Convertir en liste si nécessaire
        if isinstance(sequence, (np.ndarray, torch.Tensor)):
            sequence = sequence.tolist()

        # Filtrer les valeurs nulles ou négatives
        sequence = [item for item in sequence if item > 0]

        # Tronquer si trop long
        if len(sequence) > self.model.seq_length:
            sequence = sequence[-self.model.seq_length :]

        # Padding si trop court
        if len(sequence) < self.model.seq_length:
            # Padding au début (style GPT)
            padding_length = self.model.seq_length - len(sequence)
            sequence = [0] * padding_length + sequence

        # Convertir en tenseur
        tensor = torch.LongTensor(sequence).unsqueeze(0)  # [1, seq_length]
        tensor = tensor.to(self.device)

        return tensor

    def predict_single(
        self, sequence: List[int], top_k: int = 5, return_probabilities: bool = True
    ) -> Dict:
        """
        Fait une prédiction pour une séquence unique.

        Args:
            sequence: Historique des produits consultés
            top_k: Nombre de recommandations à retourner
            return_probabilities: Retourner les probabilités

        Returns:
            dict: Résultats de prédiction
        """
        self.model.eval()

        with torch.no_grad():
            # Préprocesser la séquence
            input_tensor = self._preprocess_sequence(sequence)

            # Forward pass
            logits = self.model(input_tensor)

            # Convertir en probabilités
            probabilities = torch.softmax(logits, dim=-1)

            # Top-K prédictions
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

            # Extraire les résultats
            predicted_items = top_indices.squeeze().cpu().numpy().tolist()
            predicted_probs = top_probs.squeeze().cpu().numpy().tolist()

            # Préparer les résultats
            results = {
                "predicted_items": predicted_items,
                "input_sequence": sequence,
                "preprocessed_sequence": input_tensor.squeeze().cpu().numpy().tolist(),
            }

            if return_probabilities:
                results["probabilities"] = predicted_probs
                results["confidence"] = max(predicted_probs)

            return results

    def predict_batch(
        self,
        sequences: List[List[int]],
        top_k: int = 5,
        return_probabilities: bool = False,
    ) -> List[Dict]:
        """
        Fait des prédictions pour un batch de séquences.

        Args:
            sequences: Liste de séquences à prédire
            top_k: Nombre de recommandations par séquence
            return_probabilities: Retourner les probabilités

        Returns:
            list: Liste des résultats de prédiction
        """
        self.model.eval()

        results = []

        with torch.no_grad():
            # Preprocesser toutes les séquences
            batch_tensors = []
            for seq in sequences:
                tensor = self._preprocess_sequence(seq)
                batch_tensors.append(tensor)

            # Concatener en batch
            batch_input = torch.cat(batch_tensors, dim=0)  # [batch_size, seq_length]

            # Forward pass
            logits = self.model(batch_input)

            # Convertir en probabilités
            probabilities = torch.softmax(logits, dim=-1)

            # Top-K pour chaque séquence
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

            # Extraire les résultats pour chaque séquence
            for i, original_seq in enumerate(sequences):
                predicted_items = top_indices[i].cpu().numpy().tolist()

                result = {
                    "predicted_items": predicted_items,
                    "input_sequence": original_seq,
                }

                if return_probabilities:
                    predicted_probs = top_probs[i].cpu().numpy().tolist()
                    result["probabilities"] = predicted_probs
                    result["confidence"] = max(predicted_probs)

                results.append(result)

        return results

    def explain_prediction(self, sequence: List[int], top_k: int = 5) -> Dict:
        """
        Fournit une explication détaillée d'une prédiction.

        Args:
            sequence: Séquence d'entrée
            top_k: Nombre de prédictions à expliquer

        Returns:
            dict: Explication détaillée
        """
        # Faire la prédiction de base
        prediction = self.predict_single(
            sequence, top_k=top_k, return_probabilities=True
        )

        # Analyse de la séquence
        unique_items = len(set([x for x in sequence if x > 0]))
        sequence_length = len([x for x in sequence if x > 0])

        # Calcul de la diversité des prédictions
        predicted_items = prediction["predicted_items"]
        diversity_score = len(set(predicted_items)) / len(predicted_items)

        # Explication
        explanation = {
            "prediction_summary": prediction,
            "sequence_analysis": {
                "original_length": sequence_length,
                "unique_items": unique_items,
                "diversity_ratio": unique_items / max(sequence_length, 1),
                "most_recent_items": sequence[-3:] if len(sequence) >= 3 else sequence,
            },
            "prediction_analysis": {
                "diversity_score": diversity_score,
                "confidence_level": prediction.get("confidence", 0),
                "top_prediction": predicted_items[0] if predicted_items else None,
            },
            "recommendations": [],
        }

        # Ajouter des recommandations explicites
        for i, (item, prob) in enumerate(
            zip(predicted_items, prediction.get("probabilities", []))
        ):
            explanation["recommendations"].append(
                {
                    "rank": i + 1,
                    "item_id": item,
                    "probability": prob,
                    "confidence_level": "High"
                    if prob > 0.3
                    else "Medium"
                    if prob > 0.1
                    else "Low",
                }
            )

        return explanation

    def evaluate_model(
        self,
        test_sequences: List[List[int]],
        test_targets: List[int],
        metrics: List[str] = ["accuracy@1", "accuracy@5", "mrr"],
    ) -> Dict:
        """
        Évalue les performances du modèle sur des données de test.

        Args:
            test_sequences: Séquences de test
            test_targets: Vrais prochains items
            metrics: Métriques à calculer

        Returns:
            dict: Résultats d'évaluation
        """
        logger.info(
            f"📊 Évaluation du modèle sur {len(test_sequences)} échantillons..."
        )

        self.model.eval()

        # Prédictions
        predictions = self.predict_batch(
            test_sequences, top_k=10, return_probabilities=False
        )

        # Calcul des métriques
        results = {}

        if "accuracy@1" in metrics:
            correct_at_1 = sum(
                1
                for pred, target in zip(predictions, test_targets)
                if pred["predicted_items"][0] == target
            )
            results["accuracy@1"] = correct_at_1 / len(test_sequences)

        if "accuracy@5" in metrics:
            correct_at_5 = sum(
                1
                for pred, target in zip(predictions, test_targets)
                if target in pred["predicted_items"][:5]
            )
            results["accuracy@5"] = correct_at_5 / len(test_sequences)

        if "mrr" in metrics:
            mrr_scores = []
            for pred, target in zip(predictions, test_targets):
                try:
                    rank = pred["predicted_items"].index(target) + 1
                    mrr_scores.append(1.0 / rank)
                except ValueError:
                    mrr_scores.append(0.0)
            results["mrr"] = np.mean(mrr_scores)

        logger.info("✅ Évaluation terminée:")
        for metric, value in results.items():
            logger.info(f"  • {metric}: {value:.4f}")

        return results

    def get_item_embeddings(self, item_ids: List[int]) -> np.ndarray:
        """
        Retourne les embeddings des items spécifiés.

        Args:
            item_ids: Liste d'IDs d'items

        Returns:
            np.ndarray: Embeddings [n_items, embedding_dim]
        """
        self.model.eval()

        with torch.no_grad():
            item_tensor = torch.LongTensor(item_ids).to(self.device)
            embeddings = self.model.get_embeddings(item_tensor)

            return embeddings.cpu().numpy()

    def find_similar_items(
        self, item_id: int, top_k: int = 10, similarity_metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """
        Trouve les items similaires à un item donné.

        Args:
            item_id: ID de l'item de référence
            top_k: Nombre d'items similaires à retourner
            similarity_metric: Métrique de similarité ('cosine', 'euclidean')

        Returns:
            list: Liste de tuples (item_id, similarity_score)
        """
        # Obtenir tous les embeddings
        all_item_ids = list(range(1, self.model.num_items + 1))
        all_embeddings = self.get_item_embeddings(all_item_ids)

        # Embedding de l'item de référence
        ref_embedding = self.get_item_embeddings([item_id])[0]

        # Calcul des similarités
        if similarity_metric == "cosine":
            # Similarité cosinus
            similarities = np.dot(all_embeddings, ref_embedding) / (
                np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(ref_embedding)
            )
        elif similarity_metric == "euclidean":
            # Distance euclidienne (convertie en similarité)
            distances = np.linalg.norm(all_embeddings - ref_embedding, axis=1)
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Métrique non supportée: {similarity_metric}")

        # Trier et retourner les top-K (en excluant l'item lui-même)
        sorted_indices = np.argsort(similarities)[::-1]

        similar_items = []
        for idx in sorted_indices:
            current_item_id = all_item_ids[idx]
            if current_item_id != item_id:  # Exclure l'item lui-même
                similar_items.append((current_item_id, similarities[idx]))
                if len(similar_items) >= top_k:
                    break

        return similar_items


def main():
    """Fonction principale pour test du module."""
    print("=" * 60)
    print("🔮 TEST DU MOTEUR DE RECOMMANDATION")
    print("=" * 60)

    # Créer un modèle de test
    model = TransformerRecommendationModel(
        num_items=50, embedding_dim=64, seq_length=10, num_heads=4, num_layers=2
    )

    # Créer le moteur de recommandation
    engine = RecommendationEngine(model=model, device="cpu")

    # Test de prédiction simple
    test_sequence = [1, 15, 23, 8, 42]
    print(f"\n📊 Test de prédiction:")
    print(f"  • Séquence: {test_sequence}")

    prediction = engine.predict_single(test_sequence, top_k=5)
    print(f"  • Prédictions: {prediction['predicted_items']}")
    print(f"  • Probabilités: {[f'{p:.3f}' for p in prediction['probabilities']]}")

    # Test de prédiction batch
    batch_sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    batch_predictions = engine.predict_batch(batch_sequences, top_k=3)

    print(f"\n📊 Test de prédiction batch:")
    for i, pred in enumerate(batch_predictions):
        print(
            f"  • Séquence {i + 1}: {pred['input_sequence']} → {pred['predicted_items']}"
        )

    # Test d'explication
    explanation = engine.explain_prediction(test_sequence, top_k=3)
    print(f"\n📋 Explication détaillée:")
    print(f"  • Items uniques: {explanation['sequence_analysis']['unique_items']}")
    print(
        f"  • Confiance: {explanation['prediction_analysis']['confidence_level']:.3f}"
    )

    # Test de similarité
    similar_items = engine.find_similar_items(item_id=1, top_k=5)
    print(f"\n🔍 Items similaires à l'item 1:")
    for item_id, similarity in similar_items:
        print(f"  • Item {item_id}: {similarity:.3f}")

    print("\n✅ Tous les tests du moteur de recommandation sont passés!")


if __name__ == "__main__":
    main()
