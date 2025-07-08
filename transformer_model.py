"""
MODULE: MODÈLE TRANSFORMER POUR RECOMMANDATION
==============================================

Ce module définit l'architecture du modèle Transformer pour
la recommandation de produits.

"""

import torch
import torch.nn as nn
import logging
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerRecommendationModel(nn.Module):
    """
    Modèle Transformer pour la recommandation de produits.

    Architecture:
    1. Couche d'embedding pour les produits
    2. Transformer Encoder pour capturer les dépendances séquentielles
    3. Couche de classification pour prédire le prochain produit

    Le modèle utilise l'approche "next-item prediction" où l'objectif
    est de prédire le prochain produit dans une séquence donnée.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        seq_length: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None,
    ):
        """
        Initialise le modèle Transformer.

        Args:
            num_items (int): Nombre total de produits
            embedding_dim (int): Dimension des embeddings
            seq_length (int): Longueur maximale des séquences
            num_heads (int): Nombre de têtes d'attention
            num_layers (int): Nombre de couches d'encodeur
            dropout (float): Taux de dropout pour la régularisation
            feedforward_dim (int, optional): Dimension du feedforward (défaut: 2 * embedding_dim)
        """
        super().__init__()

        # Configuration du modèle
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.feedforward_dim = feedforward_dim or (2 * embedding_dim)

        # Validation des paramètres
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) doit être divisible par num_heads ({num_heads})"
            )

        # Couche d'embedding pour les produits
        # padding_idx=0 pour que le token de padding ait un embedding fixe à zéro
        self.item_embedding = nn.Embedding(
            num_items + 1,  # +1 pour le token de padding
            embedding_dim,
            padding_idx=0,
        )

        # Architecture Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,  # Dimension du modèle
            nhead=num_heads,  # Nombre de têtes d'attention
            dim_feedforward=self.feedforward_dim,  # Dimension du feedforward
            dropout=dropout,  # Dropout pour la régularisation
            activation="relu",  # Fonction d'activation
            batch_first=True,  # Format [batch, seq, features]
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Couche de normalisation finale
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Couche de sortie pour prédire le prochain produit
        self.output_layer = nn.Linear(embedding_dim, num_items + 1)

        # Initialisation des poids
        self._init_weights()

        # Afficher la configuration
        self._log_model_info()

    def _init_weights(self):
        """Initialise les poids du modèle avec des valeurs appropriées."""
        # Initialisation Xavier pour les embeddings
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # S'assurer que le padding embedding reste à zéro
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)

        # Initialisation Xavier pour la couche de sortie
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _log_model_info(self):
        """Affiche les informations sur le modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("🤖 Modèle Transformer initialisé:")
        logger.info(f"  • Produits: {self.num_items}")
        logger.info(f"  • Embedding dimension: {self.embedding_dim}")
        logger.info(f"  • Longueur séquences: {self.seq_length}")
        logger.info(f"  • Têtes d'attention: {self.num_heads}")
        logger.info(f"  • Couches encodeur: {self.num_layers}")
        logger.info(f"  • Dropout: {self.dropout}")
        logger.info(f"  • Paramètres totaux: {total_params:,}")
        logger.info(f"  • Paramètres entraînables: {trainable_params:,}")

    def forward(
        self, item_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass du modèle.

        Args:
            item_sequence (torch.Tensor): Séquence d'items [batch_size, seq_length]
            mask (torch.Tensor, optional): Masque d'attention pour ignorer le padding

        Returns:
            torch.Tensor: Logits pour la prédiction du prochain item [batch_size, num_items+1]
        """
        batch_size, seq_len = item_sequence.shape

        # Étape 1: Convertir les IDs en embeddings
        embeddings = self.item_embedding(item_sequence)  # [batch, seq_len, embed_dim]

        # Étape 2: Créer un masque pour ignorer les tokens de padding si non fourni
        if mask is None:
            # Créer un masque booléen: True pour les positions à ignorer (padding)
            mask = item_sequence == 0  # [batch, seq_len]

        # Étape 3: Passer par le Transformer
        # Note: PyTorch Transformer attend src_key_padding_mask
        sequence_output = self.transformer(
            embeddings, src_key_padding_mask=mask
        )  # [batch, seq_len, embed_dim]

        # Étape 4: Normalisation
        sequence_output = self.layer_norm(sequence_output)

        # Étape 5: Utiliser la représentation du dernier token non-padding
        # Pour chaque séquence, trouver le dernier token non-padding
        if mask is not None:
            # Trouver les indices des derniers tokens valides
            valid_lengths = (~mask).sum(dim=1) - 1  # -1 car indexation base 0
            valid_lengths = torch.clamp(valid_lengths, min=0)

            # Extraire les représentations des derniers tokens valides
            batch_indices = torch.arange(batch_size, device=item_sequence.device)
            last_outputs = sequence_output[
                batch_indices, valid_lengths
            ]  # [batch, embed_dim]
        else:
            # Si pas de masque, utiliser simplement le dernier token
            last_outputs = sequence_output[:, -1, :]  # [batch, embed_dim]

        # Étape 6: Prédire le prochain item
        logits = self.output_layer(last_outputs)  # [batch, num_items+1]

        return logits

    def predict_next_items(self, item_sequence: torch.Tensor, top_k: int = 5) -> tuple:
        """
        Prédit les top-K prochains items pour une séquence.

        Args:
            item_sequence (torch.Tensor): Séquence d'items
            top_k (int): Nombre de prédictions à retourner

        Returns:
            tuple: (predicted_items, probabilities)
        """
        self.eval()  # Mode évaluation

        with torch.no_grad():
            # Forward pass
            logits = self.forward(item_sequence)

            # Convertir en probabilités
            probabilities = torch.softmax(logits, dim=-1)

            # Top-K prédictions
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

            return top_indices, top_probs

    def get_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Retourne les embeddings pour des items donnés.

        Args:
            item_ids (torch.Tensor): IDs des items

        Returns:
            torch.Tensor: Embeddings correspondants
        """
        return self.item_embedding(item_ids)

    def save_model(self, path: str, additional_info: dict = None):
        """
        Sauvegarde le modèle avec sa configuration.

        Args:
            path (str): Chemin de sauvegarde
            additional_info (dict): Informations supplémentaires à sauvegarder
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "num_items": self.num_items,
                "embedding_dim": self.embedding_dim,
                "seq_length": self.seq_length,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "feedforward_dim": self.feedforward_dim,
            },
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, path)
        logger.info(f"💾 Modèle sauvegardé: {path}")

    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        """
        Charge un modèle depuis un fichier.

        Args:
            path (str): Chemin du modèle sauvegardé
            device (str): Device sur lequel charger le modèle

        Returns:
            TransformerRecommendationModel: Modèle chargé
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["model_config"]

        # Créer le modèle avec la configuration sauvegardée
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(f"✅ Modèle chargé depuis: {path}")

        return model, checkpoint


def main():
    """Fonction principale pour test du module."""
    print("=" * 60)
    print("🤖 TEST DU MODÈLE TRANSFORMER")
    print("=" * 60)

    # Configuration du test
    num_items = 50
    batch_size = 8
    seq_length = 10

    # Créer le modèle
    model = TransformerRecommendationModel(
        num_items=num_items,
        embedding_dim=64,
        seq_length=seq_length,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )

    # Créer des données de test
    test_sequences = torch.randint(1, num_items + 1, (batch_size, seq_length))
    # Ajouter quelques tokens de padding
    test_sequences[:, -2:] = 0  # Padding sur les 2 dernières positions

    print(f"\n📊 Test avec {batch_size} séquences:")
    print(f"  • Forme input: {test_sequences.shape}")
    print(f"  • Exemple séquence: {test_sequences[0].tolist()}")

    # Forward pass
    with torch.no_grad():
        logits = model(test_sequences)
        print(f"  • Forme output: {logits.shape}")

        # Test de prédiction
        top_items, top_probs = model.predict_next_items(test_sequences[:1], top_k=5)
        print(f"  • Top 5 prédictions: {top_items[0].tolist()}")
        print(f"  • Probabilités: {top_probs[0].tolist()}")

    # Test de sauvegarde/chargement
    test_path = "test_model.pt"
    model.save_model(test_path, {"test_info": "Model test successful"})

    loaded_model, checkpoint = TransformerRecommendationModel.load_model(test_path)
    print(f"\n✅ Test de sauvegarde/chargement réussi!")

    # Nettoyer
    import os

    os.remove(test_path)

    print("🎉 Tous les tests sont passés avec succès!")


if __name__ == "__main__":
    main()
