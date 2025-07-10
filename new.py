"""
MODÈLE TRANSFORMER POUR RECOMMANDATION
==============================================
 
Ce module définit l'architecture du modèle Transformer pour
la recommandation de produits.
"""
 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
from t4rec.data_preprocessor import SequenceDataPreprocessor
 
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
 
        logger.info(" Modèle Transformer initialisé:")
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
            logits = self(item_sequence)
 
            # Convertir en probabilités
            probs = torch.nn.functional.softmax(logits, dim=-1)
 
            # Obtenir les top-k prédictions
            top_probs, top_items = torch.topk(probs, k=top_k, dim=-1)
 
            return top_items, top_probs
 
    def get_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Retourne les embeddings pour une liste d'items.
 
        Args:
            item_ids (torch.Tensor): Tensor d'IDs d'items
 
        Returns:
            torch.Tensor: Embeddings des items [batch_size, embedding_dim]
        """
        return self.item_embedding(item_ids)
 
    def save_model(self, path: str, additional_info: Optional[Dict[str, Any]] = None):
        """
        Sauvegarde le modèle et ses métadonnées.
 
        Args:
            path (str): Chemin de sauvegarde
            additional_info (dict, optional): Informations additionnelles à sauvegarder
        """
        # Préparer les données à sauvegarder
        save_data = {
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
 
        # Ajouter les informations additionnelles si fournies
        if additional_info:
            save_data["additional_info"] = additional_info
 
        # Sauvegarder
        torch.save(save_data, path)
        logger.info(f" Modèle sauvegardé: {path}")
 
    @classmethod
    def load_model(
        cls, path: str, device: str = "cpu"
    ) -> Tuple["TransformerRecommendationModel", Dict[str, Any]]:
        """
        Charge un modèle sauvegardé.
 
        Args:
            path (str): Chemin vers le modèle sauvegardé
            device (str): Device sur lequel charger le modèle
 
        Returns:
            tuple: (modèle chargé, données additionnelles)
        """
        # Charger les données
        checkpoint = torch.load(path, map_location=device)
 
        # Extraire la configuration
        config = checkpoint["model_config"]
 
        # Créer une nouvelle instance
        model = cls(**config)
 
        # Charger les poids
        model.load_state_dict(checkpoint["model_state_dict"])
 
        # Déplacer sur le device
        model = model.to(device)
 
        logger.info(f" Modèle chargé depuis: {path}")
 
        return model, checkpoint
 
 
class T4RecVectorPipeline:
    """
    Pipeline complet pour l'entraînement et l'inférence du modèle T4Rec
    avec des vecteurs en entrée.
    """
 
    def __init__(
        self,
        input_vector_dim: int,
        target_embedding_dim: int = 64,
        sequence_length: int = 10,
        num_heads: int = 4,
        num_layers: int = 2,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
    ):
        """
        Initialise le pipeline.
 
        Args:
            input_vector_dim (int): Dimension des vecteurs d'entrée
            target_embedding_dim (int): Dimension cible des embeddings
            sequence_length (int): Longueur des séquences
            num_heads (int): Nombre de têtes d'attention
            num_layers (int): Nombre de couches Transformer
            batch_size (int): Taille des batchs
            learning_rate (float): Taux d'apprentissage
            num_epochs (int): Nombre d'époques d'entraînement
        """
        self.input_vector_dim = input_vector_dim
        self.target_embedding_dim = target_embedding_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
 
        # Initialiser le préprocesseur
        self.preprocessor = SequenceDataPreprocessor(
            max_seq_length=sequence_length, target_vector_dim=input_vector_dim
        )
 
        # Initialiser le modèle
        self.model = TransformerRecommendationModel(
            num_items=input_vector_dim,  # Chaque dimension est traitée comme un "item"
            embedding_dim=target_embedding_dim,
            seq_length=sequence_length,
            num_heads=num_heads,
            num_layers=num_layers,
        )
 
        # Optimizer et critère de perte
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Pour la régression sur les vecteurs
 
        logger.info(" Pipeline initialisé avec succès!")
        self._log_config()
 
    def _log_config(self):
        """Affiche la configuration du pipeline."""
        logger.info(" Configuration du pipeline:")
        logger.info(f"  • Dimension vecteurs: {self.input_vector_dim}")
        logger.info(f"  • Dimension embeddings: {self.target_embedding_dim}")
        logger.info(f"  • Longueur séquences: {self.sequence_length}")
        logger.info(f"  • Têtes d'attention: {self.num_heads}")
        logger.info(f"  • Couches Transformer: {self.num_layers}")
        logger.info(f"  • Taille batch: {self.batch_size}")
        logger.info(f"  • Learning rate: {self.learning_rate}")
        logger.info(f"  • Époques: {self.num_epochs}")
 
    def run_pipeline(self, data_df: pd.DataFrame, output_path: str) -> Dict[str, Any]:
        """
        Exécute le pipeline complet: prétraitement, entraînement et sauvegarde.
 
        Args:
            data_df (pd.DataFrame): DataFrame avec les colonnes vectorielles
            output_path (str): Chemin pour sauvegarder le modèle
 
        Returns:
            dict: Métriques et résultats du pipeline
        """
        try:
            # 1. Prétraitement
            logger.info(" Prétraitement des données...")
            sequences = self.preprocessor.create_sequences(data_df)
            inputs, targets = self.preprocessor.create_training_pairs(sequences)
 
            # 2. Entraînement
            logger.info(" Début de l'entraînement...")
            train_metrics = self._train_model(inputs, targets)
 
            # 3. Sauvegarde
            logger.info("Sauvegarde du modèle...")
            self.model.save_model(
                output_path,
                additional_info={
                    "input_vector_dim": self.input_vector_dim,
                    "target_embedding_dim": self.target_embedding_dim,
                    "sequence_length": self.sequence_length,
                    "pipeline_config": self.__dict__,
                },
            )
 
            return {
                "status": "success",
                "metrics": train_metrics,
                "model_path": output_path,
            }
 
        except Exception as e:
            logger.error(f" Erreur dans le pipeline: {str(e)}")
            raise
 
    def _train_model(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle sur les données prétraitées.
 
        Args:
            inputs (torch.Tensor): Tenseur des séquences d'entrée
            targets (torch.Tensor): Tenseur des cibles
 
        Returns:
            dict: Métriques d'entraînement
        """
        self.model.train()
        metrics = {"epoch_losses": []}
 
        n_batches = len(inputs) // self.batch_size
 
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
 
            for batch_idx in range(n_batches):
                # Préparer le batch
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
 
                batch_inputs = inputs[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]
 
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
 
                # Backward pass
                loss.backward()
                self.optimizer.step()
 
                epoch_loss += loss.item()
 
            # Métriques de l'époque
            avg_epoch_loss = epoch_loss / n_batches
            metrics["epoch_losses"].append(avg_epoch_loss)
 
            logger.info(
                f"Époque {epoch + 1}/{self.num_epochs} - Loss: {avg_epoch_loss:.4f}"
            )
 
        return metrics
 
 
