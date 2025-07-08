"""
MODULE: ENTRAÎNEUR DE MODÈLE
===========================

Ce module gère l'entraînement du modèle Transformer pour
la recommandation de produits.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional

from .transformer_model import TransformerRecommendationModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Entraîneur pour le modèle Transformer de recommandation.

    Cette classe gère:
    - L'entraînement avec validation
    - Le monitoring des métriques
    - La sauvegarde automatique
    - L'early stopping
    """

    def __init__(
        self,
        model: TransformerRecommendationModel,
        device: str = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
    ):
        """
        Initialise l'entraîneur.

        Args:
            model: Modèle à entraîner
            device: Device d'entraînement ('auto', 'cpu', 'cuda')
            learning_rate: Taux d'apprentissage
            weight_decay: Décroissance des poids
            batch_size: Taille des batches
        """
        self.model = model
        self.batch_size = batch_size

        # Configuration du device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Configuration de l'optimiseur
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Configuration de la loss fonction
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer le padding

        # Historique d'entraînement
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "epochs": [],
        }

        logger.info("🚀 Entraîneur initialisé:")
        logger.info(f"  • Device: {self.device}")
        logger.info(f"  • Learning rate: {learning_rate}")
        logger.info(f"  • Batch size: {batch_size}")
        logger.info(
            f"  • Paramètres modèle: {sum(p.numel() for p in model.parameters()):,}"
        )

    def create_dataloaders(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        val_split: float = 0.2,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Crée les DataLoaders pour l'entraînement et la validation.

        Args:
            inputs: Tenseur d'entrée
            targets: Tenseur de sortie
            val_split: Proportion des données pour la validation
            shuffle: Mélanger les données

        Returns:
            tuple: (train_loader, val_loader)
        """
        # Déplacer les données sur le bon device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Division train/val
        n_samples = len(inputs)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        if shuffle:
            # Mélanger les indices
            indices = torch.randperm(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
        else:
            train_indices = torch.arange(n_train)
            val_indices = torch.arange(n_train, n_samples)

        # Créer les datasets
        train_dataset = TensorDataset(inputs[train_indices], targets[train_indices])
        val_dataset = TensorDataset(inputs[val_indices], targets[val_indices])

        # Créer les DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        logger.info(f"📊 DataLoaders créés:")
        logger.info(
            f"  • Train: {len(train_dataset):,} échantillons, {len(train_loader)} batches"
        )
        logger.info(
            f"  • Validation: {len(val_dataset):,} échantillons, {len(val_loader)} batches"
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Entraîne le modèle pour une époque.

        Args:
            train_loader: DataLoader d'entraînement

        Returns:
            float: Loss moyenne de l'époque
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()

            # Training avec prédiction séquentielle
            batch_size_actual, seq_len = batch_inputs.shape
            all_losses = []

            # Prédire à chaque position de la séquence
            for pos in range(seq_len):
                if pos == 0:
                    continue  # Pas de prédiction sans contexte

                # Utiliser la séquence jusqu'à la position pos
                input_seq = batch_inputs[:, : pos + 1]

                # Padding si nécessaire
                if input_seq.size(1) < self.model.seq_length:
                    padding = torch.zeros(
                        batch_size_actual,
                        self.model.seq_length - input_seq.size(1),
                        device=self.device,
                        dtype=torch.long,
                    )
                    input_seq = torch.cat([padding, input_seq], dim=1)

                # Forward pass
                logits = self.model(input_seq)
                target_items = batch_targets[:, pos - 1]

                # Loss seulement pour les tokens non-padding
                mask = target_items != 0
                if mask.sum() > 0:
                    loss = self.criterion(logits[mask], target_items[mask])
                    all_losses.append(loss)

            # Backpropagation
            if all_losses:
                batch_loss = torch.stack(all_losses).mean()
                batch_loss.backward()

                # Gradient clipping pour éviter l'explosion des gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Valide le modèle pour une époque.

        Args:
            val_loader: DataLoader de validation

        Returns:
            float: Loss moyenne de validation
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Validation avec la même logique que l'entraînement
                batch_size_actual, seq_len = batch_inputs.shape
                all_losses = []

                for pos in range(seq_len):
                    if pos == 0:
                        continue

                    input_seq = batch_inputs[:, : pos + 1]

                    if input_seq.size(1) < self.model.seq_length:
                        padding = torch.zeros(
                            batch_size_actual,
                            self.model.seq_length - input_seq.size(1),
                            device=self.device,
                            dtype=torch.long,
                        )
                        input_seq = torch.cat([padding, input_seq], dim=1)

                    logits = self.model(input_seq)
                    target_items = batch_targets[:, pos - 1]

                    mask = target_items != 0
                    if mask.sum() > 0:
                        loss = self.criterion(logits[mask], target_items[mask])
                        all_losses.append(loss)

                if all_losses:
                    batch_loss = torch.stack(all_losses).mean()
                    total_loss += batch_loss.item()
                    num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def train(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_epochs: int = 10,
        val_split: float = 0.2,
        save_path: Optional[str] = None,
        save_best: bool = True,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Entraîne le modèle.

        Args:
            inputs: Données d'entrée
            targets: Données cibles
            num_epochs: Nombre d'époques
            val_split: Proportion pour la validation
            save_path: Chemin de sauvegarde du modèle
            save_best: Sauvegarder le meilleur modèle
            early_stopping_patience: Patience pour l'early stopping
            verbose: Affichage détaillé

        Returns:
            dict: Historique d'entraînement
        """
        logger.info("🚀 Début de l'entraînement...")

        # Créer les DataLoaders
        train_loader, val_loader = self.create_dataloaders(
            inputs, targets, val_split=val_split
        )

        # Variables pour l'early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Entraînement
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate_epoch(val_loader)

            # Sauvegarder l'historique
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["learning_rates"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            self.training_history["epochs"].append(epoch + 1)

            # Temps d'époque
            epoch_time = time.time() - epoch_start_time

            if verbose:
                logger.info(
                    f"Époque {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Temps: {epoch_time:.1f}s"
                )

            # Early stopping et sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if save_best and save_path:
                    self.save_model(
                        save_path,
                        additional_info={
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "training_history": self.training_history,
                        },
                    )
            else:
                patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"⏰ Early stopping après {epoch + 1} époques")
                    break

        total_time = time.time() - start_time

        logger.info("✅ Entraînement terminé!")
        logger.info(f"  • Temps total: {total_time:.1f}s")
        logger.info(f"  • Meilleure val loss: {best_val_loss:.4f}")
        logger.info(f"  • Loss finale train: {train_loss:.4f}")

        return self.training_history

    def save_model(self, path: str, additional_info: dict = None):
        """
        Sauvegarde le modèle et l'entraîneur.

        Args:
            path: Chemin de sauvegarde
            additional_info: Informations supplémentaires
        """
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "model_config": {
                "num_items": self.model.num_items,
                "embedding_dim": self.model.embedding_dim,
                "seq_length": self.model.seq_length,
                "num_heads": self.model.num_heads,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "feedforward_dim": self.model.feedforward_dim,
            },
        }

        if additional_info:
            save_dict.update(additional_info)

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(save_dict, path)
        logger.info(f"💾 Modèle et entraîneur sauvegardés: {path}")

    def load_checkpoint(self, path: str):
        """
        Charge un checkpoint d'entraînement.

        Args:
            path: Chemin du checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]

        logger.info(f"✅ Checkpoint chargé: {path}")

        return checkpoint


def load_training_data(data_path: str):
    """
    Charge les données d'entraînement depuis un fichier.

    Args:
        data_path: Chemin vers les données

    Returns:
        tuple: (inputs, targets)
    """
    logger.info(f"📂 Chargement des données d'entraînement: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    inputs = data["inputs"]
    targets = data["targets"]

    logger.info(f"✅ Données chargées:")
    logger.info(f"  • Inputs: {inputs.shape}")
    logger.info(f"  • Targets: {targets.shape}")

    return inputs, targets


def main():
    """Fonction principale pour test du module."""
    print("=" * 60)
    print("🚀 TEST DE L'ENTRAÎNEMENT")
    print("=" * 60)

    # Créer un modèle de test
    model = TransformerRecommendationModel(
        num_items=50, embedding_dim=64, seq_length=10, num_heads=4, num_layers=2
    )

    # Créer des données de test
    batch_size = 100
    seq_length = 9  # -1 car inputs sont plus courts que sequences
    inputs = torch.randint(1, 51, (batch_size, seq_length))
    targets = torch.randint(1, 51, (batch_size, seq_length))

    # Créer l'entraîneur
    trainer = ModelTrainer(
        model=model,
        device="cpu",  # Force CPU pour le test
        learning_rate=1e-3,
        batch_size=32,
    )

    # Entraîner le modèle
    history = trainer.train(
        inputs=inputs,
        targets=targets,
        num_epochs=3,
        val_split=0.2,
        save_path="models/test_model.pt",
        verbose=True,
    )

    print(f"\n✅ Test d'entraînement terminé!")
    print(f"  • Époques: {len(history['train_loss'])}")
    print(f"  • Loss finale train: {history['train_loss'][-1]:.4f}")
    print(f"  • Loss finale val: {history['val_loss'][-1]:.4f}")

    # Nettoyer
    if os.path.exists("models/test_model.pt"):
        os.remove("models/test_model.pt")
        os.rmdir("models")


if __name__ == "__main__":
    main()
