"""
Main Pipeline for Banking Recommendation System

Complete pipeline for generating data, preprocessing, training, and evaluating
a sequential recommendation model for banking products using Transformers4Rec.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from data_generator import BankingDataGenerator
from data_preprocessor import DataPreprocessor, check_transformers4rec_availability
from model_trainer import ModelTrainer, TrainingConfig, train_banking_recommendation_model


class BankingRecommendationPipeline:
    """
    Complete pipeline for banking recommendation system.
    
    Orchestrates data generation, preprocessing, model training,
    and evaluation for sequential banking product recommendations.
    """
    
    def __init__(self, output_dir: str = "./banking_ai_pipeline"):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Base output directory for all pipeline artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Sub-directories
        self.data_dir = self.output_dir / "data"
        self.processed_dir = self.output_dir / "processed_data"
        self.models_dir = self.output_dir / "models"
        
        for dir_path in [self.data_dir, self.processed_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_data(self, 
                     n_customers: int = 3000, 
                     avg_interactions: int = 20,
                     seed: int = 42) -> str:
        """
        Generate banking interaction dataset.
        
        Args:
            n_customers: Number of customers to generate
            avg_interactions: Average interactions per customer
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated dataset
        """
        print(f"🔄 Step 1: Generating banking dataset...")
        print(f"   Customers: {n_customers}")
        print(f"   Avg interactions: {avg_interactions}")
        
        generator = BankingDataGenerator(seed=seed)
        dataset = generator.generate_dataset(n_customers, avg_interactions)
        
        # Save dataset
        dataset_path = self.data_dir / "banking_interactions.csv"
        dataset.to_csv(dataset_path, index=False)
        
        # Save dataset statistics
        stats = generator.get_dataset_stats(dataset)
        stats_path = self.data_dir / "dataset_stats.json"
        
        # Convert non-serializable objects
        serializable_stats = {}
        for key, value in stats.items():
            if key == "date_range":
                serializable_stats[key] = [str(value[0]), str(value[1])]
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_stats[key] = value
            else:
                serializable_stats[key] = str(value)
        
        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"✅ Dataset generated: {len(dataset)} interactions")
        print(f"   Saved to: {dataset_path}")
        print(f"   Stats saved to: {stats_path}")
        
        return str(dataset_path)
    
    def preprocess_data(self, dataset_path: str, test_split: float = 0.2) -> Dict[str, str]:
        """
        Preprocess data for Transformers4Rec training.
        
        Args:
            dataset_path: Path to raw dataset
            test_split: Test set ratio
            
        Returns:
            Dictionary with processed file paths
        """
        print(f"🔄 Step 2: Preprocessing data...")
        print(f"   Test split: {test_split}")
        
        import pandas as pd
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"   Loaded: {len(df)} interactions")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(test_split_ratio=test_split)
        
        # Run preprocessing pipeline
        train_df, test_df, schema_info = preprocessor.prepare_for_transformers4rec(df)
        
        # Save processed data
        file_paths = preprocessor.save_processed_data(
            train_df, test_df, str(self.processed_dir)
        )
        
        # Get and save summary
        summary = preprocessor.get_data_summary(train_df, test_df)
        summary_path = self.processed_dir / "preprocessing_summary.json"
        with open(summary_path, "w") as f:
            # Convert non-serializable types
            serializable_summary = {}
            for key, value in summary.items():
                if key == "data_types":
                    serializable_summary[key] = {k: str(v) for k, v in value.items()}
                elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_summary[key] = value
                else:
                    serializable_summary[key] = str(value)
            json.dump(serializable_summary, f, indent=2)
        
        file_paths["summary"] = str(summary_path)
        
        print(f"✅ Data preprocessing completed")
        print(f"   Train samples: {summary['train_samples']}")
        print(f"   Test samples: {summary['test_samples']}")
        print(f"   Files saved to: {self.processed_dir}")
        
        return file_paths
    
    def train_model(self, 
                   processed_files: Dict[str, str],
                   config: TrainingConfig = None) -> Dict[str, Any]:
        """
        Train recommendation model.
        
        Args:
            processed_files: Paths to processed data files
            config: Training configuration
            
        Returns:
            Dictionary with training results and model paths
            
        Raises:
            RuntimeError: If Transformers4Rec is not available or training fails
        """
        print(f"🔄 Step 3: Training model...")
        
        if config is None:
            config = TrainingConfig(
                epochs=3,
                learning_rate=0.001,
                batch_size=64,
                output_dir=str(self.models_dir)
            )
        
        print(f"   Epochs: {config.epochs}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Batch size: {config.batch_size}")
        
        # Check T4Rec availability (will raise error if not available)
        check_transformers4rec_availability()
        
        # Run training (will raise error if fails)
        results, saved_files = train_banking_recommendation_model(
            train_data_path=processed_files["train"],
            test_data_path=processed_files["test"],
            schema_info_path=processed_files["schema"],
            config=config
        )
        
        training_info = {
            "results": {
                "final_loss": results.final_loss,
                "accuracy": results.accuracy,
                "recall_at_10": results.recall_at_10,
                "ndcg_at_10": results.ndcg_at_10,
                "training_time": results.training_time,
                "epochs_completed": results.epochs_completed,
                "model_parameters": results.model_parameters,
                "convergence_achieved": results.convergence_achieved,
            },
            "saved_files": saved_files,
            "transformers4rec_used": True,
        }
        
        print(f"✅ Model training completed")
        print(f"   Final Loss: {results.final_loss:.4f}")
        print(f"   Accuracy: {results.accuracy:.1%}")
        print(f"   Recall@10: {results.recall_at_10:.1%}")
        print(f"   NDCG@10: {results.ndcg_at_10:.3f}")
        print(f"   Training time: {results.training_time:.1f}s")
        print(f"   Model saved to: {self.models_dir}")
        
        return training_info
    
    def run_complete_pipeline(self,
                            n_customers: int = 3000,
                            avg_interactions: int = 20,
                            test_split: float = 0.2,
                            training_config: TrainingConfig = None) -> Dict[str, Any]:
        """
        Run complete pipeline from data generation to model training.
        
        Args:
            n_customers: Number of customers to generate
            avg_interactions: Average interactions per customer
            test_split: Test set ratio
            training_config: Training configuration
            
        Returns:
            Complete pipeline results
        """
        print("🚀 Starting complete banking recommendation pipeline...")
        print(f"   Output directory: {self.output_dir}")
        
        pipeline_results = {}
        
        try:
            # Step 1: Generate data
            dataset_path = self.generate_data(n_customers, avg_interactions)
            pipeline_results["dataset_path"] = dataset_path
            
            # Step 2: Preprocess data
            processed_files = self.preprocess_data(dataset_path, test_split)
            pipeline_results["processed_files"] = processed_files
            
            # Step 3: Train model
            training_info = self.train_model(processed_files, training_config)
            pipeline_results["training"] = training_info
            
            # Save complete pipeline results
            pipeline_summary = {
                "pipeline_config": {
                    "n_customers": n_customers,
                    "avg_interactions": avg_interactions,
                    "test_split": test_split,
                    "training_config": {
                        "epochs": training_config.epochs if training_config else 3,
                        "learning_rate": training_config.learning_rate if training_config else 0.001,
                        "batch_size": training_config.batch_size if training_config else 64,
                    }
                },
                "results": pipeline_results,
                "success": True,
                "output_directory": str(self.output_dir),
            }
            
            summary_path = self.output_dir / "pipeline_summary.json"
            with open(summary_path, "w") as f:
                json.dump(pipeline_summary, f, indent=2)
            
            print("🎉 Complete pipeline finished successfully!")
            print(f"   Pipeline summary saved to: {summary_path}")
            
            return pipeline_summary
            
        except Exception as e:
            error_info = {
                "success": False,
                "error": str(e),
                "partial_results": pipeline_results,
            }
            
            error_path = self.output_dir / "pipeline_error.json"
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
            
            print(f"❌ Pipeline failed: {e}")
            print(f"   Error info saved to: {error_path}")
            
            raise
    
    def load_trained_model(self, model_dir: str = None):
        """
        Load a previously trained model.
        
        Args:
            model_dir: Model directory (uses default if None)
            
        Returns:
            Loaded model
            
        Raises:
            RuntimeError: If Transformers4Rec is not available or model not found
        """
        if model_dir is None:
            model_dir = self.models_dir
        else:
            model_dir = Path(model_dir)
        
        # Check T4Rec availability first
        check_transformers4rec_availability()
        
        # Check if T4Rec model exists
        t4rec_model_path = model_dir / "transformers4rec_model"
        if t4rec_model_path.exists():
            try:
                import transformers4rec.torch as tr
                model = tr.Model.load(str(t4rec_model_path))
                print(f"✅ Transformers4Rec model loaded from {t4rec_model_path}")
                return model
            except Exception as e:
                raise RuntimeError(f"❌ Could not load T4Rec model: {e}")
        
        # Check for PyTorch weights
        weights_path = model_dir / "model_weights.pth"
        if weights_path.exists():
            raise RuntimeError(
                f"❌ Found PyTorch weights at {weights_path} but model architecture is needed for loading. "
                "Use the complete Transformers4Rec model instead."
            )
        
        raise RuntimeError(f"❌ No trained model found in {model_dir}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Banking Recommendation Pipeline")
    parser.add_argument("--customers", type=int, default=3000, help="Number of customers")
    parser.add_argument("--interactions", type=int, default=20, help="Avg interactions per customer")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="./banking_ai_pipeline", help="Output directory")
    
    args = parser.parse_args()
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=f"{args.output_dir}/models"
    )
    
    # Initialize and run pipeline
    pipeline = BankingRecommendationPipeline(args.output_dir)
    
    results = pipeline.run_complete_pipeline(
        n_customers=args.customers,
        avg_interactions=args.interactions,
        test_split=args.test_split,
        training_config=training_config
    )
    
    print("\n📊 Final Results Summary:")
    training_results = results["results"]["training"]["results"]
    print(f"Accuracy: {training_results['accuracy']:.1%}")
    print(f"Recall@10: {training_results['recall_at_10']:.1%}")
    print(f"NDCG@10: {training_results['ndcg_at_10']:.3f}")
    print(f"Training Time: {training_results['training_time']:.1f}s")
    print(f"Model Parameters: {training_results['model_parameters']:,}")


if __name__ == "__main__":
    main()