"""
Weights & Biases logging for CSI-Predictor evaluation.

This module contains W&B logging functionality extracted from the original src/evaluate.py file.
"""

import wandb
from typing import Dict
from src.utils.logging import logger

def log_to_wandb(
    val_metrics: Dict,
    test_metrics: Dict,
    val_confusion_matrices: Dict,
    test_confusion_matrices: Dict,
    val_reports: Dict,
    test_reports: Dict,
    config,
    model_path: str,
    eval_model_name: str
) -> None:
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        val_metrics: Validation metrics
        test_metrics: Test metrics
        val_confusion_matrices: Validation confusion matrices
        test_confusion_matrices: Test confusion matrices
        val_reports: Validation classification reports
        test_reports: Test classification reports
        config: Configuration object
        model_path: Path to the evaluated model
        eval_model_name: Name of the evaluation run
    """
    try:
        # Initialize wandb run
        wandb.init(
            project="csi-predictor",
            name=eval_model_name,
            dir=config.wandb_dir,
            config={
                "model_path": model_path,
                "model_arch": config.model_arch,
                "batch_size": config.batch_size,
                "device": config.device,
                "evaluation_type": "model_evaluation"
            }
        )
        
        # Log overall metrics
        wandb.log({
            "validation/overall_f1_macro": val_metrics.get('f1_macro', 0.0),
            "validation/overall_accuracy": val_metrics.get('accuracy', 0.0),
            "validation/overall_precision_macro": val_metrics.get('precision_macro', 0.0),
            "validation/overall_recall_macro": val_metrics.get('recall_macro', 0.0),
            "test/overall_f1_macro": test_metrics.get('f1_macro', 0.0),
            "test/overall_accuracy": test_metrics.get('accuracy', 0.0),
            "test/overall_precision_macro": test_metrics.get('precision_macro', 0.0),
            "test/overall_recall_macro": test_metrics.get('recall_macro', 0.0),
        })
        
        # Log per-zone metrics
        zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
        
        for zone_name in zone_names:
            if zone_name in val_metrics:
                wandb.log({
                    f"validation/{zone_name}_f1": val_metrics[zone_name].get('f1_macro', 0.0),
                    f"validation/{zone_name}_accuracy": val_metrics[zone_name].get('accuracy', 0.0),
                    f"test/{zone_name}_f1": test_metrics[zone_name].get('f1_macro', 0.0),
                    f"test/{zone_name}_accuracy": test_metrics[zone_name].get('accuracy', 0.0),
                })
        
        # Log confusion matrices
        for zone_name in zone_names:
            if zone_name in val_confusion_matrices:
                # Create confusion matrix plot for validation
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Validation confusion matrix
                sns.heatmap(val_confusion_matrices[zone_name], annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title(f'Validation - {zone_name}')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                
                # Test confusion matrix
                if zone_name in test_confusion_matrices:
                    sns.heatmap(test_confusion_matrices[zone_name], annot=True, fmt='d', cmap='Blues', ax=ax2)
                    ax2.set_title(f'Test - {zone_name}')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                
                plt.tight_layout()
                wandb.log({f"confusion_matrices/{zone_name}": wandb.Image(fig)})
                plt.close()
        
        # Log classification reports as tables
        for zone_name in zone_names:
            if zone_name in val_reports:
                # Convert classification report to wandb table
                report = val_reports[zone_name]
                if 'accuracy' in report:
                    wandb.log({
                        f"validation/{zone_name}_classification_report": wandb.Table(
                            columns=["Metric", "Value"],
                            data=[
                                ["Accuracy", report['accuracy']],
                                ["Macro Avg Precision", report.get('macro avg', {}).get('precision', 0.0)],
                                ["Macro Avg Recall", report.get('macro avg', {}).get('recall', 0.0)],
                                ["Macro Avg F1", report.get('macro avg', {}).get('f1-score', 0.0)],
                            ]
                        )
                    })
        
        # Finish wandb run
        wandb.finish()
        
        logger.info(f"Successfully logged evaluation results to W&B for {eval_model_name}")
        
    except Exception as e:
        logger.warning(f"Could not log to W&B: {e}")
        logger.info("Continuing without W&B logging")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 