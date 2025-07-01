"""
Discord notification utility for CSI-Predictor.
Sends training/evaluation results with dashboard images to Discord webhooks.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger


class DiscordNotifier:
    """Discord webhook notifier for training results."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL. If None, loads from environment.
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Discord webhook not configured - notifications disabled")
        else:
            logger.info("Discord notifications enabled")
    
    def format_config_message(self, config, model_name: str = "Unknown") -> Dict[str, Any]:
        """
        Format configuration parameters into organized Discord embed.
        
        Args:
            config: Configuration object
            model_name: Name of the model
            
        Returns:
            Discord embed dictionary
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Organize parameters into logical sections
        model_section = f"""
**Architecture**: {getattr(config, 'model_arch', 'Unknown')}
**Zone Focus Method**: {getattr(config, 'zone_focus_method', 'Unknown')}
**Use Official Processor**: {getattr(config, 'use_official_processor', 'Unknown')}
"""
        
        training_section = f"""
**Batch Size**: {getattr(config, 'batch_size', 'Unknown')}
**Learning Rate**: {getattr(config, 'learning_rate', 'Unknown')}
**Optimizer**: {getattr(config, 'optimizer', 'Unknown')}
**Epochs**: {getattr(config, 'n_epochs', 'Unknown')}
**Patience**: {getattr(config, 'patience', 'Unknown')}
"""
        
        # Handle zone masking parameters
        masking_section = f"""
**Use Segmentation Masking**: {getattr(config, 'use_segmentation_masking', 'Unknown')}
**Masking Strategy**: {getattr(config, 'masking_strategy', 'Unknown')}
**Attention Strength**: {getattr(config, 'attention_strength', 'Unknown')}
"""
        
        # Handle excluded file IDs (truncate if too long)
        excluded_ids = getattr(config, 'excluded_file_ids', [])
        if isinstance(excluded_ids, list) and excluded_ids:
            excluded_count = len(excluded_ids)
            if excluded_count > 10:
                excluded_display = f"{excluded_count} files excluded (first 5: {', '.join(map(str, excluded_ids[:5]))}...)"
            else:
                excluded_display = f"{excluded_count} files excluded: {', '.join(map(str, excluded_ids))}"
        else:
            excluded_display = "None"
        
        data_section = f"""
**Excluded File IDs**: {excluded_display}
"""
        
        embed = {
            "title": f"🚀 CSI-Predictor Training Completed",
            "description": f"**Model**: `{model_name}`",
            "color": 0x00ff00,  # Green color
            "fields": [
                {
                    "name": "🏗️ Model Configuration",
                    "value": model_section.strip(),
                    "inline": True
                },
                {
                    "name": "⚙️ Training Parameters", 
                    "value": training_section.strip(),
                    "inline": True
                },
                {
                    "name": "🎭 Zone Masking Settings",
                    "value": masking_section.strip(),
                    "inline": False
                },
                {
                    "name": "📊 Data Configuration",
                    "value": data_section.strip(),
                    "inline": False
                }
            ],
            "footer": {
                "text": f"CSI-Predictor • {timestamp}"
            }
        }
        
        return embed
    
    def format_results_message(self, 
                             train_results: Optional[Dict] = None,
                             val_results: Optional[Dict] = None,
                             test_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format training/evaluation results into Discord embed.
        
        Args:
            train_results: Training metrics
            val_results: Validation metrics
            test_results: Test metrics
            
        Returns:
            Discord embed dictionary
        """
        fields = []
        
        if train_results:
            train_text = f"""
**Accuracy**: {train_results.get('accuracy', 'N/A'):.4f}
**Loss**: {train_results.get('loss', 'N/A'):.4f}
**F1 Score**: {train_results.get('f1_macro', 'N/A'):.4f}
**Precision**: {train_results.get('precision_macro', 'N/A'):.4f}
"""
            fields.append({
                "name": "📈 Training Results",
                "value": train_text.strip(),
                "inline": True
            })
        
        if val_results:
            val_text = f"""
**Accuracy**: {val_results.get('accuracy', 'N/A'):.4f}
**Loss**: {val_results.get('loss', 'N/A'):.4f}
**F1 Score**: {val_results.get('f1_macro', 'N/A'):.4f}
**Precision**: {val_results.get('precision_macro', 'N/A'):.4f}
"""
            fields.append({
                "name": "✅ Validation Results",
                "value": val_text.strip(),
                "inline": True
            })
        
        if test_results:
            test_text = f"""
**Accuracy**: {test_results.get('accuracy', 'N/A'):.4f}
**Loss**: {test_results.get('loss', 'N/A'):.4f}
**F1 Score**: {test_results.get('f1_macro', 'N/A'):.4f}
**Precision**: {test_results.get('precision_macro', 'N/A'):.4f}
"""
            fields.append({
                "name": "🏆 Test Results",
                "value": test_text.strip(),
                "inline": True
            })
        
        embed = {
            "title": "📊 Model Performance Metrics",
            "color": 0x0099ff,  # Blue color
            "fields": fields
        }
        
        return embed
    
    def send_training_completion(self,
                               config,
                               model_name: str,
                               dashboard_image_path: Optional[str] = None,
                               train_results: Optional[Dict] = None,
                               val_results: Optional[Dict] = None,
                               test_results: Optional[Dict] = None) -> bool:
        """
        Send training completion notification to Discord.
        
        Args:
            config: Configuration object
            model_name: Name of the trained model
            dashboard_image_path: Path to dashboard image
            train_results: Training metrics
            val_results: Validation metrics  
            test_results: Test metrics
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug("Discord notifications disabled, skipping")
            return False
        
        try:
            # Prepare embeds
            config_embed = self.format_config_message(config, model_name)
            results_embed = self.format_results_message(train_results, val_results, test_results)
            
            # Prepare the payload
            payload = {
                "content": f"🎯 **CSI-Predictor Training Complete!**\n",
                "embeds": [config_embed, results_embed]
            }
            
            files = {}
            
            # Add dashboard image if available
            if dashboard_image_path and Path(dashboard_image_path).exists():
                try:
                    with open(dashboard_image_path, 'rb') as f:
                        files['file'] = (Path(dashboard_image_path).name, f, 'image/png')
                        
                        # Add image reference to embed
                        results_embed["image"] = {"url": f"attachment://{Path(dashboard_image_path).name}"}
                        payload["embeds"][1] = results_embed
                        
                        # Send with file
                        response = requests.post(
                            self.webhook_url,
                            data={"payload_json": json.dumps(payload)},
                            files=files,
                            timeout=30
                        )
                except Exception as e:
                    logger.warning(f"Failed to attach dashboard image: {e}")
                    # Send without image
                    response = requests.post(self.webhook_url, json=payload, timeout=30)
            else:
                # Send without image
                response = requests.post(self.webhook_url, json=payload, timeout=30)
            
            if response.status_code == 204:
                logger.info("✅ Discord notification sent successfully")
                return True
            else:
                logger.error(f"Discord notification failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Discord notification failed due to network error: {e}")
            return False
        except Exception as e:
            logger.error(f"Discord notification failed due to unexpected error: {e}")
            return False
    
    def send_evaluation_completion(self,
                                 config,
                                 model_name: str,
                                 dashboard_image_path: Optional[str] = None,
                                 val_results: Optional[Dict] = None,
                                 test_results: Optional[Dict] = None) -> bool:
        """
        Send evaluation completion notification to Discord.
        
        Args:
            config: Configuration object
            model_name: Name of the evaluated model
            dashboard_image_path: Path to dashboard image
            val_results: Validation metrics
            test_results: Test metrics
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug("Discord notifications disabled, skipping")
            return False
        
        try:
            # Prepare embeds
            config_embed = self.format_config_message(config, model_name)
            config_embed["title"] = "🔍 CSI-Predictor Evaluation Completed"
            config_embed["color"] = 0xff9900  # Orange color for evaluation
            
            results_embed = self.format_results_message(None, val_results, test_results)
            
            # Prepare the payload
            payload = {
                "content": f"📊 **CSI-Predictor Evaluation Complete!**\n",
                "embeds": [config_embed, results_embed]
            }
            
            files = {}
            
            # Add dashboard image if available
            if dashboard_image_path and Path(dashboard_image_path).exists():
                try:
                    with open(dashboard_image_path, 'rb') as f:
                        files['file'] = (Path(dashboard_image_path).name, f, 'image/png')
                        
                        # Add image reference to embed
                        results_embed["image"] = {"url": f"attachment://{Path(dashboard_image_path).name}"}
                        payload["embeds"][1] = results_embed
                        
                        # Send with file
                        response = requests.post(
                            self.webhook_url,
                            data={"payload_json": json.dumps(payload)},
                            files=files,
                            timeout=30
                        )
                except Exception as e:
                    logger.warning(f"Failed to attach dashboard image: {e}")
                    # Send without image
                    response = requests.post(self.webhook_url, json=payload, timeout=30)
            else:
                # Send without image
                response = requests.post(self.webhook_url, json=payload, timeout=30)
            
            if response.status_code == 204:
                logger.info("✅ Discord evaluation notification sent successfully")
                return True
            else:
                logger.error(f"Discord evaluation notification failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Discord evaluation notification failed due to network error: {e}")
            return False
        except Exception as e:
            logger.error(f"Discord evaluation notification failed due to unexpected error: {e}")
            return False


# Global instance for easy access
discord_notifier = DiscordNotifier()


def send_training_notification(config, model_name: str, dashboard_image_path: Optional[str] = None,
                             train_results: Optional[Dict] = None, val_results: Optional[Dict] = None,
                             test_results: Optional[Dict] = None) -> bool:
    """
    Convenience function to send training completion notification.
    
    Args:
        config: Configuration object
        model_name: Name of the trained model
        dashboard_image_path: Path to dashboard image
        train_results: Training metrics
        val_results: Validation metrics
        test_results: Test metrics
        
    Returns:
        True if successful, False otherwise
    """
    return discord_notifier.send_training_completion(
        config, model_name, dashboard_image_path, train_results, val_results, test_results
    )


def send_evaluation_notification(config, model_name: str, dashboard_image_path: Optional[str] = None,
                                val_results: Optional[Dict] = None, test_results: Optional[Dict] = None) -> bool:
    """
    Convenience function to send evaluation completion notification.
    
    Args:
        config: Configuration object
        model_name: Name of the evaluated model
        dashboard_image_path: Path to dashboard image
        val_results: Validation metrics
        test_results: Test metrics
        
    Returns:
        True if successful, False otherwise
    """
    return discord_notifier.send_evaluation_completion(
        config, model_name, dashboard_image_path, val_results, test_results
    ) 