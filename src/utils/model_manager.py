"""Model management and persistence"""
import os
import json
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from .model_trainer import StockPredictor
from .logger import logger
from .config import Config

class ModelManager:
    """Manage model lifecycle and storage"""
    
    def __init__(self):
        """Initialize ModelManager"""
        self.models_dir = Config.MODELS_DIR
        self.metadata_file = os.path.join(self.models_dir, 'model_metadata.json')
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading model metadata: {str(e)}")
            return {}
            
    def _save_metadata(self):
        """Save model metadata"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            
    def save_model(self, 
                  symbol: str, 
                  model: StockPredictor,
                  metrics: Dict[str, float],
                  version: Optional[str] = None) -> str:
        """Save trained model"""
        try:
            # Generate model ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = f"{symbol}_{timestamp}"
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model files
            model_path = os.path.join(model_dir, 'model.keras')
            scaler_path = os.path.join(model_dir, 'scalers.pkl')
            
            # Save model and scalers
            model.model.save(model_path)
            model.data_pipeline.save_scalers(scaler_path)
            
            # Update metadata
            self.metadata[model_id] = {
                'symbol': symbol,
                'version': version or '1.0.0',
                'created_at': datetime.now().isoformat(),
                'metrics': metrics,
                'model_path': model_path,
                'scaler_path': scaler_path,
                'status': 'active'
            }
            
            self._save_metadata()
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, model_id: str) -> Optional[StockPredictor]:
        """Load model by ID"""
        try:
            if model_id not in self.metadata:
                return None
                
            model_info = self.metadata[model_id]
            if model_info['status'] != 'active':
                return None
                
            model = StockPredictor(model_path=model_info['model_path'])
            
            if model.load_model():
                return model
            return None
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
            
    def get_latest_model(self, symbol: str) -> Optional[StockPredictor]:
        """Get latest model for symbol"""
        try:
            # Find all active models for symbol
            symbol_models = {
                model_id: info for model_id, info in self.metadata.items()
                if info['symbol'] == symbol and info['status'] == 'active'
            }
            
            if not symbol_models:
                return None
                
            # Get latest model by timestamp
            latest_id = max(symbol_models.keys())
            return self.load_model(latest_id)
            
        except Exception as e:
            logger.error(f"Error getting latest model for {symbol}: {str(e)}")
            return None
            
    def delete_model(self, model_id: str) -> bool:
        """Delete model"""
        try:
            if model_id not in self.metadata:
                return False
                
            # Get model info
            model_info = self.metadata[model_id]
            model_dir = os.path.dirname(model_info['model_path'])
            
            # Remove model files
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                
            # Update metadata
            del self.metadata[model_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
            
    def archive_model(self, model_id: str) -> bool:
        """Archive model (mark as inactive)"""
        try:
            if model_id not in self.metadata:
                return False
                
            self.metadata[model_id]['status'] = 'archived'
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error archiving model {model_id}: {str(e)}")
            return False
            
    def list_models(self, 
                   symbol: Optional[str] = None, 
                   status: str = 'active') -> List[Dict[str, Any]]:
        """List available models"""
        try:
            models = []
            for model_id, info in self.metadata.items():
                if ((symbol is None or info['symbol'] == symbol) and
                    (status is None or info['status'] == status)):
                    models.append({
                        'id': model_id,
                        **info
                    })
            
            # Sort by creation date
            return sorted(models, 
                        key=lambda x: datetime.fromisoformat(x['created_at']),
                        reverse=True)
                        
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
            
    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get model performance metrics"""
        try:
            if model_id in self.metadata:
                return self.metadata[model_id].get('metrics')
            return None
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return None
            
    def update_metrics(self, model_id: str, new_metrics: Dict[str, float]) -> bool:
        """Update model metrics"""
        try:
            if model_id not in self.metadata:
                return False
                
            self.metadata[model_id]['metrics'].update(new_metrics)
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Error updating model metrics: {str(e)}")
            return False
            
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            total_size = 0
            model_count = 0
            symbols = set()
            
            for model_id, info in self.metadata.items():
                if info['status'] == 'active':
                    model_count += 1
                    symbols.add(info['symbol'])
                    
                    # Calculate storage size
                    if os.path.exists(info['model_path']):
                        total_size += os.path.getsize(info['model_path'])
                    if os.path.exists(info['scaler_path']):
                        total_size += os.path.getsize(info['scaler_path'])
                        
            return {
                'total_models': model_count,
                'active_symbols': len(symbols),
                'storage_size_mb': total_size / (1024 * 1024),
                'symbols': list(symbols)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage info: {str(e)}")
            return {}