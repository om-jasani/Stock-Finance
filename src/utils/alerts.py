"""Alert system for trading signals"""
import pandas as pd
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import json
import os
from .signals import SignalGenerator
from .logger import logger
from .config import Config

class AlertManager:
    """Class for managing trading alerts"""
    
    def __init__(self, alert_file: Optional[str] = None):
        """Initialize AlertManager"""
        self.signal_generator = SignalGenerator()
        self.alert_file = alert_file or os.path.join(Config.DATA_DIR, 'alerts.json')
        self.active_alerts: Dict[str, Dict[str, Any]] = self.load_alerts()
        self.minimum_alert_interval = timedelta(minutes=15)  # Prevent alert spam
        
    def load_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Load saved alerts"""
        try:
            if os.path.exists(self.alert_file):
                with open(self.alert_file, 'r') as f:
                    alerts = json.load(f)
                    
                # Convert string timestamps back to datetime
                for alert in alerts.values():
                    alert['created_at'] = datetime.fromisoformat(alert['created_at'])
                    alert['last_triggered'] = datetime.fromisoformat(alert['last_triggered'])
                return alerts
            return {}
        except Exception as e:
            logger.error(f"Error loading alerts: {str(e)}")
            return {}
            
    def save_alerts(self):
        """Save alerts to file"""
        try:
            # Convert datetime objects to strings
            alerts_to_save = {}
            for alert_id, alert in self.active_alerts.items():
                alert_copy = alert.copy()
                alert_copy['created_at'] = alert_copy['created_at'].isoformat()
                alert_copy['last_triggered'] = alert_copy['last_triggered'].isoformat()
                alerts_to_save[alert_id] = alert_copy
                
            os.makedirs(os.path.dirname(self.alert_file), exist_ok=True)
            with open(self.alert_file, 'w') as f:
                json.dump(alerts_to_save, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving alerts: {str(e)}")
            
    def create_alert(self,
                   symbol: str,
                   alert_type: str,
                   conditions: Dict[str, Any],
                   name: Optional[str] = None,
                   cooldown_minutes: int = 15) -> str:
        """Create new alert"""
        try:
            alert_id = f"{symbol}_{alert_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            alert = {
                'id': alert_id,
                'name': name or f"{symbol} {alert_type} Alert",
                'symbol': symbol,
                'type': alert_type,
                'conditions': conditions,
                'active': True,
                'created_at': datetime.now(),
                'last_triggered': datetime.now() - timedelta(minutes=cooldown_minutes),
                'trigger_count': 0,
                'cooldown_minutes': cooldown_minutes
            }
            
            self.active_alerts[alert_id] = alert
            self.save_alerts()
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise
            
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
                
            self.active_alerts[alert_id].update(updates)
            self.save_alerts()
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert: {str(e)}")
            return False
            
    def delete_alert(self, alert_id: str) -> bool:
        """Delete alert"""
        try:
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
                self.save_alerts()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting alert: {str(e)}")
            return False
            
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all active alerts for triggers"""
        try:
            triggered_alerts = []
            current_time = datetime.now()
            
            for alert_id, alert in self.active_alerts.items():
                if not alert['active']:
                    continue
                    
                # Check cooldown period
                time_since_last_trigger = current_time - alert['last_triggered']
                if time_since_last_trigger < timedelta(minutes=alert['cooldown_minutes']):
                    continue
                    
                # Get current signals
                signals = self.signal_generator.generate_signals(alert['symbol'])
                
                # Check if alert conditions are met
                if self._check_alert_conditions(alert, signals):
                    triggered_alert = self._trigger_alert(alert_id, signals)
                    if triggered_alert:
                        triggered_alerts.append(triggered_alert)
                        
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            return []
            
    def _check_alert_conditions(self, alert: Dict[str, Any], signals: Dict[str, Any]) -> bool:
        """Check if alert conditions are met"""
        try:
            conditions = alert['conditions']
            alert_type = alert['type'].upper()
            
            if alert_type == 'PRICE':
                current_price = signals['price']
                if 'price_above' in conditions and current_price > conditions['price_above']:
                    return True
                if 'price_below' in conditions and current_price < conditions['price_below']:
                    return True
                    
            elif alert_type == 'TECHNICAL':
                # Check technical indicators
                for signal in signals['signals']:
                    if signal['type'] in conditions['indicators']:
                        if conditions.get('signal_strength', 'Low') in ['Any', signal['strength']]:
                            return True
                            
            elif alert_type == 'PATTERN':
                # Check for specific patterns
                for signal in signals['signals']:
                    if signal['type'] == 'PATTERN' and signal['signal'] in conditions['patterns']:
                        return True
                        
            elif alert_type == 'VOLUME':
                if 'volume_multiple' in conditions:
                    current_volume = signals['signals'][0].get('Volume', 0)
                    avg_volume = signals['signals'][0].get('Volume_SMA', 1)
                    if current_volume / avg_volume > conditions['volume_multiple']:
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Error checking alert conditions: {str(e)}")
            return False
            
    def _trigger_alert(self, alert_id: str, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process alert trigger"""
        try:
            alert = self.active_alerts[alert_id]
            current_time = datetime.now()
            
            # Update alert stats
            alert['last_triggered'] = current_time
            alert['trigger_count'] += 1
            
            # Create trigger record
            trigger = {
                'alert_id': alert_id,
                'alert_name': alert['name'],
                'symbol': alert['symbol'],
                'type': alert['type'],
                'timestamp': current_time,
                'price': signal_data['price'],
                'signals': signal_data['signals'],
                'conditions_met': alert['conditions']
            }
            
            self.save_alerts()
            return trigger
            
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")
            return None
            
    def get_alert_history(self, alert_id: Optional[str] = None, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get alert trigger history"""
        try:
            history_file = os.path.join(Config.DATA_DIR, 'alert_history.json')
            if not os.path.exists(history_file):
                return []
                
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            # Filter history based on parameters
            filtered_history = []
            for entry in history:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                
                if alert_id and entry['alert_id'] != alert_id:
                    continue
                if start_date and entry_time < start_date:
                    continue
                if end_date and entry_time > end_date:
                    continue
                    
                filtered_history.append(entry)
                
            return filtered_history
            
        except Exception as e:
            logger.error(f"Error getting alert history: {str(e)}")
            return []