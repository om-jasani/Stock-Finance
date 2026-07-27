from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QLabel, QTableWidget, QTableWidgetItem,
                             QDialog, QLineEdit, QFormLayout, QDoubleSpinBox,
                             QMessageBox, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSlot
import pandas as pd
import json
import os
from ...utils.data_fetcher import DataFetcher
from ...utils.config import Config
from ...utils.logger import logger
from ..workers import FetchWorker

class AddStockDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Stock to Portfolio")
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout(self)
        
        # Symbol input
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g., AAPL")
        layout.addRow("Symbol:", self.symbol_input)
        
        # Shares input
        self.shares_input = QDoubleSpinBox()
        self.shares_input.setRange(0.0001, 1000000)
        self.shares_input.setDecimals(4)
        self.shares_input.setValue(1)
        layout.addRow("Shares:", self.shares_input)
        
        # Purchase price input
        self.price_input = QDoubleSpinBox()
        self.price_input.setRange(0.01, 1000000)
        self.price_input.setDecimals(2)
        layout.addRow("Purchase Price (₹):", self.price_input)
        
        # Buttons
        button_box = QHBoxLayout()
        
        ok_button = QPushButton("Add")
        ok_button.clicked.connect(self.accept)
        button_box.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)
        
        layout.addRow(button_box)
        
    def get_stock_info(self):
        return {
            'symbol': self.symbol_input.text().upper(),
            'shares': self.shares_input.value(),
            'purchase_price': self.price_input.value()
        }

class PortfolioWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_fetcher = DataFetcher()
        self.portfolio_file = os.path.join(Config.DATA_DIR, 'portfolio.json')
        self.portfolio = self.load_portfolio()
        self.fetch_worker = None
        self.init_ui()
        self.update_portfolio()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = QFrame()
        control_panel.setObjectName("control-panel")
        control_layout = QHBoxLayout(control_panel)
        
        # Add stock button
        add_button = QPushButton("Add Stock")
        add_button.clicked.connect(self.add_stock)
        control_layout.addWidget(add_button)
        
        # Remove stock button
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        control_layout.addWidget(remove_button)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_portfolio)
        control_layout.addWidget(refresh_button)
        
        layout.addWidget(control_panel)
        
        # Portfolio table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            'Symbol', 'Shares', 'Purchase Price', 'Current Price',
            'Invested', 'Market Value', 'Gain/Loss', 'Return %'
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        
        # Summary panel
        summary_panel = QFrame()
        summary_panel.setObjectName("summary-panel")
        summary_layout = QHBoxLayout(summary_panel)
        
        self.total_invested_label = QLabel()
        self.total_value_label = QLabel()
        self.total_gain_label = QLabel()
        self.total_return_label = QLabel()
        
        summary_layout.addWidget(self.total_invested_label)
        summary_layout.addWidget(self.total_value_label)
        summary_layout.addWidget(self.total_gain_label)
        summary_layout.addWidget(self.total_return_label)
        
        layout.addWidget(summary_panel)
        
    def load_portfolio(self) -> dict:
        """Load portfolio from file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load portfolio from {self.portfolio_file}: {e}")
        return {}
        
    def save_portfolio(self):
        """Save portfolio to file"""
        os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f)
            
    def add_stock(self):
        """Add stock to portfolio"""
        dialog = AddStockDialog(self)
        if dialog.exec():
            stock_info = dialog.get_stock_info()
            symbol = stock_info['symbol']

            self._add_worker = FetchWorker(self.data_fetcher.get_stock_data, symbol, period='1d')
            self._add_worker.finished.connect(
                lambda df: self._on_add_stock_verified(symbol, stock_info, df))
            self._add_worker.error.connect(
                lambda msg: QMessageBox.warning(self, "Error", f"Could not fetch data for symbol {symbol}: {msg}"))
            self._add_worker.start()

    def _on_add_stock_verified(self, symbol: str, stock_info: dict, df):
        """Finish adding the stock once its data has been verified off the GUI thread"""
        if df is None or df.empty:
            QMessageBox.warning(self, "Error", f"Could not fetch data for symbol {symbol}")
            return

        # Add to portfolio
        if symbol in self.portfolio:
            # Update existing position
            existing_shares = self.portfolio[symbol]['shares']
            existing_cost = existing_shares * self.portfolio[symbol]['purchase_price']
            new_shares = stock_info['shares']
            new_cost = new_shares * stock_info['purchase_price']
            total_shares = existing_shares + new_shares
            avg_price = (existing_cost + new_cost) / total_shares

            self.portfolio[symbol] = {
                'shares': total_shares,
                'purchase_price': avg_price
            }
        else:
            # Add new position
            self.portfolio[symbol] = {
                'shares': stock_info['shares'],
                'purchase_price': stock_info['purchase_price']
            }

        self.save_portfolio()
        self.update_portfolio()

    def remove_selected(self):
        """Remove selected stocks from portfolio"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return
            
        symbols = set()
        for item in selected_items:
            if item.column() == 0:  # Symbol column
                symbols.add(item.text())
                
        for symbol in symbols:
            self.portfolio.pop(symbol, None)
            
        self.save_portfolio()
        self.update_portfolio()
        
    def update_portfolio(self):
        """Refresh portfolio prices off the GUI thread, then re-render the table"""
        if not self.portfolio:
            self._render_portfolio({})
            return

        if self.fetch_worker is not None and self.fetch_worker.isRunning():
            return

        self.fetch_worker = FetchWorker(
            self.data_fetcher.get_multiple_stocks, list(self.portfolio.keys()), period='1d'
        )
        self.fetch_worker.finished.connect(self._render_portfolio)
        self.fetch_worker.error.connect(lambda msg: logger.error(f"Portfolio refresh error: {msg}"))
        self.fetch_worker.start()

    def _render_portfolio(self, price_data: dict):
        """Rebuild the portfolio table from freshly fetched prices"""
        self.table.setRowCount(len(self.portfolio))

        total_value = 0
        total_cost = 0
        row = 0

        for symbol, data in self.portfolio.items():
            df = price_data.get(symbol)
            if df is None or df.empty:
                continue

            current_price = df['Close'].iloc[-1]
            shares = data['shares']
            purchase_price = data['purchase_price']

            market_value = shares * current_price
            cost_basis = shares * purchase_price
            gain_loss = market_value - cost_basis
            return_pct = (gain_loss / cost_basis) * 100 if cost_basis else 0

            # Update totals
            total_value += market_value
            total_cost += cost_basis
            
            # Add row to table
            self.table.setItem(row, 0, QTableWidgetItem(symbol))
            self.table.setItem(row, 1, QTableWidgetItem(f"{shares:,.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"₹{purchase_price:,.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"₹{current_price:,.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"₹{cost_basis:,.2f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"₹{market_value:,.2f}"))
            
            gain_loss_item = QTableWidgetItem(f"₹{gain_loss:+,.2f}")
            gain_loss_item.setForeground(Qt.GlobalColor.green if gain_loss >= 0 else Qt.GlobalColor.red)
            self.table.setItem(row, 6, gain_loss_item)
            
            return_item = QTableWidgetItem(f"{return_pct:+.2f}%")
            return_item.setForeground(Qt.GlobalColor.green if return_pct >= 0 else Qt.GlobalColor.red)
            self.table.setItem(row, 7, return_item)
            
            row += 1
            
        # Update summary
        total_gain = total_value - total_cost
        total_return = (total_gain / total_cost) * 100 if total_cost > 0 else 0
        
        self.total_invested_label.setText(f"Total Invested: ₹{total_cost:,.2f}")

        self.total_value_label.setText(f"Total Value: ₹{total_value:,.2f}")
        
        self.total_gain_label.setText(f"Total Gain/Loss: ₹{total_gain:+,.2f}")
        self.total_gain_label.setStyleSheet(
            "color: green;" if total_gain >= 0 else "color: red;"
        )
        
        self.total_return_label.setText(f"Total Return: {total_return:+.2f}%")
        self.total_return_label.setStyleSheet(
            "color: green;" if total_return >= 0 else "color: red;"
        )
