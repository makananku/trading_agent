```python
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class TradingLogger:
    def __init__(self):
        self.trades = []
        self.daily_summary = {}
        self.model_metrics = {}  # Store model metrics if provided
        self.feature_importance = None  # Store feature importance if provided

    def set_model_metrics(self, y_true, y_pred, y_pred_proba=None, feature_importance=None):
        """Store model performance metrics for summary."""
        self.model_metrics = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        self.feature_importance = feature_importance

    def log_trade(self, trade_type, price, shares, profit=None, reason=None, timestamp=None, balance=None, indicators=None):
        trade_info = {
            'timestamp': timestamp or datetime.now(),
            'type': trade_type,
            'price': price,
            'shares': shares,
            'profit': profit,
            'reason': reason,
            'balance': balance,
            'indicators': indicators or {}
        }
        self.trades.append(trade_info)
        
        # Log trade with detailed format
        if trade_type == 'buy':
            msg = f"🟢 BUY  | {timestamp} | {shares:>6.0f} shares @ Rp{price:>8.0f} | Balance: Rp{balance:>12,.0f}"
            if indicators:
                msg += f" | RSI:{indicators.get('rsi', 0):.1f} MACD:{indicators.get('macd', 0):.3f} Prob:{indicators.get('probability', 0):.2f}"
        else:
            profit_pct = (profit / (shares * price - profit)) * 100 if shares and price and (shares * price - profit) != 0 else 0
            msg = f"🔴 SELL | {timestamp} | {shares:>6.0f} shares @ Rp{price:>8.0f} | Profit: Rp{profit:>8,.0f} ({profit_pct:>5.1f}%) | {reason} | Balance: Rp{balance:>12,.0f}"
        
        print(msg)
        logging.info(msg)

    def get_trade_summary(self):
        if not self.trades:
            summary = "No trades executed"
            logging.info(summary)
            return summary

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[(trades_df['type'] == 'sell') & (trades_df['profit'].notnull())]

        # Financial Metrics
        initial_balance = trades_df['balance'].iloc[0] if not trades_df.empty else 0
        final_balance = trades_df['balance'].iloc[-1] if not trades_df.empty else 0
        total_return = ((final_balance - initial_balance) / initial_balance * 100) if initial_balance != 0 else 0
        total_profit = sell_trades['profit'].sum() if not sell_trades.empty else 0
        total_trades = len(sell_trades)
        profitable_trades = len(sell_trades[sell_trades['profit'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        # Risk Metrics
        balance_series = pd.Series(trades_df['balance'])
        returns = balance_series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(len(returns)) * 100 if len(returns) > 0 else 0
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0
        max_drawdown = ((balance_series.cummax() - balance_series) / balance_series.cummax()).max() * 100 if not balance_series.empty else 0
        risk_reward_ratio = (sell_trades[sell_trades['profit'] > 0]['profit'].mean() / 
                            abs(sell_trades[sell_trades['profit'] < 0]['profit'].mean())) if len(sell_trades[sell_trades['profit'] < 0]) > 0 else float('inf')

        # Trade Analysis
        sell_reasons = sell_trades.groupby('reason').agg({'profit': ['count', 'mean', 'sum']}).reset_index()
        sell_reasons.columns = ['reason', 'count', 'avg_profit', 'total_profit']

        # Calculate trade durations
        trade_durations = []
        for i in range(0, len(trades_df)-1, 2):
            if trades_df.iloc[i]['type'] == 'buy' and i+1 < len(trades_df) and trades_df.iloc[i+1]['type'] == 'sell':
                duration = (trades_df.iloc[i+1]['timestamp'] - trades_df.iloc[i]['timestamp']).total_seconds() / 60
                trade_durations.append(duration)
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        exposure_time = (sum(trade_durations) / ((trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds() / 60)) * 100 if trade_durations else 0

        # Trade Frequency
        trade_days = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days + 1
        trades_per_day = total_trades / trade_days if trade_days > 0 else 0

        # Model Performance Metrics
        model_summary = ""
        if self.model_metrics.get('y_true') is not None and self.model_metrics.get('y_pred') is not None:
            y_true = self.model_metrics['y_true']
            y_pred = self.model_metrics['y_pred']
            model_summary += "\nModel Performance Metrics:\n" + "="*40 + "\n"
            model_summary += classification_report(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            model_summary += f"\nConfusion Matrix:\n{cm}\n"
            model_summary += f"True Positives: {cm[1,1]}\nFalse Positives: {cm[0,1]}\nTrue Negatives: {cm[0,0]}\nFalse Negatives: {cm[1,0]}\n"

            # Probability calibration check (basic)
            if self.model_metrics.get('y_pred_proba') is not None:
                prob_bins = np.histogram(self.model_metrics['y_pred_proba'], bins=10, range=(0,1))[0]
                model_summary += f"\nProbability Distribution (10 bins):\n{prob_bins}\n"

        # Feature Importance
        feature_summary = ""
        if self.feature_importance is not None:
            feature_summary += "\nTop 5 Feature Importance:\n" + "="*40 + "\n"
            for i, (feat, imp) in enumerate(self.feature_importance.head(5).itertuples(index=False), 1):
                feature_summary += f"{i}. {feat}: {imp:.4f}\n"

        # Format Summary
        summary = f"""
═══════════════════════════════════════════════════════════════
                         TRADING SUMMARY
═══════════════════════════════════════════════════════════════
📅 Period: {trades_df['timestamp'].min().strftime('%Y-%m-%d')} to {trades_df['timestamp'].max().strftime('%Y-%m-%d')}
💰 Financial Metrics:
   Initial Balance: Rp {initial_balance:,.0f}
   Final Balance: Rp {final_balance:,.0f}
   Total Return: {total_return:.2f}%
   Total Profit: Rp {total_profit:,.0f}
   Average Profit/Trade: Rp {avg_profit:,.0f}
   Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})
   Max Drawdown: {max_drawdown:.2f}%

📊 Risk Metrics:
   Volatility: {volatility:.2f}%
   Sharpe Ratio: {sharpe_ratio:.2f}
   Risk-Reward Ratio: {risk_reward_ratio:.2f}

📈 Trade Analysis:
   Total Trades: {total_trades}
   Buy Trades: {len(buy_trades)}
   Sell Trades: {len(sell_trades)}
   Trades/Day: {trades_per_day:.2f}
   Average Trade Duration: {avg_trade_duration:.2f} minutes
   Market Exposure: {exposure_time:.2f}% of time
   Sell Reasons:
"""
        for _, row in sell_reasons.iterrows():
            summary += f"     {row['reason']}: {int(row['count'])} trades, Avg Profit: Rp {row['avg_profit']:,.0f}, Total: Rp {row['total_profit']:,.0f}\n"

        summary += model_summary + feature_summary

        summary += """
📊 Visualizations Available:
   - Portfolio Value: trading_performance.png (Subplot 1)
   - Return Distribution: trading_performance.png (Subplot 2)
   - Cumulative Returns: trading_performance.png (Subplot 3)
   - Drawdown: trading_performance.png (Subplot 4)

📝 Notes:
   - Check 'detailed_trading_log.txt' for full trade logs.
   - Model performance metrics assume test set predictions were provided.
   - Probability calibration may indicate over/under-confidence if bins are uneven.
═══════════════════════════════════════
"""
        # Generate additional visualization: Confusion Matrix
        if self.model_metrics.get('y_true') is not None and self.model_metrics.get('y_pred') is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(self.model_metrics['y_true'], self.model_metrics['y_pred']),
                       annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

        logging.info(summary)
        return summary
```