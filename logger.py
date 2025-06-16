import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

class TradingLogger:
    def __init__(self):
        self.trades = []
        self.model_metrics = {
            'y_true': None,
            'y_pred': None,
            'y_pred_proba': None,
            'feature_importance': None
        }
        self.start_time = None
        self.end_time = None
        logging.info("TradingLogger initialized")

    def log_trade(self, action, price, shares, profit=None, reason=None, timestamp=None, balance=None, indicators=None, buy_fee=0, sell_fee=0):
        try:
            trade = {
                'action': action,
                'price': price,
                'shares': shares,
                'profit': profit,
                'reason': reason,
                'timestamp': timestamp or datetime.now(),
                'balance': balance,
                'indicators': indicators or {},
                'buy_fee': buy_fee,
                'sell_fee': sell_fee
            }
            self.trades.append(trade)
            log_message = f"Trade: {action.upper()} | Price: Rp {price:,.2f} | Shares: {shares:,} | "
            if profit is not None:
                log_message += f"Profit: Rp {profit:,.2f} | "
            if reason:
                log_message += f"Reason: {reason} | "
            log_message += f"Balance: Rp {balance:,.2f} | Buy Fee: Rp {buy_fee:,.2f} | Sell Fee: Rp {sell_fee:,.2f} | Indicators: {indicators}"
            logging.info(log_message)
        except Exception as e:
            logging.error(f"Error logging trade: {e}")

    def set_model_metrics(self, y_true, y_pred, y_pred_proba, feature_importance):
        try:
            self.model_metrics = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_importance': feature_importance
            }
            logging.info("Model metrics set successfully")
        except Exception as e:
            logging.error(f"Error setting model metrics: {e}")

    def export_trades_to_csv(self, filename='trades.csv'):
        try:
            if not self.trades:
                logging.info("No trades to export")
                return
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(filename, index=False)
            logging.info(f"Trades exported to {filename}")
        except Exception as e:
            logging.error(f"Error exporting trades to CSV: {e}")

    def get_trade_summary(self):
        try:
            if not self.trades:
                summary = "No trades recorded."
                logging.info(summary)
                return summary

            trades_df = pd.DataFrame(self.trades)
            initial_balance = trades_df['balance'].iloc[0] if 'balance' in trades_df else 100000000
            final_balance = trades_df['balance'].iloc[-1] if 'balance' in trades_df else initial_balance
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            total_profit = trades_df[trades_df['profit'].notnull()]['profit'].sum() if 'profit' in trades_df else 0
            total_buy_fees = trades_df['buy_fee'].sum() if 'buy_fee' in trades_df else 0
            total_sell_fees = trades_df['sell_fee'].sum() if 'sell_fee' in trades_df else 0
            total_trades = len(trades_df[trades_df['profit'].notnull()])
            profitable_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            avg_profit = trades_df['profit'].mean() if total_trades > 0 else 0

            max_drawdown = 0
            peak = initial_balance
            for balance in trades_df['balance'].dropna():
                if balance > peak:
                    peak = balance
                drawdown = ((peak - balance) / peak) * 100
                max_drawdown = max(max_drawdown, drawdown)

            start_time = trades_df['timestamp'].min()
            end_time = trades_df['timestamp'].max()
            period_days = (end_time - start_time).days if start_time and end_time else 1
            trades_per_day = total_trades / period_days if period_days > 0 else 0

            trade_durations = []
            buy_times = trades_df[trades_df['action'] == 'buy']['timestamp']
            sell_times = trades_df[trades_df['action'] == 'sell']['timestamp']
            for buy_time, sell_time in zip(buy_times, sell_times):
                duration = (sell_time - buy_time).total_seconds() / 60
                trade_durations.append(duration)
            avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

            exposure_time = sum(trade_durations) / ((end_time - start_time).total_seconds() / 60) * 100 if trade_durations else 0

            sell_reasons = trades_df[trades_df['action'] == 'sell']['reason'].value_counts()
            sell_summary = ""
            for reason, count in sell_reasons.items():
                reason_trades = trades_df[(trades_df['action'] == 'sell') & (trades_df['reason'] == reason)]
                avg_profit_reason = reason_trades['profit'].mean() if not reason_trades.empty else 0
                total_profit_reason = reason_trades['profit'].sum() if not reason_trades.empty else 0
                total_buy_fees_reason = reason_trades['buy_fee'].sum() if not reason_trades.empty else 0
                total_sell_fees_reason = reason_trades['sell_fee'].sum() if not reason_trades.empty else 0
                sell_summary += (f"     {reason}: {count} trades, Avg Profit: Rp {avg_profit_reason:,.0f}, "
                                 f"Total: Rp {total_profit_reason:,.0f}, Buy Fees: Rp {total_buy_fees_reason:,.0f}, "
                                 f"Sell Fees: Rp {total_sell_fees_reason:,.0f}\n")

            balance_series = trades_df['balance'].dropna()
            returns = balance_series.pct_change().dropna()
            volatility = returns.std() * np.sqrt(len(returns)) * 100 if len(returns) > 0 else 0
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0
            risk_reward = trades_df[trades_df['profit'] > 0]['profit'].mean() / abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf')

            model_summary = ""
            if self.model_metrics['y_true'] is not None:
                clf_report = classification_report(self.model_metrics['y_true'], self.model_metrics['y_pred'], zero_division=0, output_dict=False)
                model_summary += f"\nModel Performance Metrics:\n========================================\n{clf_report}\n"

                cm = confusion_matrix(self.model_metrics['y_true'], self.model_metrics['y_pred'])
                model_summary += f"Confusion Matrix:\n{cm}\n"
                model_summary += f"True Positives: {cm[1,1]}\nFalse Positives: {cm[0,1]}\nTrue Negatives: {cm[0,0]}\nFalse Negatives: {cm[1,0]}\n"

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig('confusion_matrix.png', dpi=300)
                plt.close()

                if self.model_metrics['y_pred_proba'] is not None:
                    prob_bins = np.histogram(self.model_metrics['y_pred_proba'], bins=10, range=(0, 1))[0]
                    model_summary += f"\nProbability Distribution (10 bins):\n{prob_bins.tolist()}\n"

            feature_summary = ""
            if self.model_metrics['feature_importance'] is not None:
                top_features = self.model_metrics['feature_importance'].head(5)
                feature_summary += "\nTop 5 Feature Importance:\n========================================\n"
                for i, row in top_features.iterrows():
                    feature_summary += f"{i+1}. {row['feature']}: {row['importance']:.4f}\n"

            summary = (
                f"{'='*63}\n"
                f"{'TRADING SUMMARY':^63}\n"
                f"{'='*63}\n"
                f"📅 Period: {start_time.strftime('%Y-%m-%d') if start_time else 'N/A'} to {end_time.strftime('%Y-%m-%d') if end_time else 'N/A'}\n"
                f"💰 Financial Metrics:\n"
                f"   Initial Balance: Rp {initial_balance:,.0f}\n"
                f"   Final Balance: Rp {final_balance:,.0f}\n"
                f"   Total Return: {total_return:.2f}%\n"
                f"   Total Profit: Rp {total_profit:,.0f}\n"
                f"   Total Buy Fees: Rp {total_buy_fees:,.0f}\n"
                f"   Total Sell Fees: Rp {total_sell_fees:,.0f}\n"
                f"   Net Profit (Profit - Fees): Rp {(total_profit - total_buy_fees - total_sell_fees):,.0f}\n"
                f"   Average Profit/Trade: Rp {avg_profit:,.0f}\n"
                f"   Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})\n"
                f"   Max Drawdown: {max_drawdown:.2f}%\n\n"
                f"📊 Risk Metrics:\n"
                f"   Volatility: {volatility:.2f}%\n"
                f"   Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"   Risk-Reward Ratio: {risk_reward:.2f}\n\n"
                f"📈 Trade Analysis:\n"
                f"   Total Trades: {total_trades}\n"
                f"   Buy Trades: {len(trades_df[trades_df['action'] == 'buy'])}\n"
                f"   Sell Trades: {len(trades_df[trades_df['action'] == 'sell'])}\n"
                f"   Trades/Day: {trades_per_day:.2f}\n"
                f"   Average Trade Duration: {avg_trade_duration:.2f} minutes\n"
                f"   Market Exposure: {exposure_time:.2f}% of time\n"
                f"   Sell Reasons:\n{sell_summary}"
                f"{model_summary}"
                f"{feature_summary}"
                f"\n📊 Visualizations Available:\n"
                f"   - Portfolio Value: trading_performance.png (Subplot 1)\n"
                f"   - Return Distribution: trading_performance.png (Subplot 2)\n"
                f"   - Cumulative Returns: trading_performance.png (Subplot 3)\n"
                f"   - Drawdown: trading_performance.png (Subplot 4)\n"
                f"   - Confusion Matrix: confusion_matrix.png\n"
                f"📝 Notes:\n"
                f"   - Check 'detailed_trading_log.txt' for full trade logs.\n"
                f"   - Model performance metrics assume test set predictions are provided.\n"
                f"   - Probability calibration may indicate uneven bins.\n"
                f"{'='*63}\n"
            )
            
            # Split summary into chunks for Telegram (max 4096 chars per message)
            max_length = 4000
            chunks = [summary[i:i+max_length] for i in range(0, len(summary), max_length)]
            for i, chunk in enumerate(chunks):
                logging.info(f"Trade Summary (Part {i+1}/{len(chunks)}):\n{chunk}")
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating trade summary: {e}")
            summary = "Error generating trade summary."
            logging.info(summary)
            return summary