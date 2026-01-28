import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_actual_vs_predicted_plots(y_true, y_pred, model_name, plots_dir):
    try:
        if model_name == 'emissions':
            target_columns = [
                'co2_emission_mg_s', 'co_emission_mg_s', 'nox_emission_mg_s',
                'pmx_emission_mg_s', 'fuel_consumption_mg_s'
            ]
        elif model_name == 'traffic':
            target_columns = [
                'vehicle_count', 'mean_speed_ms', 'occupancy_percent',
                'waiting_time_seconds', 'halting_vehicles'
            ]
        else:
            target_columns = ['metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5']
        
        fig, axes = plt.subplots(len(target_columns), 1, figsize=(15, 4*len(target_columns)))
        if len(target_columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(target_columns):
            ax = axes[i]
            
            actual_values = y_true[:, i::len(target_columns)].flatten()
            predicted_values = y_pred[:, i::len(target_columns)].flatten()
            
            min_length = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_length]
            predicted_values = predicted_values[:min_length]
            
            max_plot_steps = 200
            if min_length > max_plot_steps:
                start_idx = (min_length - max_plot_steps) // 2
                end_idx = start_idx + max_plot_steps
                actual_values = actual_values[start_idx:end_idx]
                predicted_values = predicted_values[start_idx:end_idx]
                min_length = max_plot_steps
            
            time_steps = range(min_length)
            
            ax.plot(time_steps, actual_values, color='blue', linewidth=1.5, 
                   label='Actual', alpha=0.8)
            ax.plot(time_steps, predicted_values, color='orange', linewidth=1.5, 
                   label='Predicted', alpha=0.8, linestyle='--')
            
            ax.set_title(f'{model_name.title()} - {col} (200 timesteps)', fontsize=12)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.set_xlim(0, min_length)
            
            if len(y_true[:, i::len(target_columns)].flatten()) > max_plot_steps:
                ax.text(0.02, 0.98, f'Showing 200/988 timesteps\n(zoomed view)', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{model_name.title()} Model: Actual vs Predicted\n'
                     f'Blue = Actual, Orange = Predicted', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{model_name}_actual_vs_predicted.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Actual vs predicted time series plots saved to: {plot_path}")
        
    except Exception as e:
        logger.error(f"Failed to create actual vs predicted plot for {model_name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def evaluate_predictions(y_true, y_pred, model_name, attack_tag):
    try:
        yt = y_true.reshape(-1)
        yp = y_pred.reshape(-1)
        min_len = min(len(yt), len(yp))
        yt = yt[:min_len]
        yp = yp[:min_len]
        mse = float(np.mean((yt - yp) ** 2))
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(mse))
        eps = 1e-8
        mape = float(np.mean(np.abs((yt - yp) / (np.maximum(np.abs(yt), eps)))))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2) + eps)
        r2 = 1.0 - ss_res / ss_tot
        return {
            'model': model_name,
            'attack': attack_tag,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}/{attack_tag}: {e}")
        return {'model': model_name, 'attack': attack_tag, 'mse': None, 'rmse': None, 'mae': None, 'mape': None, 'r2': None}


def save_evaluation_results(eval_rows):
    if not eval_rows:
        return
    
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_eval = pd.DataFrame(eval_rows)
        os.makedirs('logs', exist_ok=True)
        csv_path = os.path.join('logs', f'attack_eval_{ts}.csv')
        json_path = os.path.join('logs', f'attack_eval_{ts}.json')
        df_eval.to_csv(csv_path, index=False)
        df_eval.to_json(json_path, orient='records', indent=2)
        logger.info(f"Evaluation saved: {csv_path}, {json_path}")
        
        print("\n" + "="*60)
        print("ATTACK PERFORMANCE COMPARISON")
        print("="*60)
        for model in df_eval['model'].unique():
            model_data = df_eval[df_eval['model'] == model]
            print(f"\n{model.upper()} MODEL:")
            print("-" * 40)
            for _, row in model_data.iterrows():
                attack_tag = row['attack']
                print(f"{attack_tag:>8}: MSE={row['mse']:.4f}, RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R²={row['r2']:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")


def generate_performance_summary(predictors):
    logger.info("Generating performance summary...")
    
    trained_count = 0
    trained_models = []
    
    for name, predictor in predictors.items():
        if hasattr(predictor, 'is_model_trained') and predictor.is_model_trained(name):
            trained_count += 1
            trained_models.append(name)
    
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Trained models: {trained_count}")
    print(f"Trained Models:")
    for model in trained_models:
        print(f"    {model}")
    print("=" * 60)
