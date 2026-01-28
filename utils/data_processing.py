import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def normalize_training_data(edge_metrics_history):
    try:
        if len(edge_metrics_history) == 0:
            logger.warning("No data to normalize")
            return edge_metrics_history, {}
        
        df = pd.DataFrame(edge_metrics_history)
        logger.info(f"Normalizing data: {df.shape}")
        
        exclude_columns = ['step', 'timestamp', 'edge_id']
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
        
        df_normalized = df.copy()
        
        feature_scalers = {}
        
        traffic_columns = ['vehicle_count', 'mean_speed_ms', 'occupancy_percent', 
                         'waiting_time_seconds', 'halting_vehicles']
        emissions_columns = ['co2_emission_mg_s', 'co_emission_mg_s', 'nox_emission_mg_s',
                           'pmx_emission_mg_s', 'fuel_consumption_mg_s']
        
        if all(col in df.columns for col in traffic_columns):
            traffic_scaler = MinMaxScaler()
            df_normalized[traffic_columns] = traffic_scaler.fit_transform(df[traffic_columns])
            feature_scalers['traffic'] = traffic_scaler
        
        if all(col in df.columns for col in emissions_columns):
            emissions_scaler = MinMaxScaler()
            df_normalized[emissions_columns] = emissions_scaler.fit_transform(df[emissions_columns])
            feature_scalers['emissions'] = emissions_scaler
        
        other_numeric = [col for col in columns_to_normalize 
                       if col not in traffic_columns and col not in emissions_columns]
        if other_numeric:
            other_scaler = MinMaxScaler()
            df_normalized[other_numeric] = other_scaler.fit_transform(df[other_numeric])
            feature_scalers['other'] = other_scaler
        
        normalized_data = df_normalized.to_dict('records')
        
        logger.info("Normalization complete")
        
        df_check = pd.DataFrame(normalized_data)
        for col in columns_to_normalize:
            if col in df_check.columns:
                min_val = df_check[col].min()
                max_val = df_check[col].max()
                if min_val < 0 or max_val > 1:
                    logger.warning(f"Column {col} not properly normalized: range [{min_val:.3f}, {max_val:.3f}]")
                else:
                    logger.debug(f"Column {col} properly normalized: range [{min_val:.3f}, {max_val:.3f}]")
        
        return normalized_data, feature_scalers
        
    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        logger.warning("Continuing without data normalization")
        return edge_metrics_history, {}


def convert_predictions_to_edge_format(predictions, column_names, edge_metrics_history):
    edge_predictions = {}
    
    try:
        if len(edge_metrics_history) > 0:
            latest_metrics = edge_metrics_history[-1]
        
        if len(edge_metrics_history) > 0:
            monitoring_edges = list(set([item.get('edge_id', 'unknown') for item in edge_metrics_history if 'edge_id' in item]))
        else:
            monitoring_edges = ['default_edge']
        
        for edge_id in monitoring_edges:
            edge_predictions[edge_id] = {}
            
            for i, col_name in enumerate(column_names):
                if i < len(predictions):
                    edge_predictions[edge_id][col_name] = predictions[i]
                else:
                    edge_predictions[edge_id][col_name] = latest_metrics.get(col_name, 0)
        
        return edge_predictions
        
    except Exception as e:
        logger.error(f"Failed to convert predictions to edge format: {e}")
        return {}
