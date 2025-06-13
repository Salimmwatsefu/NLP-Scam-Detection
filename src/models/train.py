import os
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pickle
import json
from datetime import datetime
import yaml
import joblib
from pathlib import Path
from torch.utils.data import Dataset

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "models"

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def get_output_filename(base_dir, base_name, config):
    tag = config["run"]["tag"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") if config["run"]["timestamp"] else ""
    filename = f"{base_name}_{tag}{'' if not timestamp else '_' + timestamp}"
    return os.path.join(base_dir, filename)

def save_metrics(metrics, model_name, config):
    metrics_path = get_output_filename(config["output"]["metrics_path"], f"{model_name}_metrics", config) + ".json"
    os.makedirs(config["output"]["metrics_path"], exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"{model_name} metrics saved to {metrics_path}")

def train_logistic_regression(X_train, y_train, X_test, y_test, config):
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        C=config["model"]["hyperparams"]["C"],
        max_iter=config["model"]["hyperparams"]["max_iter"],
        multi_class="multinomial",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    classes = np.array([0, 1, 2])  # legit, moderate_scam, high_scam
    # Training metrics
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, y_pred_train, average="weighted")
    train_per_class = precision_recall_fscore_support(y_train, y_pred_train, average=None, labels=classes)
    logger.info(f"Logistic Regression - Train Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    logger.info(f"Logistic Regression - Per-class Precision: {train_per_class[0]}, Recall: {train_per_class[1]}, F1: {train_per_class[2]}")
    
    # Test metrics
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted")
    test_per_class = precision_recall_fscore_support(y_test, y_pred_test, average=None, labels=classes)
    logger.info(f"Logistic Regression - Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    logger.info(f"Logistic Regression - Per-class Precision: {test_per_class[0]}, Recall: {test_per_class[1]}, F1: {test_per_class[2]}")
    
    # Save metrics
    metrics = {
        "train": {
            "accuracy": train_accuracy,
            "precision_weighted": train_precision,
            "recall_weighted": train_recall,
            "f1_weighted": train_f1,
            "per_class": {
                "legit": {"precision": train_per_class[0][0], "recall": train_per_class[1][0], "f1": train_per_class[2][0]},
                "moderate_scam": {"precision": train_per_class[0][1], "recall": train_per_class[1][1], "f1": train_per_class[2][1]},
                "high_scam": {"precision": train_per_class[0][2], "recall": train_per_class[1][2], "f1": train_per_class[2][2]}
            }
        },
        "test": {
            "accuracy": test_accuracy,
            "precision_weighted": test_precision,
            "recall_weighted": test_recall,
            "f1_weighted": test_f1,
            "per_class": {
                "legit": {"precision": test_per_class[0][0], "recall": test_per_class[1][0], "f1": test_per_class[2][0]},
                "moderate_scam": {"precision": test_per_class[0][1], "recall": test_per_class[1][1], "f1": test_per_class[2][1]},
                "high_scam": {"precision": test_per_class[0][2], "recall": test_per_class[1][2], "f1": test_per_class[2][2]}
            }
        }
    }
    save_metrics(metrics, "logistic", config)
    
    model_path = get_output_filename(config["model"]["save_path"], "logistic", config) + ".pkl"
    os.makedirs(config["model"]["save_path"], exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Logistic Regression model saved to {model_path}")
    return model

def train_xgboost(X_train, y_train, X_test, y_test, config):
    logger.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=config["model"]["hyperparams"].get("n_estimators", 100),
        learning_rate=config["model"]["hyperparams"].get("learning_rate", 0.1),
        max_depth=config["model"]["hyperparams"].get("max_depth", 6),
        objective="multi:softmax",
        num_class=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    classes = np.array([0, 1, 2])
    # Training metrics
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_train, y_pred_train, average="weighted")
    train_per_class = precision_recall_fscore_support(y_train, y_pred_train, average=None, labels=classes)
    logger.info(f"XGBoost - Train Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    logger.info(f"XGBoost - Per-class Precision: {train_per_class[0]}, Recall: {train_per_class[1]}, F1: {train_per_class[2]}")
    
    # Test metrics
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average="weighted")
    test_per_class = precision_recall_fscore_support(y_test, y_pred_test, average=None, labels=classes)
    logger.info(f"XGBoost - Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    logger.info(f"XGBoost - Per-class Precision: {test_per_class[0]}, Recall: {test_per_class[1]}, F1: {test_per_class[2]}")
    
    # Save metrics
    metrics = {
        "train": {
            "accuracy": train_accuracy,
            "precision_weighted": train_precision,
            "recall_weighted": train_recall,
            "f1_weighted": train_f1,
            "per_class": {
                "legit": {"precision": train_per_class[0][0], "recall": train_per_class[1][0], "f1": train_per_class[2][0]},
                "moderate_scam": {"precision": train_per_class[0][1], "recall": train_per_class[1][1], "f1": train_per_class[2][1]},
                "high_scam": {"precision": train_per_class[0][2], "recall": train_per_class[1][2], "f1": train_per_class[2][2]}
            }
        },
        "test": {
            "accuracy": test_accuracy,
            "precision_weighted": test_precision,
            "recall_weighted": test_recall,
            "f1_weighted": test_f1,
            "per_class": {
                "legit": {"precision": test_per_class[0][0], "recall": test_per_class[1][0], "f1": test_per_class[2][0]},
                "moderate_scam": {"precision": test_per_class[0][1], "recall": test_per_class[1][1], "f1": test_per_class[2][1]},
                "high_scam": {"precision": test_per_class[0][2], "recall": test_per_class[1][2], "f1": test_per_class[2][2]}
            }
        }
    }
    save_metrics(metrics, "xgboost", config)
    
    model_path = get_output_filename(config["model"]["save_path"], "xgboost", config) + ".pkl"
    os.makedirs(config["model"]["save_path"], exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"XGBoost model saved to {model_path}")
    return model

def train_bert(config):
    logger.info("Training BERT...")
    bert_train_path = get_output_filename(config["features"]["bert_features_dir"], "bert_features", config) + ".pt"
    bert_test_path = get_output_filename(config["features"]["bert_features_dir"], "test_bert_features", config) + ".pt"
    
    if not os.path.exists(bert_train_path) or not os.path.exists(bert_test_path):
        logger.error(f"BERT feature files not found: {bert_train_path}, {bert_test_path}")
        raise FileNotFoundError("BERT feature files not found")
    
    bert_train_data = torch.load(bert_train_path)
    bert_test_data = torch.load(bert_test_path)
    
    class ScamDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]
            }

    train_dataset = ScamDataset(
        bert_train_data["input_ids"],
        bert_train_data["attention_mask"],
        bert_train_data["labels"]
    )
    test_dataset = ScamDataset(
        bert_test_data["input_ids"],
        bert_test_data["attention_mask"],
        bert_test_data["labels"]
    )
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    training_args = TrainingArguments(
        output_dir=config["model"]["save_path"],
        num_train_epochs=config["model"]["hyperparams"].get("epochs", 3),
        per_device_train_batch_size=config["model"]["hyperparams"].get("batch_size", 8),
        per_device_eval_batch_size=config["model"]["hyperparams"].get("batch_size", 8),
        eval_strategy="epoch",
        logging_dir=config["output"]["logs_path"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        per_class = precision_recall_fscore_support(labels, predictions, average=None, labels=[0, 1, 2])
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_precision": per_class[0].tolist(),
            "per_class_recall": per_class[1].tolist(),
            "per_class_f1": per_class[2].tolist()
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    # Get test metrics
    eval_results = trainer.evaluate()
    logger.info(f"BERT - Test Accuracy: {eval_results['eval_accuracy']:.4f}, Precision: {eval_results['eval_precision']:.4f}, Recall: {eval_results['eval_recall']:.4f}, F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"BERT - Per-class Precision: {eval_results['per_class_precision']}, Recall: {eval_results['per_class_recall']}, F1: {eval_results['per_class_f1']}")
    
    # Save metrics
    metrics = {
        "train": {},  # Trainer doesn't provide train metrics by default
        "test": {
            "accuracy": eval_results["eval_accuracy"],
            "precision_weighted": eval_results["eval_precision"],
            "recall_weighted": eval_results["eval_recall"],
            "f1_weighted": eval_results["eval_f1"],
            "per_class": {
                "legit": {"precision": eval_results["per_class_precision"][0], "recall": eval_results["per_class_recall"][0], "f1": eval_results["per_class_f1"][0]},
                "moderate_scam": {"precision": eval_results["per_class_precision"][1], "recall": eval_results["per_class_recall"][1], "f1": eval_results["per_class_f1"][1]},
                "high_scam": {"precision": eval_results["per_class_precision"][2], "recall": eval_results["per_class_recall"][2], "f1": eval_results["per_class_f1"][2]}
            }
        }
    }
    save_metrics(metrics, "bert", config)
    
    model_path = get_output_filename(config["model"]["save_path"], "bert", config)
    os.makedirs(config["model"]["save_path"], exist_ok=True)
    model.save_pretrained(model_path)
    logger.info(f"BERT model saved to {model_path}")
    return model

def run_training(config):
    logger.info("Starting model training...")
    tfidf_train_path = get_output_filename(config["features"]["tfidf_features_dir"], "tfidf_features", config) + ".npy"
    tfidf_test_path = get_output_filename(config["features"]["tfidf_features_dir"], "test_tfidf_features", config) + ".npy"
    labels_train_path = get_output_filename(config["features"]["labels_dir"], "labels", config) + ".npy"
    labels_test_path = get_output_filename(config["features"]["labels_dir"], "test_labels", config) + ".npy"
    
    for path in [tfidf_train_path, tfidf_test_path, labels_train_path, labels_test_path]:
        if not os.path.exists(path):
            logger.error(f"Feature or label file {path} not found")
            raise FileNotFoundError(f"Feature or label file {path} not found")
    
    X_train = np.load(tfidf_train_path)
    X_test = np.load(tfidf_test_path)
    y_train = np.load(labels_train_path)
    y_test = np.load(labels_test_path)
    logger.info(f"Loaded TF-IDF train features: {X_train.shape}, test: {X_test.shape}")
    logger.info(f"Loaded labels train: {y_train.shape}, test: {y_test.shape}")

    models = {}
    model_type = config["model"]["type"]
    
    if model_type in ["logistic_regression", "all"]:
        models["logistic"] = train_logistic_regression(X_train, y_train, X_test, y_test, config)
    if model_type in ["xgboost", "all"]:
        models["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test, config)
    if model_type in ["bert", "all"]:
        models["bert"] = train_bert(config)
    logger.info("✅ Training completed.")
    return models

def train_model(X, y, model_type="XGBoost"):
    """
    Train a model on the given data.
    
    Args:
        X: Training features
        y: Training labels
        model_type (str): Type of model to train ('XGBoost', 'Logistic Regression', or 'BERT')
        
    Returns:
        The trained model
    """
    try:
        if model_type == "XGBoost":
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == "Logistic Regression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Train the model
        model.fit(X, y)
        
        # Save the model
        model_path = MODEL_DIR / f"{model_type.lower()}_model.joblib"
        joblib.dump(model, model_path)
        
        logger.info(f"{model_type} model trained and saved successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    run_training(config)