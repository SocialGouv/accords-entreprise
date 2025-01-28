#!/usr/bin/env python
"""
- Mixed precision training (FP16)
- 8-bit Adam optimizer for memory efficiency
- Gradient checkpointing
- Gradient accumulation
- Cosine learning rate scheduling with warmup
- Efficient data loading with proper batching
- Mean pooling for better sentence embeddings
- Contrastive learning with temperature scaling
- Automatic checkpointing and model saving
- Progress tracking with Weights & Biases
"""

import json
import logging
import os
import platform
from dataclasses import dataclass
from typing import Dict, List

import torch
import yaml
from bitsandbytes.optim import Adam8bit
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.functional import cosine_similarity, normalize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingExample:
    anchor: str
    positive: str
    negative: str


class ContrastiveDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = [TrainingExample(**ex) for ex in examples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize all three texts
        anchor_encoding = self.tokenizer(
            example.anchor,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        positive_encoding = self.tokenizer(
            example.positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        negative_encoding = self.tokenizer(
            example.negative,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_encoding["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
        }


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # Use mean pooling instead of CLS token
        embeddings = self.mean_pooling(outputs, attention_mask)
        # Normalize embeddings
        normalized_embeddings = normalize(embeddings, p=2, dim=1)
        return normalized_embeddings


def contrastive_loss(
    anchor_embeddings, positive_embeddings, negative_embeddings, temperature=0.05
):
    # Compute similarities
    pos_similarities = cosine_similarity(anchor_embeddings, positive_embeddings)
    neg_similarities = cosine_similarity(anchor_embeddings, negative_embeddings)

    # Scale similarities by temperature
    pos_similarities = pos_similarities / temperature
    neg_similarities = neg_similarities / temperature

    # Compute loss
    losses = -pos_similarities + torch.log(
        torch.exp(pos_similarities) + torch.exp(neg_similarities)
    )
    return losses.mean()


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Determine the best available device
        if platform.processor() == "arm" and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")  # Fallback to CPU

        logging.info(f"Using device: {self.device}")
        self.setup_model()
        self.setup_wandb()

    def setup_model(self):
        logging.info(f"Loading model {self.config['model']['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])
        self.model = EmbeddingModel(
            self.config["model"]["name"], self.config["model"]["max_length"]
        )

        # Enable gradient checkpointing if configured
        if self.config["optimization"]["use_gradient_checkpointing"]:
            self.model.model.gradient_checkpointing_enable()

        self.model.to(self.device)

    def setup_wandb(self):
        wandb.init(project=self.config["logging"]["wandb_project"], config=self.config)

    def save_checkpoint(self, step: int):
        output_dir = os.path.join(
            self.config["output"]["model_dir"], f"checkpoint-{step}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logging.info(f"Saved checkpoint to {output_dir}")

    def train(self, train_data_path: str):
        # Load and prepare dataset
        logging.info(f"Loading training data from {train_data_path}")
        with open(train_data_path) as f:
            train_data = json.load(f)

        train_dataset = ContrastiveDataset(
            train_data, self.tokenizer, self.config["model"]["max_length"]
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

        # Optimizer
        if self.config["optimization"]["use_8bit_adam"]:
            optimizer = Adam8bit(
                self.model.parameters(),
                lr=float(self.config["training"]["learning_rate"]),
                weight_decay=self.config["training"]["weight_decay"],
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config["training"]["learning_rate"]),
                weight_decay=self.config["training"]["weight_decay"],
            )

        # Learning rate scheduler
        num_training_steps = (
            len(train_dataloader) * self.config["training"]["num_epochs"]
        )
        warmup_steps = int(num_training_steps * self.config["training"]["warmup_ratio"])

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision training - only supported on CUDA
        scaler = (
            GradScaler()
            if self.config["training"]["fp16"] and self.device.type == "cuda"
            else None
        )

        # Training loop
        logging.info("Starting training")
        global_step = 0
        for epoch in range(self.config["training"]["num_epochs"]):
            self.model.train()
            epoch_loss: float = 0.0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Mixed precision training
                with autocast(device_type=self.device.type, enabled=bool(scaler)):
                    # Get embeddings for all three texts
                    anchor_embeddings = self.model(
                        batch["anchor_input_ids"], batch["anchor_attention_mask"]
                    )
                    positive_embeddings = self.model(
                        batch["positive_input_ids"], batch["positive_attention_mask"]
                    )
                    negative_embeddings = self.model(
                        batch["negative_input_ids"], batch["negative_attention_mask"]
                    )

                    # Compute loss
                    loss = contrastive_loss(
                        anchor_embeddings, positive_embeddings, negative_embeddings
                    )

                # Gradient accumulation
                loss = loss / self.config["training"]["gradient_accumulation_steps"]

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (step + 1) % self.config["training"][
                        "gradient_accumulation_steps"
                    ] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["optimization"]["max_grad_norm"],
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step + 1) % self.config["training"][
                        "gradient_accumulation_steps"
                    ] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["optimization"]["max_grad_norm"],
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                epoch_loss = float(epoch_loss + loss.item())

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": epoch_loss / (step + 1), "lr": scheduler.get_last_lr()[0]}
                )

                # Log to wandb
                if global_step % self.config["logging"]["log_steps"] == 0:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step,
                        }
                    )

                # Save checkpoint
                if global_step % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(global_step)

                global_step += 1

            # Log epoch metrics
            epoch_loss = epoch_loss / len(train_dataloader)
            logging.info(f"Epoch {epoch + 1} - Average loss: {epoch_loss:.4f}")
            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                    "epoch": epoch,
                }
            )

        # Save final model
        final_output_dir = os.path.join(self.config["output"]["model_dir"], "final")
        self.model.model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        logging.info(f"Saved final model to {final_output_dir}")


def main():
    trainer = Trainer("config/training_config.yaml")
    trainer.train(
        "fine_tuning_data/contrastive_fine_tuning/sample_training_data_en.json"
    )


if __name__ == "__main__":
    main()
