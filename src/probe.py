import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import jsonlines
from datasets import load_dataset
from PIL import Image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Import at the top of the file
from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length inputs"""
    # Filter out None values from failed samples in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:  # Check if batch is empty after filtering
        return None # Return None to signal an empty batch

    # --- Text Padding (Your original logic is correct here) ---
    max_input_len = max(item['input_ids'].shape[0] for item in batch)
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        
        pad_len = max_input_len - input_ids.shape[0]
        
        if pad_len > 0:
            pad_id = 0  # Pad token ID
            input_ids_pad = torch.full((pad_len,), pad_id, dtype=input_ids.dtype)
            attention_mask_pad = torch.zeros(pad_len, dtype=attention_mask.dtype)
            
            item['input_ids'] = torch.cat([input_ids, input_ids_pad], dim=0)
            item['attention_mask'] = torch.cat([attention_mask, attention_mask_pad], dim=0)

    # --- Pixel Values Handling ---
    # DELETED: The custom pixel_values padding logic.
    # It was unnecessary because the processor ensures all images are the same size.
    # The default_collate function will now handle stacking them correctly.

    # Use the default collate function for the processed batch.
    # It will stack all tensors correctly, including pixel_values which are now (C, H, W)
    # and will become (B, C, H, W).
    return default_collate(batch)


class COCOProbeDataset(Dataset):
    """Load COCO image-text matching data from constructed JSONL file"""
    
    def __init__(
        self, 
        jsonl_path: str,
        processor,
        coco_dataset,  # Original COCO dataset for loading images
        max_samples: Optional[int] = None
    ):
        self.processor = processor
        self.coco_ds = coco_dataset
        
        print(f"Loading data from {jsonl_path}...")
        self.pairs = []
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                self.pairs.append(obj)
                if max_samples and len(self.pairs) >= max_samples:
                    break
        
        print(f"Loaded {len(self.pairs)} samples")
        
        num_pos = sum(1 for p in self.pairs if p["label"] == 1)
        print(f"  Positive samples: {num_pos}")
        print(f"  Negative samples: {len(self.pairs) - num_pos}")
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            pair = self.pairs[idx]
            img_path = pair['image']

            # 1. Load the image as a PIL object
            image = Image.open(img_path).convert('RGB')
            target_size = (336, 336)
            image = image.resize(target_size, Image.BICUBIC)

            # 2. Prepare the prompt string
            prompt = f"USER: <image>\n{pair['caption']}\nASSISTANT:"

            # 3. Return the raw data. No processing or tensors yet.
            return {
                "image": image,
                "prompt": prompt,
                "label": pair["label"],
                "image_id": pair["image_id"]
            }
        except Exception as e:
            print(f"Warning: Skipping sample {idx} at path {self.pairs[idx].get('image', 'N/A')} due to error: {e}")
            return None

# Add this new class to your script
class LlavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        # Filter out any samples that failed during loading
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Extract the raw data from the batch
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        image_ids = [item['image_id'] for item in batch]
        image_sizes = torch.tensor([[img.height, img.width] for img in images])
        
        # Use the processor to handle the entire batch at once.
        # It correctly pads text and stacks images.
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )

        # The processor output is what the model needs.
        # We just need to add our labels and metadata back in.
        inputs['label'] = labels
        inputs['image_id'] = image_ids
        inputs['image_sizes'] = image_sizes
        
        return inputs

class LinearProbe(nn.Module):
    """Linear probe for image-text matching"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vision_feat, text_feat):
        """
        Args:
            vision_feat: [batch, dim] vision features
            text_feat: [batch, dim] text features
        Returns:
            match_score: [batch] matching scores
        """
        # Multiple feature combinations
        # Convert inputs to float32
        vision_feat = vision_feat.float()
        text_feat = text_feat.float()
        
        cos_sim = nn.functional.cosine_similarity(
            vision_feat, text_feat, dim=-1
        ).unsqueeze(-1)
        element_wise = vision_feat * text_feat
        diff = torch.abs(vision_feat - text_feat)
        
        # Concatenate all features
        combined = torch.cat([
            vision_feat, 
            text_feat, 
            element_wise, 
            diff,
            cos_sim
        ], dim=-1)
        
        return self.probe(combined).squeeze(-1)


class LLaVAProbeTrainer:
    """LLaVA image-text alignment probe trainer"""
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda:0"
    ):
        self.device = device
        print(f"Loading LLaVA model: {model_name}")
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Hook-related attributes
        self.activations = {}
        self.hooks = []
        
        # Layers to analyze
        self.analysis_layers = list(range(31))
        
    def register_hooks(self):
        """Register forward hooks"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
            return hook
        
        # Hook projector
        self.hooks.append(
            self.model.model.multi_modal_projector.register_forward_hook(
                get_activation('projector')
            )
        )
        
        # Hook language model layers
        for idx in self.analysis_layers:
            self.hooks.append(
                self.model.model.language_model.layers[idx].register_forward_hook(
                    get_activation(f'layer_{idx}')
                )
            )
        
        print(f"Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    @torch.no_grad()
    def extract_features(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract multi-layer features"""
        if batch is None:
            return {}
        try:
            # Move all tensor items in the batch to the designated device
            batch_on_device = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            # The collator has already prepared the batch.
            # We can unpack it directly into the model.
            # The processor correctly includes image_sizes internally.
            _ = self.model(
                input_ids=batch_on_device['input_ids'],
                attention_mask=batch_on_device['attention_mask'],
                pixel_values=batch_on_device['pixel_values'].to(self.model.dtype),
                image_sizes=batch_on_device['image_sizes']  # This is the critical line
            )
            
            # The rest of the logic uses the input_ids and attention_mask from the batch
            return self._extract_batch_features(
                batch_on_device['input_ids'], 
                batch_on_device['attention_mask']
            )
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def _extract_single_sample_features(self, input_ids, attention_mask):
        """Extract features for a single sample"""
        features = {}
        
        # 1. Projector features
        if 'projector' in self.activations:
            vision_feat = self.activations['projector']
            features['projector_vision'] = vision_feat.mean(dim=1)
        
        # 2. Find image token positions
        image_token_index = self.model.config.image_token_index
        img_token_pos = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]
        
        if len(img_token_pos) > 0:
            start_pos = img_token_pos[0].item()
            end_pos = img_token_pos[-1].item() + 1
        else:
            # Use default length from config
            start_pos = 0
            end_pos = self.model.config.image_seq_length
        
        # 3. Text embeddings
        text_embeds = self.model.model.language_model.embed_tokens(input_ids)
        text_mask = attention_mask.clone()
        text_mask[:, start_pos:end_pos] = 0
        
        text_feat_sum = (text_embeds * text_mask.unsqueeze(-1)).sum(dim=1)
        text_length = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        features['embed_tokens'] = text_feat_sum / text_length
        
        # 4. Language model layers
        for key in self.activations:
            if key.startswith('layer_'):
                hidden_states = self.activations[key]
                
                # Vision part
                vision_part = hidden_states[:, start_pos:end_pos, :]
                features[f'{key}_vision'] = vision_part.mean(dim=1)
                
                # Text part
                text_part = hidden_states * text_mask.unsqueeze(-1)
                features[f'{key}_text'] = text_part.sum(dim=1) / text_length
        
        return features
    
    def _extract_batch_features(self, input_ids, attention_mask):
        """Extract features for a batch"""
        features = {}
        batch_size = input_ids.shape[0]
        
        # 1. Projector features
        if 'projector' in self.activations:
            vision_feat = self.activations['projector']
            
            # Reorganize shape, assuming each sample has multiple patches
            # Determine the number of patches per sample
            patches_per_sample = vision_feat.shape[0] // batch_size
            
            # Reshape to [batch_size, patches_per_sample, seq_len, hidden_dim]
            vision_feat = vision_feat.reshape(batch_size, patches_per_sample, vision_feat.shape[1], vision_feat.shape[2])
            
            # Average all patches for each sample, resulting in [batch_size, seq_len, hidden_dim]
            vision_feat = vision_feat.mean(dim=1)
            
            features['projector_vision'] = vision_feat.mean(dim=1)
        
        # 2. Image token positions
        image_token_index = self.model.config.image_token_index
        
        # 3. Text embeddings
        text_embeds = self.model.model.language_model.embed_tokens(input_ids)
        text_mask = attention_mask.clone()
        
        # Find image token positions for each sample
        for b in range(batch_size):
            img_token_pos = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
            if len(img_token_pos) > 0:
                start_pos = img_token_pos[0].item()
                end_pos = img_token_pos[-1].item() + 1
                text_mask[b, start_pos:end_pos] = 0
        
        text_feat_sum = (text_embeds * text_mask.unsqueeze(-1)).sum(dim=1)
        text_length = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        features['embed_tokens'] = text_feat_sum / text_length
        
        # 4. Language model layers
        for key in self.activations:
            if key.startswith('layer_'):
                hidden_states = self.activations[key]
                
                vision_feats = []
                text_feats = []
                
                for b in range(batch_size):
                    img_token_pos = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
                    
                    if len(img_token_pos) > 0:
                        start_pos = img_token_pos[0].item()
                        end_pos = img_token_pos[-1].item() + 1
                        
                        vision_part = hidden_states[b, start_pos:end_pos, :]
                        vision_feats.append(vision_part.mean(dim=0))
                        
                        text_part = hidden_states[b] * text_mask[b].unsqueeze(-1)
                        text_feats.append(text_part.sum(dim=0) / text_mask[b].sum().clamp(min=1))
                    else:
                        # Fallback
                        default_len = self.model.config.image_seq_length
                        vision_feats.append(hidden_states[b, :default_len, :].mean(dim=0))
                        text_feats.append(hidden_states[b, default_len:, :].mean(dim=0))
                
                features[f'{key}_vision'] = torch.stack(vision_feats)
                features[f'{key}_text'] = torch.stack(text_feats)
        
        return features
    
    def train_probes(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        save_dir: str = "probe_checkpoints"
    ):
        """Train probes for all layers"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect sample features to determine dimensions
        print("Collecting sample features to determine dimensions...")
        sample_batch = next(iter(train_loader))
        sample_features = self.extract_features(sample_batch)
        
        # Create probes
        probes = {}
        optimizers = {}
        schedulers = {}
        
        # Projector probe
        proj_dim = sample_features['projector_vision'].shape[-1]
        embed_dim = sample_features['embed_tokens'].shape[-1]
        input_dim = proj_dim + embed_dim + proj_dim + proj_dim + 1  # vision+text+element+diff+cos
        
        probes['projector'] = LinearProbe(input_dim).to(self.device)
        optimizers['projector'] = torch.optim.AdamW(probes['projector'].parameters(), lr=lr)
        schedulers['projector'] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers['projector'], T_max=epochs
        )
        
        # Language model layer probes
        for key in sample_features:
            if '_text' in key:
                layer_key = key.replace('_text', '')
                if f'{layer_key}_vision' in sample_features:
                    feat_dim = sample_features[key].shape[-1]
                    input_dim = feat_dim * 4 + 1
                    
                    probes[layer_key] = LinearProbe(input_dim).to(self.device)
                    optimizers[layer_key] = torch.optim.AdamW(
                        probes[layer_key].parameters(), lr=lr
                    )
                    schedulers[layer_key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizers[layer_key], T_max=epochs
                    )
        
        print(f"\nCreated {len(probes)} probes")
        
        criterion = nn.BCELoss()
        best_metrics = {name: 0.0 for name in probes}
        results = {
            name: {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []} 
            for name in probes
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Training
            train_losses = self._train_epoch(
                probes, optimizers, criterion, train_loader
            )
            
            # Validation
            val_metrics = self._validate(probes, val_loader)
            
            # Update learning rate
            for scheduler in schedulers.values():
                scheduler.step()
            
            # Record results
            for name in probes:
                results[name]['train_loss'].append(train_losses[name])
                results[name]['val_acc'].append(val_metrics[name]['accuracy'])
                results[name]['val_f1'].append(val_metrics[name]['f1'])
                results[name]['val_auc'].append(val_metrics[name]['auc'])
                
                # Save best model
                if val_metrics[name]['f1'] > best_metrics[name]:
                    best_metrics[name] = val_metrics[name]['f1']
                    torch.save(
                        probes[name].state_dict(),
                        os.path.join(save_dir, f'probe_{name}_best.pt')
                    )
            
            # Print results
            print(f"\n{'Layer':<20} {'Loss':>8} {'Acc':>8} {'F1':>8} {'AUC':>8}")
            print("-" * 60)
            for name in sorted(probes.keys()):
                print(f"{name:<20} "
                      f"{train_losses[name]:>8.4f} "
                      f"{val_metrics[name]['accuracy']:>8.4f} "
                      f"{val_metrics[name]['f1']:>8.4f} "
                      f"{val_metrics[name]['auc']:>8.4f}")
        
        return probes, results
    
    def _train_epoch(self, probes, optimizers, criterion, train_loader):
        """Train for one epoch"""
        for probe in probes.values():
            probe.train()
        
        train_losses = {name: [] for name in probes}
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            features = self.extract_features(batch)
            labels = batch['label'].to(self.device)
            
            # Train each probe
            for name in probes:
                if name == 'projector':
                    vision_feat = features['projector_vision']
                    text_feat = features['embed_tokens']
                else:
                    vision_feat = features[f'{name}_vision']
                    text_feat = features[f'{name}_text']
                
                optimizers[name].zero_grad()
                pred = probes[name](vision_feat, text_feat)
                loss = criterion(pred, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probes[name].parameters(), 1.0)
                optimizers[name].step()
                
                train_losses[name].append(loss.item())
            
            # Update progress bar
            avg_loss = np.mean([np.mean(train_losses[n]) for n in probes])
            pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        return {name: np.mean(train_losses[name]) for name in probes}
    
    @torch.no_grad()
    def _validate(self, probes, val_loader):
        """Validation"""
        for probe in probes.values():
            probe.eval()
        
        predictions = {name: [] for name in probes}
        all_labels = []
        
        for batch in tqdm(val_loader, desc="Validating"):
            features = self.extract_features(batch)
            labels = batch['label'].cpu().numpy()
            all_labels.extend(labels)
            
            for name in probes:
                if name == 'projector':
                    vision_feat = features['projector_vision']
                    text_feat = features['embed_tokens']
                else:
                    vision_feat = features[f'{name}_vision']
                    text_feat = features[f'{name}_text']
                
                pred = probes[name](vision_feat, text_feat).cpu().numpy()
                predictions[name].extend(pred)
        
        # Calculate metrics
        metrics = {}
        for name in probes:
            preds = np.array(predictions[name])
            preds_binary = (preds > 0.5).astype(int)
            
            metrics[name] = {
                'accuracy': accuracy_score(all_labels, preds_binary),
                'f1': f1_score(all_labels, preds_binary),
                'auc': roc_auc_score(all_labels, preds)
            }
        
        return metrics
    
    def visualize_results(self, results: Dict, save_path: str = "probe_results.png"):
        """Visualize training results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sort by layer
        layers = sorted(results.keys(), key=lambda x: (
            0 if x == 'projector' else int(x.split('_')[1])
        ))
        
        # 1. Training loss curve
        ax = axes[0, 0]
        for name in layers:
            ax.plot(results[name]['train_loss'], label=name, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Final accuracy comparison
        ax = axes[0, 1]
        final_accs = [results[name]['val_acc'][-1] for name in layers]
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
        bars = ax.bar(range(len(layers)), final_accs, color=colors)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Final Validation Accuracy by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, final_accs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. F1 score comparison
        ax = axes[1, 0]
        final_f1s = [results[name]['val_f1'][-1] for name in layers]
        bars = ax.bar(range(len(layers)), final_f1s, color=colors, alpha=0.8)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Final F1 Score by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, f1) in enumerate(zip(bars, final_f1s)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. AUC comparison
        ax = axes[1, 1]
        final_aucs = [results[name]['val_auc'][-1] for name in layers]
        bars = ax.bar(range(len(layers)), final_aucs, color=colors, alpha=0.6)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('Final AUC-ROC by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, auc) in enumerate(zip(bars, final_aucs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization results saved to: {save_path}")
        
    def plot_layer_progression(self, results: Dict, save_path: str = "layer_progression.png"):
        """Plot layer-wise alignment progression"""
        layers = sorted([k for k in results.keys() if k.startswith('layer_')],
                       key=lambda x: int(x.split('_')[1]))
        layer_nums = [int(l.split('_')[1]) for l in layers]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extract final metrics
        accs = [results[l]['val_acc'][-1] for l in layers]
        f1s = [results[l]['val_f1'][-1] for l in layers]
        aucs = [results[l]['val_auc'][-1] for l in layers]
        
        # Plot curves
        ax.plot(layer_nums, accs, marker='o', label='Accuracy', linewidth=2, markersize=8)
        ax.plot(layer_nums, f1s, marker='s', label='F1 Score', linewidth=2, markersize=8)
        ax.plot(layer_nums, aucs, marker='^', label='AUC-ROC', linewidth=2, markersize=8)
        
        # Add projector baseline
        if 'projector' in results:
            proj_acc = results['projector']['val_acc'][-1]
            proj_f1 = results['projector']['val_f1'][-1]
            proj_auc = results['projector']['val_auc'][-1]
            
            ax.axhline(proj_acc, color='C0', linestyle='--', alpha=0.5, label='Projector Acc')
            ax.axhline(proj_f1, color='C1', linestyle='--', alpha=0.5, label='Projector F1')
            ax.axhline(proj_auc, color='C2', linestyle='--', alpha=0.5, label='Projector AUC')
        
        ax.set_xlabel('Layer Index', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Vision-Language Alignment Across Transformer Layers', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer progression plot saved to: {save_path}")


def main():
    """Main training pipeline"""
    
    # 1. Load COCO dataset
    print("Loading COCO dataset...")
    coco_ds = load_dataset("jxie/coco_captions")
    
    # 2. Initialize trainer
    trainer = LLaVAProbeTrainer(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        device="cuda:0"
    )
    
    # 3. Register hooks
    trainer.register_hooks()
    
    # 4. Create data loaders
    print("\nCreating data loaders...")
    train_dataset = COCOProbeDataset(
        jsonl_path="coco_probe_pairs.jsonl",
        processor=trainer.processor,
        coco_dataset=coco_ds,
        max_samples=None  # Use all data
    )
    
    # Split into training and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    # Create an instance of our new data collator
    data_collator = LlavaDataCollator(processor=trainer.processor)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator  # Use the new collator instance
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator  # Use the new collator instance
    )
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # 5. Train probes
    print("\nStart training probes...")
    probes, results = trainer.train_probes(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=1e-3,
        save_dir="probe_checkpoints"
    )
    
    # 6. Generate visualizations
    print("\nGenerating visualizations...")
    trainer.visualize_results(results, save_path="probe_results.png")
    trainer.plot_layer_progression(results, save_path="layer_progression.png")
    
    # 7. Save result summary
    print("\nSummary of results:")
    print(f"{'Layer':<20} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
    print("-" * 60)
    
    for name in sorted(results.keys(), key=lambda x: (
        0 if x == 'projector' else int(x.split('_')[1])
    )):
        acc = results[name]['val_acc'][-1]
        f1 = results[name]['val_f1'][-1]
        auc = results[name]['val_auc'][-1]
        print(f"{name:<20} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    
    # 8. Analyze alignment evolution
    print("\nCross-modal alignment analysis:")
    layer_results = {k: v for k, v in results.items() if k.startswith('layer_')}
    if layer_results:
        sorted_layers = sorted(layer_results.items(), 
                              key=lambda x: int(x[0].split('_')[1]))
        
        best_layer = max(sorted_layers, key=lambda x: x[1]['val_f1'][-1])
        worst_layer = min(sorted_layers, key=lambda x: x[1]['val_f1'][-1])
        
        print(f"  Best aligned layer: {best_layer[0]} (F1: {best_layer[1]['val_f1'][-1]:.4f})")
        print(f"  Worst aligned layer: {worst_layer[0]} (F1: {worst_layer[1]['val_f1'][-1]:.4f})")
        
        # Trend analysis
        f1_scores = [x[1]['val_f1'][-1] for x in sorted_layers]
        if len(f1_scores) > 2:
            early_avg = np.mean(f1_scores[:len(f1_scores)//3])
            late_avg = np.mean(f1_scores[-len(f1_scores)//3:])
            improvement = late_avg - early_avg
            
            print(f"  Early layers avg F1: {early_avg:.4f}")
            print(f"  Late layers avg F1: {late_avg:.4f}")
            print(f"  Alignment improvement: {improvement:+.4f}")
            
            if improvement > 0.05:
                print("  → Deeper layers show stronger cross-modal alignment")
            elif improvement < -0.05:
                print("  → Shallower layers retain more discriminative information")
            else:
                print("  → Relatively balanced alignment across layers")
    
    # 9. Cleanup
    trainer.remove_hooks()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()