# -----------------------------------------------------------------
# train_jsae.py (Joint Sparse Autoencoder training script)
# -----------------------------------------------------------------
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
# from modelscope import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional, List, Dict
from datasets import load_dataset
import os
import jsonlines
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Global variable for storing activations
g_activations = {}
# Device configuration
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# --- [New] SAE Model Definition ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=1024, sparsity_lambda=1e-4, topk=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        self.sparsity_lambda = sparsity_lambda 
        self.topk = topk

    def forward(self, x):
        # x shape: [N, D]
        z = self.encoder(x) # [N, latent_dim]
        x_hat = self.decoder(z) # [N, D]
        return x_hat, z

class JointSparseAutoencoder(nn.Module):
    """
    A new model encapsulating Vision SAE and Text SAE.
    This model will be jointly trained to minimize a composite loss.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.vision_sae = SparseAutoencoder(input_dim, latent_dim)
        self.text_sae = SparseAutoencoder(input_dim, latent_dim)

    def forward(self, A_v, A_t):
        """
        Args:
            A_v: [B, D_in] - Batch of pooled vision activations
            A_t: [B, D_in] - Batch of pooled text activations
        Returns:
            Tuple containing components for loss calculation
        """
        x_hat_v, z_v = self.vision_sae(A_v)
        x_hat_t, z_t = self.text_sae(A_t)
        
        return x_hat_v, z_v, x_hat_t, z_t

class COCOProbeDataset(Dataset):
    def __init__( self, jsonl_path: str, processor, coco_dataset, max_samples: Optional[int] = None ):
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
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            pair = self.pairs[idx]
            img_path = pair['image']
            image = Image.open(img_path).convert('RGB')
            prompt = f"USER: <image>\n{pair['caption']}\nASSISTANT:"
            return { "image": image, "prompt": prompt, "label": pair["label"] }
        except Exception as e:
            return None

class LlavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        batch = [item for item in batch if item is not None]
        if not batch: return None
        images = [item['image'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
        image_sizes = torch.tensor([[img.height, img.width] for img in images])
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", 
            padding=True, truncation=True, max_length=4096
        )
        inputs['label'] = labels
        inputs['image_sizes'] = image_sizes
        return inputs

def get_activation_hook(name: str):
    def hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        g_activations[name] = act.detach().to(dtype=torch.float32)
    return hook

def train_joint_sae(
    model: LlavaNextForConditionalGeneration, 
    dataloader: DataLoader, 
    hook_module: nn.Module, 
    activation_key: str,
    input_dim: int, 
    latent_dim: int,
    layer_name: str,
    epochs: int = 20
):
    """
    [New] Train *joint* SAE model.
    This loop extracts (A_v, A_t) pairs simultaneously and uses
    a composite loss function containing reconstruction, sparsity, 
    and *alignment* losses.
    """
    
    print(f"\n--- Starting Joint SAE Training: {layer_name} ---")
    jsae = JointSparseAutoencoder(input_dim, latent_dim).to(device)
    jsae.train() 
    
    # Hyperparameters (adjustable)
    recon_coef = 1.0     # Reconstruction loss weight
    sparse_coef = 0.03   # L1 sparsity loss weight
    align_coef = 1.0     # [New] Alignment loss weight
    
    
    learning_rate = 1e-5
    
    optimizer = torch.optim.Adam(jsae.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-7
    )
    
    image_token_index = model.config.image_token_index#model.config.image_token_index
    best_total_loss = float('inf')
    
    for epoch in range(epochs):
        
        
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training {layer_name})")
        epoch_l0_vision = []
        epoch_l0_text = []
        
        for batch in pbar:
            if batch is None:
                continue

            try:
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                g_activations.clear()
                handle = hook_module.register_forward_hook(get_activation_hook(activation_key))
                with torch.no_grad():
                    model(
                        input_ids=batch_on_device['input_ids'],
                        attention_mask=batch_on_device['attention_mask'],
                        pixel_values=batch_on_device['pixel_values'].to(model.dtype),
                        image_sizes=batch_on_device['image_sizes']
                    )
                handle.remove()

                if activation_key not in g_activations:
                    continue
                
                full_activations = g_activations[activation_key]
                input_ids = batch_on_device['input_ids']
                attention_mask = batch_on_device['attention_mask']
                batch_size = input_ids.shape[0]

                text_mask = attention_mask.clone()
                for b in range(batch_size):
                    img_token_pos = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
                    if len(img_token_pos) > 0:
                        start_pos, end_pos = img_token_pos[0].item(), img_token_pos[-1].item() + 1
                        text_mask[b, start_pos:end_pos] = 0
                vision_mask = (text_mask == 0) & (attention_mask == 1)

                A_v_list = []
                A_t_list = []
                
                for b in range(batch_size):
                    v_mask_b = vision_mask[b].bool()
                    t_mask_b = text_mask[b].bool()
                    
                    if v_mask_b.sum() > 0 and t_mask_b.sum() > 0:
                        # print('shape',full_activations[b].shape, v_mask_b.shape, t_mask_b.shape)
                        # print('shape2',full_activations[b][v_mask_b].shape)
                        # print('shape3',full_activations[b][t_mask_b].shape)
                        pooled_v = full_activations[b][v_mask_b].mean(dim=0)
                        pooled_t = full_activations[b][t_mask_b].mean(dim=0)
                        A_v_list.append(pooled_v)
                        A_t_list.append(pooled_t)

                if not A_v_list:
                    continue 
                
                A_v_batch = torch.stack(A_v_list) # [B_valid, D_in]
                A_t_batch = torch.stack(A_t_list) # [B_valid, D_in]
                
                optimizer.zero_grad()
                
                x_hat_v, z_v, x_hat_t, z_t = jsae(A_v_batch, A_t_batch)
                
                
                loss_recon_v = F.mse_loss(x_hat_v, A_v_batch)
                loss_recon_t = F.mse_loss(x_hat_t, A_t_batch)
                loss_recon = (loss_recon_v + loss_recon_t) / 2.0
                
                loss_sparse_v = torch.abs(z_v).mean()
                loss_sparse_t = torch.abs(z_t).mean()
                loss_sparse = (loss_sparse_v + loss_sparse_t) / 2.0
                
                loss_align = 1.0 - F.cosine_similarity(z_v, z_t).mean()

                total_loss = (recon_coef * loss_recon) + \
                             (sparse_coef * loss_sparse) + \
                             (align_coef * loss_align)

                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    l0_v_pct = (torch.where(z_v > 0, 1.0, 0.0)).mean().item()
                    l0_t_pct = (torch.where(z_t > 0, 1.0, 0.0)).mean().item()
                    epoch_l0_vision.append(l0_v_pct)
                    epoch_l0_text.append(l0_t_pct)

                epoch_losses.append(total_loss.item())
                pbar.set_postfix({
                    "Total": f"{total_loss.item():.4f}",
                    "Recon": f"{loss_recon.item():.4f}",
                    "Sparse(L1)": f"{loss_sparse.item():.4f}",
                    "Align": f"{loss_align.item():.4f}",
                    "L0-V%": f"{l0_v_pct:.2%}",
                    "L0-T%": f"{l0_t_pct:.2%}"
                })

            except Exception as e:
                print(f'batch error: {e}')
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

        if not epoch_losses:
            print(f"Epoch {epoch+1} fail batch,skip")
            continue

        avg_total_loss = sum(epoch_losses) / len(epoch_losses)
        avg_l0_v = sum(epoch_l0_vision) / len(epoch_l0_vision)
        avg_l0_t = sum(epoch_l0_text) / len(epoch_l0_text)
        
        print(f"Epoch {epoch+1}, avg_total_loss: {avg_total_loss:.6f}")
        print(f"  -> avg_l0 (Vision): {avg_l0_v:.2%}")
        print(f"  -> avg_l0 (Text):   {avg_l0_t:.2%}")
        print(f"Epoch {epoch+1}, avg_total_loss: {avg_total_loss:.6f}")
        scheduler.step(avg_total_loss)
        
        save_path = f"./{layer_name}.pth"
        if avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            torch.save({
                "model_state_dict": jsae.state_dict(),
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "recon_coef": recon_coef,
                "sparse_coef": sparse_coef,
                "align_coef": align_coef,
            }, save_path)
            print(f"Best model saved to {save_path}")
        
    return jsae

def main():
    """Main training process (JSAE version)"""
    
    # 1. Load model and processor
    print("Loading LLaVA model...")
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf" #llava-hf/llava-v1.6-mistral-7b-hf
    processor = AutoProcessor.from_pretrained(model_name) #LlavaNextForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
    model.eval()

    # 2. Load COCO dataset
    print("Loading COCO dataset...")
    coco_ds = load_dataset("jxie/coco_captions")
    
    # 3. Create data loader
    print("\nCreating data loader...")
    
    jsonl_file_path = "coco_probe_pairs_positive.jsonl"
    if not os.path.exists(jsonl_file_path):
        print(f"Error: File '{jsonl_file_path}' not found.")
        return
        
    train_dataset = COCOProbeDataset(
        jsonl_path=jsonl_file_path,
        processor=processor,
        coco_dataset=coco_ds,
        max_samples=None  # Use all data
    )
    
    # Split dataset (we use all data here)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    data_collator = LlavaDataCollator(processor=processor)

    # Merge train_subset and val_subset to use all data
    all_data_subset = torch.utils.data.ConcatDataset([train_subset, val_subset])
    
    train_loader = DataLoader(
        all_data_subset,
        batch_size=8, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    print(f"Samples for JSAE training: {len(all_data_subset)}")
    
    # 4. Define target layer and hooks
    layer = 13
    activation_key = f'layer_{layer}_mlp'
    hook_module = model.language_model.layers[layer]
    
    input_dim = model.language_model.config.hidden_size # 4096
    latent_dim = input_dim * 4 # 16384 
    
    print(f"Target module: LM Layer {layer} MLP")
    print(f"Activation dimension (Input Dim): {input_dim}")
    print(f"SAE latent dimension (Latent Dim): {latent_dim}")

    # 5. [New] Train *joint* SAE
    train_joint_sae(
         model=model,
         dataloader=train_loader,
         hook_module=hook_module,
         activation_key=activation_key,
         input_dim=input_dim,
         latent_dim=latent_dim,
         layer_name=f"jsae_lm_{layer}_0.03", 
         epochs=20 # Joint training may require more epochs
     )

if __name__ == "__main__":
    main()