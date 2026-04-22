# -----------------------------------------------------------------
# analysis_jsae.py (Full analysis script for Joint SAE)
# -----------------------------------------------------------------

import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
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
import random

# --- Imports for analysis and visualization ---
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import html
from skimage.transform import resize # pip install scikit-image

# Global variable for storing activations
g_activations = {}
# Device configuration (consistent with training)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# --- [Copied] Model definitions used during training ---
# (Must be exactly replicated here so that weights load correctly)

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
        z = self.encoder(x)
        # ... (topk logic, if used) ...
        x_hat = self.decoder(z)
        return x_hat, z

class JointSparseAutoencoder(nn.Module):
    """
    Joint JSAE model definition (for loading checkpoint)
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vision_sae = SparseAutoencoder(input_dim, latent_dim)
        self.text_sae = SparseAutoencoder(input_dim, latent_dim)

    def forward(self, A_v, A_t):
        x_hat_v, z_v = self.vision_sae(A_v)
        x_hat_t, z_t = self.text_sae(A_t)
        return x_hat_v, z_v, x_hat_t, z_t

# --- [New] JSAE model loader ---
def load_jsae_model(jsae_path: str, device: torch.device):
    """
    Load a pre-trained *joint* SAE model and return two
    independent (vision, text) SAEs for analysis.
    """
    print(f"Loading JSAE model: {jsae_path}")
    if not os.path.exists(jsae_path):
        raise FileNotFoundError(f"JSAE model file not found: {jsae_path}")
        
    checkpoint = torch.load(jsae_path, map_location=device)
    
    # Read dimensions from checkpoint
    input_dim = checkpoint.get('input_dim', 4096)  # default fallback
    latent_dim = checkpoint.get('latent_dim', 16384)
    
    # 1. Instantiate *joint* model
    jsae = JointSparseAutoencoder(input_dim, latent_dim).to(device)
    
    # 2. Load joint state dict
    jsae.load_state_dict(checkpoint['model_state_dict'])
    jsae.eval()
    print(f"  -> Joint model loaded (In={input_dim}, Latent={latent_dim})")
    
    # 3. [Key] Return the two trained sub-models
    vision_sae = jsae.vision_sae.eval()
    text_sae = jsae.text_sae.eval()
    return vision_sae, text_sae

# --- [Reused] Data loader (from original script) ---
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
            return {
                "image": image,
                "prompt": prompt,
                "label": pair["label"],
                "image_id": pair["image_id"],
                "caption": pair["caption"]
            }
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
        image_ids = [item['image_id'] for item in batch]
        captions = [item['caption'] for item in batch]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", 
            padding=True, truncation=True, max_length=4096
        )
        inputs['label'] = labels
        inputs['image_sizes'] = image_sizes
        inputs['image_ids'] = image_ids
        inputs['captions'] = captions
        return inputs

# --- [Reused] Hook function ---
def get_activation_hook(name: str):
    def hook(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        g_activations[name] = act.detach().to(dtype=torch.float32)
    return hook

# --- [Analysis 1] Generate paired latent vectors (for correlation analysis) ---
def generate_latent_pairs(
    model: LlavaNextForConditionalGeneration, dataloader: DataLoader, hook_module: nn.Module, 
    activation_key: str, text_sae: SparseAutoencoder, vision_sae: SparseAutoencoder, 
    image_token_index: int, dataset_name: str = "data"
) -> (np.ndarray, np.ndarray, List[Dict]):
    print(f"\n--- Generating paired latent vectors (z_v, z_t) for {dataset_name} ---")
    all_z_text, all_z_vision, all_sample_info = [], [], []
    pbar = tqdm(dataloader, desc=f"Extracting {dataset_name} latent vectors")
    with torch.no_grad():
        for batch in pbar:
            if batch is None: continue
            try:
                batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k not in ['image_ids', 'captions']}
                g_activations.clear()
                handle = hook_module.register_forward_hook(get_activation_hook(activation_key))
                model(
                    input_ids=batch_on_device['input_ids'], attention_mask=batch_on_device['attention_mask'],
                    pixel_values=batch_on_device['pixel_values'].to(model.dtype),
                    image_sizes=batch_on_device['image_sizes']
                )
                handle.remove()
                if activation_key not in g_activations: continue
                full_activations = g_activations[activation_key]
                input_ids, attention_mask = batch_on_device['input_ids'], batch_on_device['attention_mask']
                batch_size = input_ids.shape[0]
                text_mask = attention_mask.clone()
                for b in range(batch_size):
                    img_token_pos = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
                    if len(img_token_pos) > 0: text_mask[b, img_token_pos[0]:img_token_pos[-1]+1] = 0
                vision_mask = (text_mask == 0) & (attention_mask == 1)
                for b in range(batch_size):
                    text_mask_b, vision_mask_b = text_mask[b].bool(), vision_mask[b].bool()
                    if text_mask_b.sum() > 0 and vision_mask_b.sum() > 0:
                        pooled_text_feat = full_activations[b][text_mask_b].mean(dim=0, keepdim=True)
                        _, z_text = text_sae(pooled_text_feat)
                        pooled_vision_feat = full_activations[b][vision_mask_b].mean(dim=0, keepdim=True)
                        _, z_vision = vision_sae(pooled_vision_feat)
                        all_z_text.append(z_text.squeeze(0).cpu())
                        all_z_vision.append(z_vision.squeeze(0).cpu())
                        all_sample_info.append({"image_id": batch['image_ids'][b], "caption": batch['captions'][b]})
            except Exception as e:
                print(f'Error extracting batch: {e}')
                torch.cuda.empty_cache()
    Z_text = torch.stack(all_z_text).numpy()
    Z_vision = torch.stack(all_z_vision).numpy()
    print(f"  -> Successfully extracted {Z_text.shape[0]} valid pairs")
    return Z_vision, Z_text, all_sample_info

# --- [Analysis 2] Neuron correlation & visualization (Task 3) ---
def find_neuron_correlations_and_visualize(
    Z_vision: np.ndarray, Z_text: np.ndarray, sample_info_list: List[Dict],
    top_k_pairs: int = 10, top_m_samples: int = 5
):
    print(f"\n--- Task 3: Finding Top-{top_k_pairs} cross-modal neuron pairs ---")
    n_samples, d_sae_v = Z_vision.shape
    _, d_sae_t = Z_text.shape
    v_var, t_var = np.var(Z_vision, axis=0), np.var(Z_text, axis=0)
    var_thresh = 1e-5 
    v_active_indices = np.where(v_var > var_thresh)[0]
    t_active_indices = np.where(t_var > var_thresh)[0]
    print(f"  Filtering low-variance neurons (variance < {var_thresh}):")
    print(f"    Vision: {d_sae_v} -> {len(v_active_indices)} active neurons")
    print(f"    Text:   {d_sae_t} -> {len(t_active_indices)} active neurons")
    if len(v_active_indices) == 0 or len(t_active_indices) == 0:
        print("  Error: No active neurons found. Skipping.")
        return None
    Z_v_active, Z_t_active = Z_vision[:, v_active_indices], Z_text[:, t_active_indices]
    scaler_v, scaler_t = StandardScaler(), StandardScaler()
    Z_v_std = scaler_v.fit_transform(Z_v_active)
    Z_t_std = scaler_t.fit_transform(Z_t_active)
    print(f"  Finding best text neuron match for {len(v_active_indices)} active vision neurons...")
    cross_corr_matrix = (Z_v_std.T @ Z_t_std) / (n_samples - 1)
    abs_corr = np.abs(cross_corr_matrix)
    indices = np.argsort(abs_corr, axis=None)[-top_k_pairs*2:] 
    v_indices_local, t_indices_local = np.unravel_index(indices, cross_corr_matrix.shape)
    top_pairs = []
    added_pairs = set()
    for v_local_idx, t_local_idx in zip(v_indices_local, t_indices_local):
        v_global_idx, t_global_idx = v_active_indices[v_local_idx], t_active_indices[t_local_idx]
        if (v_global_idx, t_global_idx) not in added_pairs:
            corr = cross_corr_matrix[v_local_idx, t_local_idx]
            top_pairs.append((corr, v_global_idx, t_global_idx))
            added_pairs.add((v_global_idx, t_global_idx))
    top_pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    top_pairs = top_pairs[:top_k_pairs]
    
    print("\n  --- Top cross-modal neuron pairs (V, T, Correlation) ---")
    print("  (Expected: Correlation should be very high, > 0.9)")
    for corr, v_idx, t_idx in top_pairs:
        print(f"    (V: {v_idx}, T: {t_idx}) -> Corr: {corr:.6f}")

    print(f"\n--- Visualizing Top-{top_k_pairs} pairs (Top {top_m_samples} activating samples) ---")
    for rank, (corr, v_idx, t_idx) in enumerate(top_pairs):
        print("\n" + "="*50)
        print(f"  Rank {rank+1} / {top_k_pairs} | (V_neuron={v_idx}, T_neuron={t_idx}) | Corr={corr:.4f}")
        print("="*50)
        co_activation_scores = Z_vision[:, v_idx] * Z_text[:, t_idx]
        top_sample_indices = np.argsort(co_activation_scores)[-top_m_samples:][::-1]
        print(f"  --- Top {top_m_samples} co-activating samples (Z_v, Z_t, Caption): ---")
        for sample_idx in top_sample_indices:
            v_val, t_val = Z_vision[sample_idx, v_idx], Z_text[sample_idx, t_idx]
            info = sample_info_list[sample_idx]
            print(f"    [V_act={v_val:.3f}, T_act={t_val:.3f}] Caption: \"{info['caption']}\"")
            print(f"                                (Image ID: {info['image_id']})")
    
    # Return dict for causal testing
    # Format: {t_idx: v_idx}
    neuron_pair_dict = {t_idx: v_idx for corr, v_idx, t_idx in top_pairs}
    return neuron_pair_dict
    




from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def analyze_neuron_clustering(Z_matrix, active_indices, label_name="Vision", n_clusters=20,layer=13):
    """
    Perform neuron clustering analysis and visualization.
    
    Args:
        Z_matrix: Activation matrix of shape (N_samples, D_features)
        active_indices: List of active neuron indices (from prior filtering)
        label_name: "Vision" or "Text", used for plot titles
        n_clusters: Number of K-Means clusters
    """
    print(f"\n--- Starting {label_name} neuron clustering analysis ---")
    
    # 1. Get activation vectors of active neurons
    # Transpose the matrix because we cluster "neurons" (columns), not "data samples" (rows)
    # Shape becomes: (n_active_neurons, n_samples)
    neuron_features = Z_matrix[:, active_indices].T 
    
    # 2. Normalization (important for clustering; L2-normalize so that each neuron
    #    vector has unit length, making Euclidean distance equivalent to cosine similarity)
    norms = np.linalg.norm(neuron_features, axis=1, keepdims=True)
    neuron_features_norm = neuron_features / (norms + 1e-8)

    # 3. K-Means clustering
    print(f"  Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(neuron_features_norm)
    
    # 4. Dimensionality reduction for visualization (t-SNE)
    # For large neuron counts, first reduce to 50 dims with PCA, then apply t-SNE
    print("  Running t-SNE for visualization...")
    n_components_pca = min(50, neuron_features_norm.shape[1], neuron_features_norm.shape[0])
    pca = PCA(n_components=n_components_pca)
    features_pca = pca.fit_transform(neuron_features_norm)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(features_pca)
    
    # 5. Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'{label_name} Neurons Clustering (t-SNE projection)')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    
    save_path = f"clustering_{label_name.lower()}_{layer}_neurons.png"
    plt.savefig(save_path,dpi=300)
    print(f"  [Saved] Clustering visualization saved to: {save_path}")
    
    return cluster_labels, embedding

def interpret_clusters(Z_matrix, active_indices, cluster_labels, sample_info_list, n_clusters=20,top100v_neuron=None):
    """
    Interpret clustering results: find the most representative neurons in each cluster
    and examine which samples they activate most strongly.
    """
    print("\n--- Interpreting clustering results ---")
    for k in range(n_clusters):
        # Find local indices (within active_indices) of all neurons in this cluster
        members_local_indices = np.where(cluster_labels == k)[0]
        if len(members_local_indices) == 0: continue
        
        # Map local indices back to global neuron indices
        members_global_indices = active_indices[members_local_indices]
        
        # Compute the "mean activation vector" of this cluster as its centroid
        cluster_centroid_act = np.mean(Z_matrix[:, members_global_indices], axis=1)
        
        # Find samples that activate this cluster centroid most strongly (Top 3)
        top_sample_indices = np.argsort(cluster_centroid_act)[-5:][::-1]
        
        print(f"\n[Cluster {k}] Contains {len(members_global_indices)} neurons")
        print(members_global_indices)
        new_members_global_indices=[]
        for i in list(members_global_indices):
            if i in top100v_neuron:
                new_members_global_indices.append(i)
        new_members_global_indices=np.array(new_members_global_indices)
        print(new_members_global_indices)

        print("  Main activation concepts (based on top samples):")
        for idx in top_sample_indices:
            caption = sample_info_list[idx]['caption']
            print(f"    - {caption}...")

from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy
import numpy as np

def evaluate_clusters(Z_matrix, active_indices, neuron_features_norm, cluster_labels, n_clusters=20):
    """
    Evaluate the independence of each cluster via:
      1. Silhouette score (geometric independence)
      2. Inter/Intra distance ratio (geometric compactness + separation)
      3. Activation entropy (functional/semantic independence)
    
    Args:
        Z_matrix: Raw activation matrix (N_samples, D_features)
        active_indices: List of active neuron indices
        neuron_features_norm: L2-normalized neuron vectors (n_active_neurons, n_samples)
        cluster_labels: K-Means cluster labels
        n_clusters: Number of clusters
    
    Returns:
        dict: {cluster_id: {'silhouette': x, 'inter_intra': y, 'entropy': z}}
    """
    results = {}
    
    # 1. Silhouette
    sil_samples = silhouette_samples(neuron_features_norm, cluster_labels, metric="cosine")
    
    # 2. K-Means centers + inter-cluster distances
    cluster_centers = []
    for k in range(n_clusters):
        members = neuron_features_norm[cluster_labels == k]
        if len(members) == 0:
            cluster_centers.append(np.zeros(neuron_features_norm.shape[1]))
        else:
            cluster_centers.append(members.mean(axis=0))
    cluster_centers = np.stack(cluster_centers)
    center_dists = cosine_distances(cluster_centers)

    for k in range(n_clusters):
        members_idx = np.where(cluster_labels == k)[0]
        if len(members_idx) == 0:
            continue

        # Silhouette
        sil_score = sil_samples[members_idx].mean()

        # Inter/Intra distance ratio
        members = neuron_features_norm[members_idx]
        intra_dist = cosine_distances(members, cluster_centers[k:k+1]).mean()
        inter_dist = np.min(center_dists[k][np.arange(n_clusters) != k])
        inter_intra_ratio = inter_dist / (intra_dist + 1e-8)

        # Activation entropy
        global_indices = np.array(active_indices)[members_idx]
        acts = Z_matrix[:, global_indices].mean(axis=1)
        p = np.abs(acts)
        p = p / (p.sum() + 1e-8)
        act_entropy = entropy(p)

        results[k] = {
            'silhouette': sil_score,
            'inter_intra': inter_intra_ratio,
            'entropy': act_entropy
        }

    return results

# --- [Main function] ---
def main():
    """
    Main analysis pipeline (JSAE version)
    """
    
    # 1. Load LLaVA model
    print("Loading LLaVA model...")
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True
        )
    model.eval()
    
    image_token_index = model.config.image_token_index

    # 2. Load COCO dataset
    print("Loading COCO dataset...")
    coco_ds = load_dataset("jxie/coco_captions")
    jsonl_file_path = "coco_probe_pairs_positive.jsonl"
    if not os.path.exists(jsonl_file_path):
        print(f"Error: '{jsonl_file_path}' not found.")
        return
        
    train_dataset = COCOProbeDataset(
        jsonl_path=jsonl_file_path,
        processor=processor,
        coco_dataset=coco_ds,
        max_samples=None 
    )
    
    # 3. Split dataset (use validation split for all analyses)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    data_collator = LlavaDataCollator(processor=processor)
    
    # 4. Define hook
    layer = 13
    activation_key = f'layer_{layer}_mlp'
    hook_module = model.language_model.layers[layer]
    
    # 5. Load trained JSAE model
    jsae_path = 'jsae_lm_13_0align.pth'#"jsae_lm_{}_0.03.pth".format(layer)#"jsae_lm_7_0.03.pth" #"jsae_lm_13_0.03.pth" 
    try:
        vision_sae, text_sae = load_jsae_model(jsae_path, device)
    except FileNotFoundError:
        print(f"Error: Model '{jsae_path}' not found. Make sure it is in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- [Analysis 1] Run neuron correlation (check alignment success) ---
    print("\n" + "*"*60)
    print(" Step 1: Run neuron correlation analysis (verify JSAE training)")
    print("*"*60)
    
    val_loader = DataLoader(
        val_subset, batch_size=8, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=data_collator
    )
    
    Z_vision_val, Z_text_val, val_samples = generate_latent_pairs(
        model=model, dataloader=val_loader, hook_module=hook_module,
        activation_key=activation_key, text_sae=text_sae, vision_sae=vision_sae,
        image_token_index=image_token_index, dataset_name="validation set"
    )
    
    neuron_pair_dict = find_neuron_correlations_and_visualize(
        Z_vision=Z_vision_val, Z_text=Z_text_val, sample_info_list=val_samples,
        top_k_pairs=100, top_m_samples=5
    )
    top100v_neuron = list(neuron_pair_dict.values())
    print("top100v_neuron", top100v_neuron)
    if neuron_pair_dict is None:
        print("Error: Neuron correlation analysis failed. Stopping.")
        return

# --- [Analysis 4] Neuron clustering analysis ---
    print("\n" + "*"*60)
    print(" Step 4: Neuron clustering analysis")
    print("*"*60)
    
    # Recompute active neuron indices
    v_var = np.var(Z_vision_val, axis=0)
    t_var = np.var(Z_text_val, axis=0)
    var_thresh = 1e-5
    v_active_indices = np.where(v_var > var_thresh)[0]
    t_active_indices = np.where(t_var > var_thresh)[0]
    n_clusters=20
        # 2. Cluster vision neurons
    if len(v_active_indices) > 0:
        v_labels, _ = analyze_neuron_clustering(
            Z_vision_val, v_active_indices, label_name="Vision", n_clusters=n_clusters,layer=layer
        )
        interpret_clusters(Z_vision_val, v_active_indices, v_labels, val_samples, n_clusters=n_clusters,top100v_neuron=top100v_neuron)
                # Evaluate cluster independence
        neuron_features = Z_vision_val[:, v_active_indices].T
        norms = np.linalg.norm(neuron_features, axis=1, keepdims=True)
        neuron_features_norm = neuron_features / (norms + 1e-8)

        cluster_scores = evaluate_clusters(
            Z_vision_val,
            v_active_indices,
            neuron_features_norm,
            v_labels,
            n_clusters=n_clusters
        )

        # Print independence metrics
        for k, score_dict in cluster_scores.items():
            print(f"Cluster {k}: Silhouette={score_dict['silhouette']:.3f}, "
                f"Inter/Intra={score_dict['inter_intra']:.3f}, Entropy={score_dict['entropy']:.3f}")

        
    # 3. Cluster text neurons
    if len(t_active_indices) > 0:
        t_labels, _ = analyze_neuron_clustering(
            Z_text_val, t_active_indices, label_name="Text", n_clusters=n_clusters,layer=layer
        )
        interpret_clusters(Z_text_val, t_active_indices, t_labels, val_samples, n_clusters=n_clusters)
                        # Evaluate cluster independence
        neuron_features = Z_text_val[:, t_active_indices].T
        norms = np.linalg.norm(neuron_features, axis=1, keepdims=True)
        neuron_features_norm = neuron_features / (norms + 1e-8)

        cluster_scores = evaluate_clusters(
            Z_text_val,
            t_active_indices,
            neuron_features_norm,
            t_labels,
            n_clusters=n_clusters
        )

        # Print independence metrics
        for k, score_dict in cluster_scores.items():
            print(f"Cluster {k}: Silhouette={score_dict['silhouette']:.3f}, "
                f"Inter/Intra={score_dict['inter_intra']:.3f}, Entropy={score_dict['entropy']:.3f}")

if __name__ == "__main__":
    main()
