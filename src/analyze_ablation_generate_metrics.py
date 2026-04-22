# -----------------------------------------------------------------
# analyze_ablation_generate_metrics.py (Causal intervention script)
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
import copy
from metrics import InterventionEvaluator, evaluate_repetition

# Device configuration (consistent with training)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
import pandas as pd

# --- [Copied] Model definitions used during training ---
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
        x_hat = self.decoder(z)
        return x_hat, z

class JointSparseAutoencoder(nn.Module):
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

# --- [Reused] JSAE model loader ---
def load_jsae_model(jsae_path: str, device: torch.device):
    print(f"Loading JSAE model: {jsae_path}")
    if not os.path.exists(jsae_path):
        raise FileNotFoundError(f"JSAE model file not found: {jsae_path}")
    checkpoint = torch.load(jsae_path, map_location=device)
    input_dim = checkpoint.get('input_dim', 4096)
    latent_dim = checkpoint.get('latent_dim', 16384)
    jsae = JointSparseAutoencoder(input_dim, latent_dim).to(device)
    jsae.load_state_dict(checkpoint['model_state_dict'])
    jsae.eval()
    print(f"  -> Joint model loaded (In={input_dim}, Latent={latent_dim})")
    vision_sae = jsae.vision_sae.eval()
    text_sae = jsae.text_sae.eval()
    return vision_sae, text_sae

# Global state for intervention hooks
g_intervention_gate = False     # Master switch: controls whether intervention is active
g_intervention_sae = None       # The JSAE used for intervention
g_intervention_indices = []     # Neuron indices to ablate (set to target value)
g_sequence_mask_for_hook = None # Mask covering all valid tokens (image + text)

def causal_intervention_hook(module, input_args, output):
    """
    Hook function: applies SAE reconstruction and intervention simultaneously
    to FFN activations for both image and text tokens.
    """
    if not g_intervention_gate or g_sequence_mask_for_hook is None:
        return output

    H_pre = output[0]
    H_post = H_pre.clone()

    try:
        sequence_mask = g_sequence_mask_for_hook[0].bool()

        # Adjust mask length dynamically to handle variable sequence lengths during generation
        if sequence_mask.shape[0] != H_pre.shape[0]:
            sequence_mask = torch.zeros(1, dtype=torch.bool, device=device)

        seq_len = H_pre.shape[0]
        hidden_dim = H_pre.shape[-1]

        H_pre = H_pre.view(seq_len, hidden_dim)
        full_mask = sequence_mask.unsqueeze(-1).expand_as(H_pre)

        A_sequence_flat = torch.masked_select(H_pre, full_mask)

        if A_sequence_flat.numel() == 0:
            return output

        A_sequence = A_sequence_flat.view(-1, hidden_dim)
        A_sequence = A_sequence.to(torch.float32)
        _, z_sequence = g_intervention_sae(A_sequence)

        # Core intervention: set target neuron activations to 1.0
        z_ablated = z_sequence.clone()
        z_ablated[:, g_intervention_indices] = 1.0

        A_ablated_hat = g_intervention_sae.decoder(z_ablated)
        A_ablated_hat = A_ablated_hat.to(H_post.dtype)
        A_ablated_hat_flat = A_ablated_hat.view(-1)

        H_post = torch.masked_scatter(H_post, full_mask, A_ablated_hat_flat)

        if isinstance(output, tuple):
            output_list = list(output)
            output_list[0] = H_post.unsqueeze(0)
            return tuple(output_list)
        else:
            return H_post.unsqueeze(0)

    except Exception as e:
        print(f"Error inside hook: {e}")
        raise


def steering_hook(module, input_args, output):
    """
    Hook function: feature-vector addition (steering).
    Adds SAE decoder feature vectors directly to original activations,
    avoiding reconstruction loss.
    """
    if not g_intervention_gate or g_sequence_mask_for_hook is None:
        return output

    H_pre = output[0] if isinstance(output, tuple) else output
    H_post = H_pre.clone()

    try:
        # Remove batch dimension -> [seq_len, hidden_dim] (assumes batch_size=1)
        if len(H_pre.shape) == 3:
            H_pre_view = H_pre.squeeze(0)
        else:
            H_pre_view = H_pre

        seq_len = H_pre_view.shape[0]
        hidden_dim = H_pre_view.shape[-1]

        sequence_mask = g_sequence_mask_for_hook[0].bool()

        # Adjust mask length dynamically for newly generated tokens
        if sequence_mask.shape[0] != seq_len:
            sequence_mask = torch.ones(seq_len, dtype=torch.bool, device=H_pre.device)

        # Build steering vector from decoder weights
        decoder_weights = g_intervention_sae.decoder[0].weight  # [hidden_dim, latent_dim]
        steering_vector = torch.zeros(hidden_dim, device=H_pre.device, dtype=H_pre.dtype)

        indices = g_intervention_indices
        if isinstance(indices, int):
            indices = [indices]

        for idx in indices:
            vec = decoder_weights[:, idx]
            vec = vec / (vec.norm() + 1e-8)  # Normalize for controllable intervention strength
            steering_vector += vec

        alpha = g_alpha

        # Add alpha * steering_vector only at mask-selected token positions
        steering_vector = steering_vector.unsqueeze(0)
        offset = torch.zeros_like(H_pre_view)
        offset[sequence_mask] = alpha * steering_vector
        H_post_view = H_pre_view + offset
        H_post = H_post_view.unsqueeze(0)

        if isinstance(output, tuple):
            output_list = list(output)
            output_list[0] = H_post
            return tuple(output_list)
        else:
            return H_post

    except Exception as e:
        print(f"Error inside hook: {e}")
        return output


def run_inference(model, processor, image, caption, layer=13):
    """
    Run one forward pass and return:
    1. response: generated text (for CLIP/PPL evaluation)
    2. inputs_origin: processor inputs (for mask extraction)
    """
    prompt = f"[INST] <image>\n{caption} [/INST]"
    hook_handle = model.language_model.layers[layer].register_forward_hook(steering_hook)
    inputs_origin = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs_origin,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding for stable results
            temperature=0.0,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    response = generated_text.split("[/INST]")[-1].strip()
    hook_handle.remove()

    return response, inputs_origin


def main():
    # --- A. Initialize ---
    print(">>> Initializing evaluator (loading CLIP, SBERT, GPT-2)...")
    evaluator = InterventionEvaluator(device=device)

    print(">>> Loading LLaVA model...")
    from train_jsae import _resolve_hf_snapshot_dir
    local_cache_root = "/mnt/data/shuhuizhen/models/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf"
    model_dir = _resolve_hf_snapshot_dir(local_cache_root)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="cuda:6",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    model.eval()

    # --- B. Load experiment configuration ---
    print(">>> Loading experiment configuration from experiments_config_0align.xlsx...")
    import ast
    try:
        experiments_df = pd.read_excel("experiments_config_0align.xlsx")
        print(experiments_df.columns)

        # Rename Chinese column headers to English for consistency
        column_map = {
            '\u539f\u59cb\u7c7b\u522b': 'original_category',
            '\u64cd\u7eb5\u7c7b\u522b': 'manipulation_category',
            '\u8986\u76d6rank':         'coverage_rank',
        }
        experiments_df = experiments_df.rename(
            columns={k: v for k, v in column_map.items() if k in experiments_df.columns}
        )

        experiments_df['neurons'] = experiments_df['neurons'].apply(ast.literal_eval)
        experiments_df['target_keywords'] = experiments_df['target_keywords'].apply(ast.literal_eval)
    except FileNotFoundError:
        print("Error: 'experiments_config_0align.xlsx' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error: Failed to read or parse 'experiments_config_0align.xlsx': {e}")
        return

    results_table = []
    global g_intervention_gate, g_sequence_mask_for_hook, g_intervention_sae, g_intervention_indices, g_rank, g_alpha

    # --- C. Run experiment loop ---
    for idx, exp in experiments_df.iterrows():
        print(f"\nRunning Experiment: {exp['id']}")
        layer = exp.get('layer', 13)
        jsae_path = "jsae_lm_{}_0.03.pth".format(layer)
        vision_sae, text_sae = load_jsae_model(jsae_path, device)
        vision_sae_wrapper = text_sae
        img_path = exp['img_path']
        if not os.path.exists(img_path):
            print(f"Skipping: image {img_path} not found")
            continue

        image = Image.open(img_path).convert('RGB')

        # ===========================
        # 1. Baseline (no intervention)
        # ===========================
        g_intervention_gate = False
        g_sequence_mask_for_hook = None
        base_text, inputs = run_inference(model, processor, image, exp['caption'], layer=layer)

        # ===========================
        # 2. Steering (with intervention)
        # ===========================
        best_sim_gain = None
        best_g_alpha = None
        best_sim_metrics = None
        best_sem_metrics = None
        best_qual_metrics = None
        best_steer_text = None
        prev_sim_gain = None
        prev_ppl = None

        for i in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
            g_alpha = i
            g_intervention_gate = True
            g_intervention_sae = vision_sae_wrapper

            g_intervention_indices = exp['neurons']
            g_rank = exp['coverage_rank']
            print(g_intervention_indices)

            image_token_mask = torch.zeros_like(inputs.attention_mask, dtype=torch.bool, device=device)
            image_token_id = model.config.image_token_index
            img_token_positions = (inputs.input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
            attention_mask = inputs.attention_mask

            text_mask = attention_mask.clone()
            img_token_pos = (inputs.input_ids[0] == model.config.image_token_index).nonzero(as_tuple=True)[0]
            if len(img_token_pos) > 0:
                start_pos, end_pos = img_token_pos[0].item(), img_token_pos[-1].item() + 1
                text_mask[0, start_pos:end_pos] = 0

            if len(img_token_positions) > 0:
                image_token_mask[0, img_token_positions] = True
                print(f"  -> Located {len(img_token_positions)} image tokens for intervention.")
            g_sequence_mask_for_hook = image_token_mask

            steer_text, _ = run_inference(model, processor, image, exp['caption'], layer=layer)
            if not steer_text.strip():
                print("Warning: steer_text is empty, skipping this alpha.")
                continue

            # ===========================
            # 3. Compute metrics
            # ===========================
            rep_metrics = evaluate_repetition(steer_text)
            sim_metrics = evaluator.compute_target_similarity(
                baseline_text=base_text,
                intervened_text=steer_text,
                target_sentence=exp['target_concept'],
            )
            sem_metrics = evaluator.evaluate_semantics(
                clean_text=base_text,
                intervened_text=steer_text,
                target_concept_text=exp['target_concept'],
                target_keywords=exp['target_keywords']
            )
            qual_metrics = evaluator.evaluate_quality(
                intervened_text=steer_text,
                original_image=image
            )
            qual_metrics_base = evaluator.evaluate_quality(
                intervened_text=base_text,
                original_image=image
            )

            # Stop if sim_gain starts decreasing
            if prev_sim_gain is not None and sim_metrics['sim_gain'] < prev_sim_gain and sim_metrics['sim_gain'] > 0.1:
                print(f"Sim_Gain started decreasing. Best alpha: {best_g_alpha}, gain: {best_sim_gain}")
                break
            # Stop if perplexity degrades significantly
            if prev_ppl is not None and (prev_ppl - qual_metrics['perplexity'] > 10 or qual_metrics['perplexity'] > 40):
                print(f"Perplexity degraded. Previous: {prev_ppl:.2f}, Current: {qual_metrics['perplexity']:.2f}")
                break
            if sim_metrics['sim_gain'] is not None and prev_sim_gain is not None:
                if sim_metrics['sim_gain'] < -0.2 and prev_sim_gain < -0.2:
                    break
            if rep_metrics['distinct_3'] is not None and rep_metrics['compression_ratio'] is not None:
                if rep_metrics['distinct_3'] < 0.3 and rep_metrics['compression_ratio'] < 0.3:
                    break

            # Record best
            best_sim_gain = sim_metrics['sim_gain']
            best_g_alpha = g_alpha
            best_sim_metrics = sim_metrics
            best_sem_metrics = sem_metrics
            best_qual_metrics = qual_metrics
            best_steer_text = steer_text
            prev_sim_gain = sim_metrics['sim_gain']
            prev_ppl = qual_metrics['perplexity']

        # ===========================
        # 4. Record results
        # ===========================
        row = {
            "Experiment": exp['id'],
            "Layer": layer,
            "Original_Category": exp.get('original_category', ''),
            "Manipulation_Category": exp.get('manipulation_category', ''),
            "CLIP_Gain": round(best_sem_metrics['clip_score_increase'], 4),
            "Sim_Gain": round(best_sim_metrics['sim_gain'], 4),
            "Sim_Absolute": round(best_sim_metrics['sim_interv'], 4),
            "Key_Hit": best_sem_metrics['keyword_hit'],
            "PPL": round(best_qual_metrics['perplexity'], 2),
            'PPL_Base': round(qual_metrics_base['perplexity'], 2),
            'Distinct_3': round(rep_metrics['distinct_3'], 2),
            'Compression_Ratio': round(rep_metrics['compression_ratio'], 2),
            "Text_Base": base_text,
            "Text_Steer": best_steer_text,
            "Best_Alpha": best_g_alpha
        }
        results_table.append(row)
        print(f"Original: {exp.get('original_category', '')}  Manipulation: {exp.get('manipulation_category', '')}")
        print(f"  -> Base: {base_text}")
        print(f"  -> Steer: {steer_text}")
        print(f"  -> CLIP Gain: {row['CLIP_Gain']}")
        print(f"  -> Semantic Sim Gain: {sim_metrics['sim_gain']:.4f}")
        print(f"  -> Keyword Hit: {sem_metrics['keyword_hit']}")
        print(f"  -> Perplexity: {qual_metrics['perplexity']:.2f} (Base: {qual_metrics_base['perplexity']:.2f})")

    # --- D. Final report ---
    print("\n" + "="*30 + " Evaluation Report " + "="*30)
    df = pd.DataFrame(results_table)
    pd.set_option('display.max_colwidth', 50)
    cols = ["Experiment", "Layer", "Original_Category", "Manipulation_Category",
            "CLIP_Gain", "Sim_Gain", "Sim_Absolute", "Key_Hit",
            "PPL", "PPL_Base", 'Distinct_3', 'Compression_Ratio',
            "Text_Base", "Text_Steer", "Best_Alpha"]
    print(df[cols].to_markdown(index=False))
    df.to_csv("steering_evaluation_metrics_align.csv", index=False)
    print("\nResults saved to steering_evaluation_metrics_align.csv")


if __name__ == "__main__":
    main()
