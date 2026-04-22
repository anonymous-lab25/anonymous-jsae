import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sentence_transformers import util

import zlib

def evaluate_repetition(text):
    """
    Compute repetition metrics for a given text.
    Returns:
    - distinct_2: 2-gram diversity (closer to 1.0 is better; lower means more repetition)
    - distinct_3: 3-gram diversity
    - compression_ratio: compression ratio (lower means more repetition)
    """
    if not text:
        return {"distinct_2": 0.0, "distinct_3": 0.0, "compression_ratio": 0.0}

    # 1. Simple whitespace tokenization (sufficient for repetition detection)
    tokens = text.split()
    total_tokens = len(tokens)
    
    if total_tokens < 4:
        # Text too short for reliable repetition measurement; return safe defaults
        return {"distinct_2": 1.0, "distinct_3": 1.0, "compression_ratio": 1.0}

    # 2. Compute Distinct-N
    def calculate_distinct_n(n):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if len(ngrams) == 0:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    distinct_2 = calculate_distinct_n(2)
    distinct_3 = calculate_distinct_n(3)

    # 3. Compute compression ratio (zlib)
    bytes_text = text.encode('utf-8')
    compressed = zlib.compress(bytes_text)
    compression_ratio = len(compressed) / len(bytes_text)

    return {
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "compression_ratio": compression_ratio
    }

class InterventionEvaluator:
    def __init__(self, device="cuda"):
        self.device = device

        # --- 1. Load auxiliary models (loaded on demand to avoid OOM) ---
        print("Loading evaluation auxiliary models...")
        
        # CLIP model (for semantic similarity and image-text consistency)
        self.clip_model_name = "openai/clip-vit-large-patch14"
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device).eval()
        
        # Sentence-BERT (for embedding space drift)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu').eval()
        
        # PPL model (lightweight model such as GPT-2 recommended)
        self.ppl_model_name = "gpt2" 
        self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_name)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_name).to(self.device).eval()
        
        print("Evaluation models loaded.")
        
    def compute_target_similarity(self, 
                                  baseline_text: str, 
                                  intervened_text: str, 
                                  target_sentence: str):
        """
        Compute the semantic similarity gain between generated text and a target concept
        sentence (SBERT Cosine Similarity).
        """
        # 1. Encode texts (batch encoding is more efficient)
        embeddings = self.sbert_model.encode(
            [baseline_text, intervened_text, target_sentence], 
            convert_to_tensor=True
        )
        
        emb_base = embeddings[0]
        emb_interv = embeddings[1]
        emb_target = embeddings[2]
        
        # 2. Compute cosine similarity with the target
        sim_base_target = util.pytorch_cos_sim(emb_base, emb_target).item()
        sim_interv_target = util.pytorch_cos_sim(emb_interv, emb_target).item()
        
        # 3. Compute gain (positive means the intervened text is closer to the target)
        similarity_gain = sim_interv_target - sim_base_target
        
        return {
            "sim_base": sim_base_target,
            "sim_interv": sim_interv_target,
            "sim_gain": similarity_gain
        }

    def _get_ids_for_words(self, tokenizer, words: List[str]):
        """Helper: get token IDs for target words in the tokenizer vocabulary"""
        ids = []
        for w in words:
            # Assumes word is a single token; prepend space as LLaVA/Llama typically does
            tokenized = tokenizer.encode(" " + w, add_special_tokens=False)
            if len(tokenized) > 0:
                ids.append(tokenized[0])
        return torch.tensor(ids, device=self.device)

    # ============================================================
    # 1. Prediction-probability-based metrics (Micro-level: Logits)
    # ============================================================
    def evaluate_logits(self, 
                        clean_logits: torch.Tensor, 
                        intervened_logits: torch.Tensor, 
                        tokenizer, 
                        target_words: List[str]):
        """
        Compute Logit Difference and Probability Mass Shift.
        Note: logits should be [batch_size, vocab_size] (typically the first generated token).
        """
        target_ids = self._get_ids_for_words(tokenizer, target_words)
        
        if len(target_ids) == 0:
            return {"logit_diff": 0.0, "prob_mass_shift": 0.0}

        # 1. Logit Difference (average change across all target word logits)
        clean_target_logits = clean_logits[:, target_ids]
        interv_target_logits = intervened_logits[:, target_ids]
        
        logit_diff = (interv_target_logits - clean_target_logits).mean().item()

        # 2. Probability Mass Shift (change in total probability mass)
        probs_clean = F.softmax(clean_logits, dim=-1)
        probs_interv = F.softmax(intervened_logits, dim=-1)
        
        mass_clean = probs_clean[:, target_ids].sum(dim=-1).mean().item()
        mass_interv = probs_interv[:, target_ids].sum(dim=-1).mean().item()
        
        prob_shift = mass_interv - mass_clean

        return {
            "logit_diff": logit_diff,
            "prob_mass_shift": prob_shift,
            "prob_clean": mass_clean,
            "prob_interv": mass_interv
        }

    # ============================================================
    # 2. Generated-text-semantics-based metrics (Macro-level: Semantics)
    # ============================================================
    def evaluate_semantics(self, 
                           clean_text: str, 
                           intervened_text: str, 
                           target_concept_text: str, 
                           target_keywords: List[str]):
        """
        Compute CLIP Score, Keyword Hit Rate, and Embedding Drift.
        """
        # 1. CLIP-Score (Text-to-Concept)
        inputs = self.clip_processor(
            text=[intervened_text, target_concept_text, clean_text], 
            return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # features[0]: intervened, [1]: target concept, [2]: clean
        sim_interv_target = (text_features[0] @ text_features[1].T).item()
        sim_clean_target = (text_features[2] @ text_features[1].T).item()
        clip_score_increase = sim_interv_target - sim_clean_target

        # 2. Keyword Hit Rate
        hit = any(k.lower() in intervened_text.lower() for k in target_keywords)
        hit_rate = 1.0 if hit else 0.0

        # 3. Embedding Space Drift (using SBERT)
        embeddings = self.sbert_model.encode([clean_text, intervened_text])
        # Cosine distance (1 - cosine_similarity); range [0, 2]
        drift = cosine(embeddings[0], embeddings[1])

        return {
            "clip_score_target": sim_interv_target,
            "clip_score_increase": clip_score_increase,
            "keyword_hit": hit_rate,
            "embedding_drift": drift
        }

    # ============================================================
    # 3. Task-based metrics
    # ============================================================
    def evaluate_task(self, 
                      intervened_text: str, 
                      target_answers: List[str], 
                      hallucination_objects: List[str]):
        """
        Compute VQA Attack Success Rate and Object Hallucination Rate.
        """
        intervened_lower = intervened_text.lower()
        
        # 1. VQA ASR (Attack Success Rate)
        success = any(ans.lower() in intervened_lower for ans in target_answers)
        asr = 1.0 if success else 0.0

        # 2. Object Hallucination Rate
        hallucinated = any(obj.lower() in intervened_lower for obj in hallucination_objects)
        hallucination_rate = 1.0 if hallucinated else 0.0

        return {
            "vqa_asr": asr,
            "hallucination_rate": hallucination_rate
        }

    # ============================================================
    # 4. Quality and side-effect metrics (Control Metrics)
    # ============================================================
    def evaluate_quality(self, 
                         intervened_text: str, 
                         original_image):
        """
        Compute PPL and Image-Text Consistency.
        original_image: PIL Image object
        """
        # 1. Perplexity (PPL)
        encodings = self.ppl_tokenizer(intervened_text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids
        
        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        # 2. CLIP Image-Text Consistency
        inputs = self.clip_processor(
            text=[intervened_text], 
            images=original_image, 
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Manually compute cosine similarity
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            consistency = (image_embeds @ text_embeds.T).item()

        return {
            "perplexity": ppl,
            "image_consistency": consistency
        }

if __name__ == "__main__":
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    evaluator = InterventionEvaluator(device=device)
