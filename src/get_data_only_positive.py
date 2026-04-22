import os
import random
import numpy as np
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import io

# Set Hugging Face mirror site
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from PIL import Image
import io, base64, json
def img_base(image, image_id): # Add image_id parameter
    # Create img directory (if not exists)
    output_dir = "img_positive"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use image_id as filename, save as JPEG format
    filename = f"{image_id}.jpg"
    relative_path = os.path.join(output_dir, filename)
    
    # Save PIL Image object to file
    image.save(relative_path, format="JPEG", quality=90) # Recommend using JPEG format and quality parameter to reduce file size
    
    # Return relative path of the image
    return relative_path

class COCOProbeDataConstructor:
    """COCO image-text matching dataset constructor"""
    
    def __init__(
        self,
        num_images=10000,
        num_easy_neg=5000,
        bert_model_name="bert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_images = num_images
        self.num_easy_neg = num_easy_neg
        self.device = device
        
        print(f"Using device: {self.device}")
        print("Loading COCO dataset...")
        self.ds = load_dataset("jxie/coco_captions")
        
        print(f"Loading BERT model: {bert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()
        
        self.id2caps = {}
        self.id2imgs = {}
        self.selected_img_ids = []
        
    def build_mappings(self):
        """Build mappings from image_id to captions and image_id to image"""
        print("\nBuilding image_id mappings...")
        
        all_img_ids = set()
        for sample in tqdm(self.ds['train'], desc="Scanning dataset"):
            img_id = sample["cocoid"]
            all_img_ids.add(img_id)
            
            caption = sample["caption"].strip()
            image = sample["image"]
            
            # Build mappings
            if img_id not in self.id2caps:
                self.id2caps[img_id] = []
                self.id2imgs[img_id] = image  # Only keep one image instance
            
            self.id2caps[img_id].append(caption)
        
        print(f"Found {len(all_img_ids)} unique images in total")
        
        # Randomly select num_images images
        self.selected_img_ids = random.sample(list(all_img_ids), 
                                             min(self.num_images, len(all_img_ids)))
        print(f"Selected {len(self.selected_img_ids)} images for dataset construction")
        
    def compute_bert_embeddings(self):
        """Compute BERT embeddings for all positive sample captions"""
        print("\nComputing BERT embeddings...")
        
        # Collect all positive sample captions
        all_pos_captions = []
        caption_to_imgid = {}  # Track which image each caption belongs to
        
        for img_id in self.selected_img_ids:
            for cap in self.id2caps[img_id]:
                all_pos_captions.append(cap)
                caption_to_imgid[cap] = img_id
        
        print(f"Total {len(all_pos_captions)} positive sample captions")
        
        # Compute embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_pos_captions), batch_size), 
                         desc="Computing embeddings"):
                batch_captions = all_pos_captions[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_captions,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get [CLS] token embedding
                outputs = self.bert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                all_embeddings.append(cls_embeddings.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [num_captions, hidden_dim]
        
        # Normalize for cosine similarity computation
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        
        return all_pos_captions, all_embeddings.numpy(), caption_to_imgid
    
    def find_hard_negatives(self, all_captions, embeddings, caption_to_imgid):
        """Find hard negatives for each image"""
        print("\nFinding hard negatives for each image...")
        
        img_to_hard_neg = {}
        
        for img_id in tqdm(self.selected_img_ids, desc="Finding hard negatives"):
            # Get indices of all positive sample captions for this image
            pos_indices = [i for i, cap in enumerate(all_captions) 
                          if caption_to_imgid[cap] == img_id]
            
            if not pos_indices:
                continue
            
            # Compute average embedding of positive samples for this image
            pos_embeds = embeddings[pos_indices]
            avg_pos_embed = pos_embeds.mean(axis=0, keepdims=True)
            
            # Compute similarity with all captions
            similarities = (embeddings @ avg_pos_embed.T).squeeze()
            
            # Exclude positive samples, find most similar negative sample
            sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order
            
            # Find first caption that doesn't belong to current image
            for idx in sorted_indices:
                candidate_cap = all_captions[idx]
                if caption_to_imgid[candidate_cap] != img_id:
                    img_to_hard_neg[img_id] = candidate_cap
                    break
        
        print(f"Found hard negatives for {len(img_to_hard_neg)} images")
        return img_to_hard_neg
    
    def construct_pairs(self, img_to_hard_neg):
        """Construct final image-text pair data - only generate positive samples"""
        print("\nConstructing image-text pairs...")
        
        pairs = []
        
        for img_id in tqdm(self.selected_img_ids, desc="Constructing sample pairs"):
            image = self.id2imgs[img_id]
            pos_caps = self.id2caps[img_id]
            
            # Positive sample: randomly select a matching caption
            selected_pos_cap = random.choice(pos_caps)
            pairs.append({
                "image_id": img_id,
                "image": image,
                "caption": selected_pos_cap,
                "label": 1,
                "type": "positive"
            })
        
        return pairs
    
    def save_pairs(self, pairs, output_file="coco_probe_pairs_positive.jsonl"):
        """Save data pairs to file"""
        print(f"\nSaving data to {output_file}...")
        
        # Statistics
        num_pos = sum(1 for p in pairs if p["label"] == 1)
        
        print(f"Data statistics:")
        print(f"  Positive samples: {num_pos}")
        print(f"  Total: {len(pairs)}")
        
        # Need to process image objects before saving
        pairs_to_save = []
        for pair in tqdm(pairs, desc="Processing image data"):
            pair_copy = pair.copy()
            pair_copy["image"] = img_base(pair["image"],pair["image_id"])
            pairs_to_save.append(pair_copy)
        
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(pairs_to_save)
        
        print(f"Data saved to {output_file}")
        
        return pairs  # Return version with image objects for training
    
    def run(self, output_file="coco_probe_pairs_positive.jsonl"):
        """Run the complete data construction pipeline"""
        # 1. Build mappings
        self.build_mappings()
        
        # 2. Compute BERT embeddings - can be skipped for positive-only generation, but kept for future extension
        all_captions, embeddings, caption_to_imgid = self.compute_bert_embeddings()
        
        # 3. Find hard negatives - can be skipped for positive-only generation, but kept for future extension
        img_to_hard_neg = self.find_hard_negatives(
            all_captions, embeddings, caption_to_imgid
        )
        
        # 4. Construct sample pairs - only construct positive samples
        pairs = self.construct_pairs(img_to_hard_neg)
        
        # 5. Save data
        final_pairs = self.save_pairs(pairs, output_file)
        
        print("\nData construction completed!")
        return final_pairs

def main():
    """Main function"""
    constructor = COCOProbeDataConstructor(
        num_images=50000,
        num_easy_neg=0,  # Set to 0, do not generate easy negatives
        bert_model_name="bert-base-uncased"
    )
    
    pairs = constructor.run(output_file="coco_probe_pairs_positive.jsonl")
    
    # Display some sample examples
    print("\nSample examples:")
    for i, pair in enumerate(pairs[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Image ID: {pair['image_id']}")
        print(f"  Caption: {pair['caption'][:100]}...")
        print(f"  Label: {pair['label']} ({pair.get('type', 'unknown')})")


if __name__ == "__main__":
    main()