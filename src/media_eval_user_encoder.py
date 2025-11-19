                       
""""""

import os, csv, json, argparse, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BertModel,
    BertTokenizer,
)
from peft import PeftModel
from tqdm import tqdm


                         
                      
                         
def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def masked_mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

def bert_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "cls":
        return last_hidden[:, 0, :]
    elif mode == "last":
        lengths = attn_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden.size(-1))
        return last_hidden.gather(1, idx).squeeze(1)
    else:
        return masked_mean_pool(last_hidden, attn_mask)

                                            
                                               
                                               
                                 
                                                                 
                        
                                                    
                                                      
                               
                                                        
                                                
                               
                                                
                                                                                  
                                                                                                                
                                                                                               
                                                                          
                                                 
           
                                            
                                                                                               
                                                 

                                                                                                           
    if mode == "mean":
        return masked_mean_pool(hidden, attn_mask)
    else:
        lengths = attn_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

def _split_triple(s: str) -> Union[Tuple[str, str, str], None]:
    if not isinstance(s, str): return None
    row = next(csv.reader([s], skipinitialspace=True))
    if len(row) >= 3:
        dom, sub, val = row[0].strip(), row[1].strip(), ",".join(row[2:]).strip()
        return dom, sub, val
    parts = [x.strip() for x in s.split(',', 2)]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


                         
                           
                         
def format_preferences_naturally(lst: List[str], exclude_idx: int = -1) -> str:
    sents = []
    for i, p in enumerate(lst):
        if i == exclude_idx:
            continue
        triple = _split_triple(p)
        if not triple:
            continue
        domain, sub, val = triple
        sub_l = sub.lower()
        if "genre" in sub_l:
            sents.append(f"prefers {val} genres")
        elif "actor" in sub_l or "director" in sub_l:
            sents.append(f"enjoys works by {val}")
        elif "favorite media" in sub_l or "favourite media" in sub_l:
                                                   
            sents.append(f"favorite titles include {val}")
        else:
            sents.append(f"interested in {val}")
    return ". ".join(sents)

def format_target_query(domain: str, sub: str) -> str:
                             
    return "Recommend images related to movies and TV shows"


                         
                        
                         
def load_test_rows(filepath: str,
                   target_policy: str = "all",
                   allowed_subs: Union[List[str], None] = None,
                   skip_on_mismatch: bool = False) -> List[Dict[str, Any]]:
    """"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    allowed = set([s.strip().lower() for s in (allowed_subs or [])])
    out: List[Dict[str, Any]] = []

    def add_row_core(uid: str, triples: List[str], t_idx: int):
        parsed = _split_triple(triples[t_idx])
        if not parsed: 
            return
        domain, sub, val = parsed
        prefs = format_preferences_naturally(triples, exclude_idx=t_idx)                   
        if not prefs:
            return                           
        out.append({
            "user_id": uid,
            "demographic_and_preferences": prefs.strip(),
            "target_query": format_target_query(domain, sub),
            "gt": f"{val} (Movie or TV series)",
            "domain": domain,
            "subdomain": sub
        })

    for sample in data:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        meta = sample[0]
        uid = str(meta.get("user_id", ""))
        triples = list(sample[1:])

        if target_policy == "all":
            for t_idx, _ in enumerate(triples):
                add_row_core(uid, triples, t_idx)

        elif target_policy == "last":
            t_idx = len(triples) - 1
            add_row_core(uid, triples, t_idx)

        elif target_policy == "filter_last":
            t_idx = len(triples) - 1
            parsed = _split_triple(triples[t_idx])
            if not parsed: 
                continue
            _, sub, _ = parsed
            if sub.strip().lower() in allowed:
                add_row_core(uid, triples, t_idx)
            else:
                if not skip_on_mismatch:
                    add_row_core(uid, triples, t_idx)
        else:
            raise ValueError(f"Unknown target_policy: {target_policy}")

    return out


                         
                              
                         
                                  
                                                                                                
                            
                                
                                        
                                            
                            
                                       
                                                         
                                                    
               
               
                                                                
                                                           
                                                     
                              
                                      

                           
                                                    
                                                                

class InferenceModel(nn.Module):
    def __init__(self,
                 e5v="royokong/e5-v",
                 bert="bert-base-uncased",
                 bert_pool_mode="mean",
                 prefix_pos="after_bos",
                                       
                 proj_type="linear",
                                
                 temp_init=2.6592,
                 device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        self.bert = BertModel.from_pretrained(bert)

        self.llm = LlavaNextForConditionalGeneration.from_pretrained(e5v, torch_dtype=torch.float16)
        for p in self.llm.parameters(): p.requires_grad = False

        self.h_bert = self.bert.config.hidden_size
        self.h_llm  = self.llm.config.text_config.hidden_size


                                                                                                         
                                                               
        self.proj = nn.Linear(self.h_bert, self.h_llm)
        self.proj_norm = nn.LayerNorm(self.h_llm)

        self.bert_pool_mode = bert_pool_mode
        self.prefix_pos     = prefix_pos
                                            
        self.num_prefix     = 1
        self.bos_id         = getattr(self.llm.config.text_config, "bos_token_id", 128000)
        self.eos_id         = getattr(self.llm.config.text_config, "eos_token_id", 128009)
        self.logit_scale    = nn.Parameter(torch.ones([]) * temp_init)

        self.to(self.device)

    def tok_embeds(self, input_ids):
                                      
                                         
                               
        return self.llm.language_model.model.embed_tokens(input_ids)

    def pool_ids(self, input_ids, attn_mask):
        out = self.llm(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True, return_dict=True)
        hid = out.hidden_states[-1]
                                 
        emb = hid[:, -1, :]                
                                                                                           
        return F.normalize(emb, p=2, dim=1)

    def pool_embeds(self, embeds, attn_mask):
        lm = self.llm.language_model
        
                                                
        embeds = embeds.to(self.llm.dtype) 
        attn_mask = attn_mask.to(self.device)                


        out = lm(inputs_embeds=embeds, attention_mask=attn_mask,
                 output_hidden_states=True, return_dict=True)
        hid = out.hidden_states[-1]
                                 
        emb = hid[:, -1, :]                
        return F.normalize(emb, p=2, dim=1)

                                                                             
                                             

    @torch.no_grad()
    def encode_query(self, demo_input_ids, demo_attn, target_query_input_ids, target_query_attn):
        bert_out = self.bert(input_ids=demo_input_ids, attention_mask=demo_attn, return_dict=True)
        pooled = bert_pool(bert_out.last_hidden_state, demo_attn, self.bert_pool_mode)
                                                    

                                           
        projected = self.proj(pooled)
        projected = self.proj_norm(projected)
        soft_prefix = projected.unsqueeze(1)             

        q_emb = self.tok_embeds(target_query_input_ids)

        B = demo_input_ids.size(0)
        if self.prefix_pos == "after_bos":
            bos, rest = q_emb[:, :1, :], q_emb[:, 1:, :]
            bos_m, rest_m = target_query_attn[:, :1], target_query_attn[:, 1:]
            ones = torch.ones((B, self.num_prefix), dtype=target_query_attn.dtype, device=target_query_attn.device)
            comb  = torch.cat([bos, soft_prefix, rest],  dim=1)
            cmask = torch.cat([bos_m, ones,        rest_m], dim=1)
        else:
            ones  = torch.ones((B, self.num_prefix), dtype=target_query_attn.dtype, device=target_query_attn.device)
            comb  = torch.cat([soft_prefix, q_emb], dim=1)
            cmask = torch.cat([ones,        target_query_attn], dim=1)

        left = self.pool_embeds(comb, cmask)                    
        return left


                         
            
                         
def load_image_embeddings(embedding_dir: str, device: torch.device):
    files = sorted(Path(embedding_dir).glob("*.npy"))
    names, embs = [], []
    for f in tqdm(files, desc="Load image embeddings"):
        names.append(f.stem)
        arr = np.load(str(f))
        embs.append(torch.from_numpy(arr))
    if not embs:
        raise RuntimeError(f"No .npy found in {embedding_dir}")
    E = torch.stack(embs, dim=0).to(device)
    E = F.normalize(E, p=2, dim=1)
    return names, E

@torch.no_grad()
def retrieve_top_k(query_emb: torch.Tensor, img_names: List[str], img_mat: torch.Tensor, k: int = 10):
    query_emb = query_emb.to(dtype=img_mat.dtype, device=img_mat.device)
    sims = torch.matmul(img_mat, query_emb)        
    topk = torch.topk(sims, k=min(k, img_mat.size(0)))
    return [(img_names[i], sims[i].item()) for i in topk.indices.tolist()]


                         
    
                         
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_tok = BertTokenizer.from_pretrained(args.bert_model_name)
    processor = LlavaNextProcessor.from_pretrained(args.model_name)

    llama_tpl = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    fmt = "{}\nSummary above sentence in one word: "
                                          

           
    model = InferenceModel(
        e5v=args.model_name,
        bert=args.bert_model_name,
        bert_pool_mode=args.bert_pool,
        prefix_pos=args.prefix_pos,
                                    
        proj_type=args.proj_type,
                                     
        temp_init=args.temp_init,
        device=device
    )
    model.eval()

                          
    if not Path(args.checkpoint_dir).exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {args.checkpoint_dir}")
                      
    model.bert = PeftModel.from_pretrained(model.bert, args.checkpoint_dir)
    model.bert.eval().to(device)

                              
    add_p = Path(args.checkpoint_dir) / "additional_params.pt"
    if not add_p.exists():
        raise FileNotFoundError(f"additional_params.pt not found in {args.checkpoint_dir}")
    blob = torch.load(str(add_p), map_location=device)
    model.proj.load_state_dict(blob["projection"])
                           
                                                                 
                                
                                          
    if "proj_norm" in blob:
        model.proj_norm.load_state_dict(blob["proj_norm"])
    else:
        print("[Warning] 'proj_norm' not found in checkpoint. Using initialized weights.")
        
    with torch.no_grad():
        model.logit_scale.data = blob["logit_scale"].to(device)

                                        
    allowed = args.allowed_subs.split(";") if args.allowed_subs else None
    test_rows = load_test_rows(
        args.test_file,
        target_policy=args.target_policy,
        allowed_subs=allowed,
        skip_on_mismatch=args.skip_on_mismatch
    )
    print(f"[DATA] test rows: {len(test_rows)}")

                         
    img_names, img_mat = load_image_embeddings(args.image_embedding_dir, device)
    print(f"[IMG] embeddings: {len(img_names)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.image_source_dir:
        Path(args.image_output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    @torch.no_grad()
    def encode_texts(texts: List[str]):
        if args.use_plain_text:
            q_inputs = processor.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True, max_length=256
            )
        else:
            wrapped = [llama_tpl.format(fmt.format(t)) for t in texts]
            q_inputs = processor(wrapped, return_tensors='pt', padding=True)
        return q_inputs.input_ids.to(device), q_inputs.attention_mask.to(device)

    BS = args.batch_size
    for i in tqdm(range(0, len(test_rows), BS), desc="Inference"):
        chunk = test_rows[i:i+BS]
        demos   = [r["demographic_and_preferences"] for r in chunk]
        queries = [r["target_query"] for r in chunk]
        gts     = [r["gt"] for r in chunk]

        demo_inputs = bert_tok(demos, return_tensors='pt', padding=True, truncation=True, max_length=512)
        demo_ids  = demo_inputs.input_ids.to(device)
        demo_attn = demo_inputs.attention_mask.to(device)

                          
        q_ids, q_attn = encode_texts(queries)

        q_embs = model.encode_query(demo_ids, demo_attn, q_ids, q_attn)                    

                              
        gt_sims = []
        if args.eval_gt_similarity:
            gt_ids, gt_attn = encode_texts(gts)
            if args.gt_path == "ids":
                gt_embs = model.pool_ids(gt_ids, gt_attn)
            else:
                gt_tok_emb = model.tok_embeds(gt_ids)
                gt_embs = model.pool_embeds(gt_tok_emb, gt_attn)
            gt_sims = F.cosine_similarity(q_embs, gt_embs, dim=1).tolist()
        else:
            gt_sims = [None] * len(chunk)

                   
        for j, (qe, row, gt_sim) in enumerate(zip(q_embs, chunk, gt_sims)):
            topk = retrieve_top_k(qe, img_names, img_mat, k=args.top_k)
            res = {
                "index": i + j,
                "user_id": row.get("user_id", ""),
                "demographic_and_preferences": row["demographic_and_preferences"],
                "target_query": row["target_query"],
                "domain": row["domain"],
                "subdomain": row["subdomain"],
                "ground_truth": row["gt"],
                "gt_text_similarity": gt_sim,
                "top_k_results": [
                    {"rank": rank, "image_name": name, "similarity": sim}
                    for rank, (name, sim) in enumerate(topk, 1)
                ],
                "top_1_match": (len(topk) > 0 and topk[0][0] == row["gt"]),
                "gt_in_top_k": any(name == row["gt"] for name, _ in topk),
            }
            results.append(res)

                        
            if args.image_source_dir:
                out_dir = Path(args.image_output_dir) / f"sample_{i+j:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for rank, (name, sim) in enumerate(topk, 1):
                    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                        src = Path(args.image_source_dir) / f"{name}{ext}"
                        if src.exists():
                            dst = out_dir / f"rank{rank:02d}_sim{sim:.4f}_{name}{ext}"
                            shutil.copy(str(src), str(dst))
                            break

           
    out_file = Path(args.output_dir) / "retrieval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_file}")

    top1 = sum(r["top_1_match"] for r in results)
    topk = sum(r["gt_in_top_k"] for r in results)
    n = len(results)
    print(f"[STATS] Total={n} | Top-1={top1/n:.4f} ({top1}/{n}) | Top-{args.top_k}={topk/n:.4f} ({topk}/{n})")

    stats = {
        "total_samples": n,
        "top_1_accuracy": top1 / n if n else 0.0,
        f"top_{args.top_k}_accuracy": topk / n if n else 0.0,
        "top_1_correct": top1,
        f"top_{args.top_k}_correct": topk
    }
    with open(Path(args.output_dir) / "evaluation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVE] {Path(args.output_dir) / 'evaluation_stats.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
        
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--image_embedding_dir", type=str, required=True)
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="")
        
    ap.add_argument("--model_name", type=str, default="royokong/e5-v")
    ap.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--output_dir", type=str, default="./inference_output")
    ap.add_argument("--image_source_dir", type=str, default=None)
    ap.add_argument("--image_output_dir", type=str, default="./inference_output/top_k_images")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

               
    ap.add_argument("--bert_pool", type=str, default="mean", choices=["cls","mean","last"])
    ap.add_argument("--prefix_pos", type=str, default="after_bos", choices=["prepend","after_bos"])
                                                                                           
    ap.add_argument("--proj_type", type=str, default="mlp", choices=["linear","mlp"])
                                                          
    ap.add_argument("--temp_init", type=float, default=2.6592)
    ap.add_argument("--gt_path", type=str, default="embeds", choices=["ids","embeds"])
    ap.add_argument("--use_plain_text", action="store_true")
    ap.add_argument("--eval_gt_similarity", action="store_true")

                            
    ap.add_argument("--target_policy", type=str, default="all", choices=["all","last","filter_last"])
    ap.add_argument("--allowed_subs", type=str,
                    default="favorite media;favourite media;favorite actors and directors;favourite actors and directors")
    ap.add_argument("--skip_on_mismatch", action="store_true")

    args = ap.parse_args()
    main(args)
