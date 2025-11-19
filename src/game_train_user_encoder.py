                       
""""""

import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import json
import csv
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BertModel,
    BertTokenizer,
           
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from typing import Optional

print("DEBUG A: Script started and imports complete")


                         
       
                         
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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



                         
                 
                         
def format_demographics_naturally(d: dict) -> str:
    if not isinstance(d, dict): return ""
    age = d.get('age')
    if not age or age == 'N/A': return ""
    parts = []
    ordered_keys = ['gender', 'employment_status', 'marital_status']
    processed = {'user_id', 'age'}
    for k in ordered_keys:
        v = d.get(k)
        if v and v != 'N/A':
            if k == 'gender':
                parts.append(v.lower())
            elif k == 'employment_status':
                parts.append("currently unemployed and seeking work" if "Unemployed" in v else v.lower())
            else:
                parts.append(v.lower())
            processed.add(k)
    for k, v in d.items():
        if k in processed or not v or v == 'N/A': continue
        parts.append(f"{k.replace('_',' ')} {v}")
    pref = f"The user is {age}"
    if not parts: return f"{pref}."
    if len(parts) == 1: return f"{pref} and {parts[0]}."
    return f"{pref}, {', '.join(parts[:-1])}, and {parts[-1]}."



def _split_triple(s: str) -> Optional[Tuple[str, str, str]]:
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
        d = (domain or "").strip().lower()
        s = (sub or "").strip().lower()
        v = (val or "").strip()
        if not v:
            continue

                             
                                                                                           
                      

                 
                                
                                                                
        if "favorite media" in s or "favourite media" in s:
            continue
        if "preferred game name" in s:
                                      
            sents.append(f"likes the video game title {v}")
            continue

        if d == "games":
            if "genre" in s:
                sents.append(f"prefers {v} genre")
            elif "multiplayer" in s:
                sents.append(f"prefers {v} mode")
            elif "frequency" in s:
                sents.append(f"usually plays {v}")
            else:
                sents.append(f"interested in {v}")
        elif d == "media":
            if "genre" in s:
                sents.append(f"prefers {v} genre")
            elif "actor" in s or "director" in s:
                sents.append(f"likes works by {v}")
            else:
                sents.append(f"interested in {v}")
        else:
            if "genre" in s:
                sents.append(f"prefers {v} genre")
            else:
                sents.append(f"interested in {v}")

           
    uniq, seen = [], set()
    for x in sents:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return ". ".join(uniq[:100])


def format_target_query(domain: str, sub: str) -> str:
    """"""
    return "Recommend images related to video games"


                                                                 
                                                      
                             

                                    
                         
                                                                      
                      
                          
                                                                
                                    
                                             
                                       
                            
                          
                                       
                     
                                                                              
                           
                                                                    
                          
                          
                                 
                                                               
                                                                   
                                                    
                
                

def load_and_parse_data(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out: List[Dict[str, str]] = []
    for sample in data:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            continue
        demo = sample[0]
        uid = str(demo.get("user_id", ""))                    
        triples = list(sample[1:])
        
        for t_idx, t in enumerate(triples):
            gt_value = None
            domain, sub, val = "", "", ""
            
            if isinstance(t, dict):
                                                      
                if "originalGT" in t and "templateGT" in t:
                    gt_dict = t
                    
                                                                      
                    original_gt_str = gt_dict["originalGT"]
                    parsed = _split_triple(original_gt_str)
                    if parsed:
                        domain, sub, val = parsed
                    
                                                   
                    gt_value = gt_dict["templateGT"]
                else:
                    continue
            else:
                                     
                parsed = _split_triple(t)
                if not parsed:
                    continue
                domain, sub, val = parsed
                                
                gt_value = val
            
                                                  
            if not val or not gt_value:
                continue

            prefs = format_preferences_naturally(triples, exclude_idx=t_idx)
            if not prefs:
                                                                  
                continue
            out.append({
                "user_id": uid,
                "demographic_and_preferences": prefs.strip(),
                "target_query": format_target_query(domain, sub),
                                      
                "gt": gt_value 
            })
    return out


class PersonalizedDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


                         
       
                         
class Model(nn.Module):
    def __init__(self,
                 e5v="royokong/e5-v",
                 bert="bert-base-uncased",
                 use_lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 bert_pool_mode="mean", prefix_pos="after_bos",
                 proj_type="mlp", num_prefix=1,                                
                 temp_init=2.6592):
        super().__init__()

        print("DEBUG 1: Model __init__ started, loading BERT...")

                                         
        self.bert = BertModel.from_pretrained(bert)
        if use_lora:
            self.bert = get_peft_model(self.bert, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, target_modules=["query","value"], bias="none"
            ))

        print("DEBUG 2: BERT loaded, loading E5-V...")


                       
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

                                                          
    def forward(self,
                demographic_preferences_input_ids: torch.Tensor,
                demographic_preferences_attention_mask: torch.Tensor,
                target_query_input_ids: torch.Tensor,
                target_query_attention_mask: torch.Tensor,
                gt_input_ids: torch.Tensor,
                gt_attention_mask: torch.Tensor,
                gt_path: str = "embeds"):
        B = demographic_preferences_input_ids.size(0)

                                                                 
        bert_out = self.bert(input_ids=demographic_preferences_input_ids,
                             attention_mask=demographic_preferences_attention_mask,
                             return_dict=True)
        pooled = bert_pool(bert_out.last_hidden_state,
                           demographic_preferences_attention_mask,
                           self.bert_pool_mode)        

        
                                                                       

                                      
        projected = self.proj(pooled)
                                    
        projected = self.proj_norm(projected)
                                                    
        soft_prefix = projected.unsqueeze(1)

        q_emb = self.llm.language_model.model.embed_tokens(target_query_input_ids)

                         
        if self.prefix_pos == "after_bos":
            bos, rest = q_emb[:, :1, :], q_emb[:, 1:, :]
            bos_m, rest_m = target_query_attention_mask[:, :1], target_query_attention_mask[:, 1:]
            ones = torch.ones((B, self.num_prefix), dtype=target_query_attention_mask.dtype, device=target_query_attention_mask.device)
            comb  = torch.cat([bos, soft_prefix, rest],  dim=1)
            cmask = torch.cat([bos_m, ones,        rest_m], dim=1)
        else:
            ones  = torch.ones((B, self.num_prefix), dtype=target_query_attention_mask.dtype, device=target_query_attention_mask.device)
            comb  = torch.cat([soft_prefix, q_emb], dim=1)
            cmask = torch.cat([ones,        target_query_attention_mask], dim=1)

        out_left = self.llm.language_model(inputs_embeds=comb.to(self.llm.dtype), 
                                           attention_mask=cmask,
                                           output_hidden_states=True, 
                                           return_dict=True)       
        hid_left = out_left.hidden_states[-1]            
        emb_left = hid_left[:, -1, :]                
        left = F.normalize(emb_left, p=2, dim=1)

                                                             
        if gt_path == "ids":
                                                 
            out_right = self.llm(input_ids=gt_input_ids, 
                                 attention_mask=gt_attention_mask,
                                 output_hidden_states=True, 
                                 return_dict=True)
            hid_right = out_right.hidden_states[-1]            
            emb_right = hid_right[:, -1, :]               
            right = F.normalize(emb_right, p=2, dim=1)
        else:
                                                                      
            gt_emb = self.llm.language_model.model.embed_tokens(gt_input_ids)
                                                                         
            out_right = self.llm.language_model(inputs_embeds=gt_emb.to(self.llm.dtype), 
                                                attention_mask=gt_attention_mask,
                                                output_hidden_states=True, 
                                                return_dict=True)
            hid_right = out_right.hidden_states[-1]            
            emb_right = hid_right[:, -1, :]               
            right = F.normalize(emb_right, p=2, dim=1)

                                                                      
        scale = self.logit_scale.clamp(max=5).exp()
        return left, right, scale


                         
                             
                         
def _uid_to_int(u: str) -> int:
    return abs(hash(u)) % (2**31 - 1)

def build_same_user_mask(global_user_ids: torch.Tensor) -> torch.Tensor:
    eq = global_user_ids.view(-1,1) == global_user_ids.view(1,-1)         
    eye = torch.eye(eq.size(0), dtype=torch.bool, device=eq.device)
    same_user_offdiag = eq & (~eye)
    return same_user_offdiag                                

      
def compute_cosine_stats_masked(left_g: torch.Tensor, right_g: torch.Tensor, rs: int, re: int, neg_mask: Optional[torch.Tensor] = None):
    sims = torch.matmul(left_g, right_g.t())         
    rows = sims[rs:re]                               
    B = re - rs
    diag = torch.arange(rs, re, device=left_g.device)

    if neg_mask is not None:
                                                        
        rows = rows.masked_fill(neg_mask[rs:re], float('-inf'))

    pos = rows[torch.arange(B, device=left_g.device), diag]
    mask_eye = torch.ones_like(rows, dtype=torch.bool)
    mask_eye[torch.arange(B, device=left_g.device), diag] = False
    hardneg = rows.masked_fill(~mask_eye, float('-inf')).max(1).values
    return pos.mean().item(), hardneg.mean().item(), (pos - hardneg).mean().item(), sims

def margin_loss_from_sims_masked(sims: torch.Tensor, rs: int, re: int, margin: float = 0.15, topk: int = 5, neg_mask: Optional[torch.Tensor] = None):
    losses = []
    for i in range(rs, re):
        row = sims[i]        
        pos = row[i]
        mask = torch.zeros_like(row, dtype=torch.bool)
        mask[i] = True
        if neg_mask is not None:
            mask = mask | neg_mask[i]
        negs = row.masked_fill(mask, float('-inf'))
                               
        valid = torch.isfinite(negs)
        if not valid.any():
            continue
        k = min(topk, int(valid.sum().item()))
        topk_vals, _ = torch.topk(negs, k=k)
        loss_i = F.relu(margin - pos + topk_vals).mean()
        losses.append(loss_i)
    if not losses:
        return sims.new_tensor(0.0)
    return torch.stack(losses).mean()

def compute_inbatch_ranking_metrics_masked(logits: torch.Tensor, rs: int, re: int, neg_mask: Optional[torch.Tensor], ks: List[int]):
    """"""
    device = logits.device
    rows = logits[rs:re].clone()              
    B = re - rs
    labels = torch.arange(rs, re, device=device)

    if neg_mask is not None:
        rows = rows.masked_fill(neg_mask[rs:re], float('-inf'))

    sorted_idx = torch.argsort(rows, dim=1, descending=True)             
    match = (sorted_idx == labels.unsqueeze(1))                                            
                                                  
                                         
    BIG = rows.size(1) + 1
                           
    pos_rank = torch.where(match.any(dim=1),
                           torch.argmax(match.to(torch.int32), dim=1),
                           torch.full((B,), BIG, device=device))

    out = {}
    for k in ks:
        hitk = (pos_rank < k).float().mean().item()
        out[f"hits@{k}"] = hitk
        out[f"recall@{k}"] = hitk
                                   
    mrr = torch.where(pos_rank < BIG, 1.0 / (pos_rank.float() + 1.0), torch.zeros_like(pos_rank, dtype=torch.float)).mean().item()
    out["mrr"] = mrr
    return out


                         
                  
                         
def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                    acc: Accelerator, args) -> float:
    model.train(); total = 0.0
    ks = [int(x) for x in args.topk.split(",")]
    bar = tqdm(loader, desc="Train", leave=False, disable=not acc.is_local_main_process)

    for batch in bar:
        if batch is None: continue
        opt.zero_grad()

        left, right, scale = model(**{k:v for k,v in batch.items() if k != "user_ids"}, gt_path=args.gt_path)

        with torch.no_grad():
            L = acc.gather(left.contiguous())
            R = acc.gather(right.contiguous())
            U = acc.gather(batch["user_ids"].to(acc.device).contiguous())

        bs = batch['demographic_preferences_input_ids'].size(0)
        rs, re = acc.process_index * bs, acc.process_index * bs + bs

        L = L.clone(); R = R.clone()
        L[rs:re] = left; R[rs:re] = right

        same_user_mask = build_same_user_mask(U)

        logits = torch.matmul(L, R.t()) * scale
        logits = logits.masked_fill(same_user_mask, float('-inf'))

        labels = torch.arange(rs, re, device=acc.device)

        ce = F.cross_entropy(logits[rs:re], labels)
        loss = ce

        pos, neg, mar, sims = compute_cosine_stats_masked(L, R, rs, re, neg_mask=same_user_mask)
        if args.use_margin:
            mloss = margin_loss_from_sims_masked(sims, rs, re, margin=args.margin, topk=args.hn_topk, neg_mask=same_user_mask)
            loss = loss + args.hn_weight * mloss

        acc.backward(loss); opt.step()
        total += loss.item()

        if acc.is_main_process:
                                                 
            m = compute_inbatch_ranking_metrics_masked(logits, rs, re, same_user_mask, ks)
            logit_scale_clamped = torch.log(scale).detach().item()
            wandb.log({
                "train_loss": loss.item(), "train_ce": ce.item(),
                "train_mean_pos_cos": pos, "train_mean_hardneg_cos": neg, "train_margin": mar,
                "logit_scale_clamped": logit_scale_clamped, "temperature": 1.0 / scale.item(),
                **{f"train_inbatch_{k}": v for k, v in m.items()}
            })
        bar.set_postfix(loss=loss.item())
    return total / (len(loader) or 1)


def validate(model: nn.Module, loader: DataLoader, acc: Accelerator, args):
    model.eval(); total = 0.0; steps = 0
    ks = [int(x) for x in args.topk.split(",")]
    agg = {**{f"hits@{k}":0.0 for k in ks}, **{f"recall@{k}":0.0 for k in ks}, "mrr":0.0}
    bar = tqdm(loader, desc="Val", leave=False, disable=not acc.is_local_main_process)

    with torch.no_grad():
        for batch in bar:
            if batch is None: continue
            left, right, scale = model(**{k:v for k,v in batch.items() if k != "user_ids"}, gt_path=args.gt_path)

            L = acc.gather(left.contiguous()); R = acc.gather(right.contiguous())
            U = acc.gather(batch["user_ids"].to(acc.device).contiguous())

            bs = batch['demographic_preferences_input_ids'].size(0)
            rs, re = acc.process_index * bs, acc.process_index * bs + bs
            L = L.clone(); R = R.clone(); L[rs:re] = left; R[rs:re] = right

            same_user_mask = build_same_user_mask(U)

            logits = torch.matmul(L, R.t()) * scale
            logits = logits.masked_fill(same_user_mask, float('-inf'))

            labels = torch.arange(rs, re, device=acc.device)
            ce = F.cross_entropy(logits[rs:re], labels)
            total += ce.item(); steps += 1

            m = compute_inbatch_ranking_metrics_masked(logits, rs, re, same_user_mask, ks)
            for k, v in m.items():
                agg[k] += v

            if acc.is_main_process:
                wandb.log({"val_step_loss": ce.item(),
                           **{f"val_inbatch_{k}": v for k, v in m.items()}})
            bar.set_postfix(loss=ce.item())

    avg_ce = total / max(steps, 1)
    for k in agg: agg[k] /= max(steps, 1)
    hits1 = agg.get("hits@1", 0.0)

    if acc.is_main_process:
        wandb.log({"val_ce": avg_ce, **{f"val_{k}": v for k, v in agg.items()}})
    return avg_ce, hits1


                         
      
                         
def main(args):

    print("DEBUG F: Main function entered")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    acc = Accelerator(kwargs_handlers=[ddp_kwargs])

    print("DEBUG G: Accelerator initialized")

    set_seed(args.seed)

    if acc.is_main_process:
        wandb.init(project=args.wandb_project, entity="VAI_Lab", name=args.wandb_run_name, config=vars(args))

    per_dev = max(1, args.batch_size // max(1, acc.num_processes))
    if acc.is_main_process:
        print(f"\nGPUs:{acc.num_processes} total_bs:{args.batch_size} per_dev:{per_dev}\n")

    print("DEBUG H: Starting Model() creation")

    model = Model(
        e5v=args.model_name, bert=args.bert_model_name,
        use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bert_pool_mode=args.bert_pool, prefix_pos=args.prefix_pos,
        proj_type=args.proj_type, num_prefix=args.num_prefix,
        temp_init=args.temp_init
    )

    if acc.is_main_process:
        wandb.watch(model, log="all", log_freq=100)
    
    print("DEBUG 3: Model loaded, loading tokenizers/processors...")

                             
    bert_tok = BertTokenizer.from_pretrained(args.bert_model_name)
    processor = LlavaNextProcessor.from_pretrained(args.model_name)
    llama_tpl = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    fmt = "{}\nSummary above sentence in one word:"
                                          

          
    acc.print(f"Loading {args.train_file} ...")

    print("DEBUG 4: Tokenizers loaded, calling load_and_parse_data...")

    train_data = load_and_parse_data(args.train_file)
    acc.print(f"Loading {args.val_file} ...")
    val_data = load_and_parse_data(args.val_file)

    print("DEBUG 5: Data loaded, setting up DataLoader...")

    train_ds = PersonalizedDataset(train_data)
    val_ds   = PersonalizedDataset(val_data)

    def collate(batch: List[Dict[str, str]]) -> Optional[Dict[str, torch.Tensor]]:
        valid = [x for x in batch if all([x.get("demographic_and_preferences"), x.get("target_query"), x.get("gt")])]
        if not valid: return None

        demo = [x['demographic_and_preferences'] for x in valid]
        demo_inputs = bert_tok(demo, return_tensors='pt', padding=True, truncation=True, max_length=256)

        if args.use_plain_text:
            q_texts  = [x['target_query'] for x in valid]
            gt_texts = [x['gt'] for x in valid]
            q_inputs  = processor.tokenizer(q_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            gt_inputs = processor.tokenizer(gt_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        else:
            q_texts  = [llama_tpl.format(fmt.format(x['target_query'])) for x in valid]
            gt_texts = [llama_tpl.format(fmt.format(x['gt'])) for x in valid]
            q_inputs  = processor(text=q_texts,  return_tensors='pt', padding=True)
            gt_inputs = processor(text=gt_texts, return_tensors='pt', padding=True)

        user_ids = torch.tensor([_uid_to_int(x['user_id']) for x in valid], dtype=torch.long)

        return {
            "demographic_preferences_input_ids": demo_inputs.input_ids,
            "demographic_preferences_attention_mask": demo_inputs.attention_mask,
            "target_query_input_ids": q_inputs.input_ids,
            "target_query_attention_mask": q_inputs.attention_mask,
            "gt_input_ids": gt_inputs.input_ids,
            "gt_attention_mask": gt_inputs.attention_mask,
            "user_ids": user_ids,
        }

    train_loader = DataLoader(train_ds, batch_size=per_dev, shuffle=True,  collate_fn=collate, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=per_dev, shuffle=False, collate_fn=collate, drop_last=True)

    logit_scale_params = [p for n, p in model.named_parameters() if "logit_scale" in n]
    other_params = [p for n, p in model.named_parameters() if "logit_scale" not in n]

    opt = AdamW([
        {"params": other_params, "lr": args.learning_rate},
        {"params": logit_scale_params, "lr": args.learning_rate * 0.05, "weight_decay": 0.0}
    ])
    os.makedirs(args.output_dir, exist_ok=True)

    model, opt, train_loader, val_loader = acc.prepare(model, opt, train_loader, val_loader)

    best = float('inf')
    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, opt, acc, args)
        val_loss, val_hits1 = validate(model, val_loader, acc, args)
        if acc.is_main_process:
            print(f"[Epoch {ep+1}] train {train_loss:.4f} | val CE {val_loss:.4f} | in-batch Hits@1 {val_hits1:.4f}")
            wandb.log({"epoch": ep+1, "train_epoch_loss": train_loss, "val_ce": val_loss, "val_inbatch_hits@1": val_hits1})

            if val_loss < best:
                best = val_loss
                unwrapped = acc.unwrap_model(model)
                save_dir = os.path.join(
                    args.output_dir,
                    f"best_ep{ep+1}_lr{args.learning_rate}_proj{args.proj_type}_P{args.num_prefix}_path{args.gt_path}{'_plain' if args.use_plain_text else ''}"
                )
                os.makedirs(save_dir, exist_ok=True)
                if args.use_lora:
                    unwrapped.bert.save_pretrained(save_dir)
                torch.save({
                    "projection": unwrapped.proj.state_dict(),
                    "proj_norm": unwrapped.proj_norm.state_dict(),
                    "logit_scale": unwrapped.logit_scale.detach().cpu()
                }, os.path.join(save_dir, "additional_params.pt"))
                print(f"[BEST] saved to {save_dir}")

    if acc.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--model_name", type=str, default="royokong/e5-v")
    p.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="./model_output")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)                            
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="bert_e5v_niihau")
    p.add_argument("--wandb_run_name", type=str, default=None)

          
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)

             
    p.add_argument("--bert_pool", type=str, default="mean", choices=["cls","mean","last"])
    p.add_argument("--prefix_pos", type=str, default="after_bos", choices=["prepend","after_bos"])
                                                                                                       
    p.add_argument("--proj_type", type=str, default="linear", choices=["linear","mlp"])
    p.add_argument("--num_prefix", type=int, default=1)

                                        
    p.add_argument("--gt_path", type=str, default="embeds", choices=["ids","embeds"])
    p.add_argument("--use_margin", action="store_true")
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--hn_topk", type=int, default=5)
    p.add_argument("--hn_weight", type=float, default=0.5)
    p.add_argument("--use_plain_text", action="store_true")

                 
    p.add_argument("--temp_init", type=float, default=2.6592)

                       
    p.add_argument("--topk", type=str, default="1,5,10",
                   help="Comma-separated K list for in-batch Hits@K/Recall@K")

    args = p.parse_args()
    main(args)
