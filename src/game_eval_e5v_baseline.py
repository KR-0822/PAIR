                      
                       
""""""

import os, csv, json, argparse, shutil, re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from tqdm import tqdm

                         
       
                         
def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _split_triple(s: str):
    if not isinstance(s, str): return None
    row = next(csv.reader([s], skipinitialspace=True))
    if len(row) >= 3:
        dom, sub, val = row[0].strip(), row[1].strip(), ",".join(row[2:]).strip()
        return dom, sub, val
    parts = [x.strip() for x in s.split(',', 2)]
    if len(parts) == 3: return parts[0], parts[1], parts[2]
    return None

                         
                                  
                         
def format_preferences_naturally(
    lst: List[Union[str, dict]],
    exclude_idx: int = -1,
    allow_game_names: bool = False
) -> str:
    """"""
    sents = []
    for i, p in enumerate(lst):
        if i == exclude_idx: continue
        if isinstance(p, dict):                            
            continue
        triple = _split_triple(p)
        if not triple: continue
        domain, sub, val = triple
        d = (domain or "").strip().lower()
        s = (sub or "").strip().lower()
        v = (val or "").strip()
        if not v: continue

                            
        if "favorite media" in s or "favourite media" in s:
            continue

                           
        if "preferred game name" in s:
            if allow_game_names:
                sents.append(f"likes the game {v}")
            continue

        if d == "games":
            if "genre" in s:
                sents.append(f"prefers {v} game genre")
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
            seen.add(x); uniq.append(x)
    return ". ".join(uniq[:100])

def format_target_query(domain: str, sub: str) -> str:
    return "Recommend images related to video games"

                         
            
                         
def load_test_rows(filepath: str,
                   target_policy: str = "all",
                   allowed_subs: Optional[List[str]] = None,
                   skip_on_mismatch: bool = False,
                   allow_game_names: bool = False) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    allowed = set([s.strip().lower() for s in (allowed_subs or [])])
    out: List[Dict[str, Any]] = []

    def add_row_core(uid: str, triples: List[Union[str, dict]], t_idx: int):
        tgt = triples[t_idx]
        domain = sub = val = ""
        if isinstance(tgt, dict) and "originalGT" in tgt and "templateGT" in tgt:
            parsed = _split_triple(tgt["originalGT"])
            if parsed: domain, sub, val = parsed
            gt_text = tgt["templateGT"]
        else:
            parsed = _split_triple(tgt) if isinstance(tgt, str) else None
            if parsed: domain, sub, val = parsed
            gt_text = val
        if not (domain and sub and gt_text): return

        prefs = format_preferences_naturally(triples, exclude_idx=t_idx, allow_game_names=allow_game_names)
        if not prefs: return

        out.append({
            "user_id": uid,
            "demographic_and_preferences": prefs.strip(),
            "target_query": format_target_query(domain, sub),
            "gt": gt_text,
            "domain": domain,
            "subdomain": sub
        })

    for sample in data:
        if not isinstance(sample, (list, tuple)) or len(sample) < 2: continue
        meta = sample[0]
        uid = str(meta.get("user_id", ""))
        triples = list(sample[1:])

        if target_policy == "all":
            for t_idx, _ in enumerate(triples):
                add_row_core(uid, triples, t_idx)
        elif target_policy == "last":
            add_row_core(uid, triples, len(triples) - 1)
        elif target_policy == "filter_last":
            t_idx = len(triples) - 1
            tgt = triples[t_idx]
            subname = ""
            if isinstance(tgt, dict) and "originalGT" in tgt:
                parsed = _split_triple(tgt["originalGT"])
                if parsed: _, subname, _ = parsed
            elif isinstance(tgt, str):
                parsed = _split_triple(tgt)
                if parsed: _, subname, _ = parsed
            if subname.strip().lower() in allowed:
                add_row_core(uid, triples, t_idx)
            else:
                if not skip_on_mismatch:
                    add_row_core(uid, triples, t_idx)
        else:
            raise ValueError(f"Unknown target_policy: {target_policy}")

    return out

                         
                                     
                         
def load_image_embeddings(embedding_dir: str, device: torch.device):
    root = Path(embedding_dir)
    files = sorted(root.rglob("*.npy"))             
    names, embs = [], []
    for f in tqdm(files, desc="Load image embeddings"):
        rel = f.relative_to(root).with_suffix("")
        rel_str = str(rel).replace(os.sep, "/")                            
        names.append(rel_str)
        arr = np.load(str(f))
        if arr.ndim == 1:
            pass
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        elif arr.ndim > 2:
            arr = arr.reshape(-1)
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

                                   
    processor = LlavaNextProcessor.from_pretrained(args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    llama_tpl = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    fmt = "{}\nSummary above sentence in one word: "

                         
    allowed = args.allowed_subs.split(";") if args.allowed_subs else None
    test_rows = load_test_rows(
        args.test_file,
        target_policy=args.target_policy,
        allowed_subs=allowed,
        skip_on_mismatch=args.skip_on_mismatch,
        allow_game_names=args.allow_game_names_in_context
    )
    print(f"[DATA] test rows: {len(test_rows)}")

                         
    img_names, img_mat = load_image_embeddings(args.image_embedding_dir, device)
    print(f"[IMG] embeddings: {len(img_names)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    BS = args.batch_size

    @torch.no_grad()
    def encode_e5v_texts(texts: List[str]) -> torch.Tensor:
                              
        if args.use_plain_text:
            text_inputs = processor.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(device)
            outputs = model.language_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            wrapped = [llama_tpl.format(fmt.format(t)) for t in texts]
            text_inputs = processor(wrapped, return_tensors='pt', padding=True).to(device)
            outputs = model.language_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        hidden = outputs.hidden_states[-1]
        embs = hidden[:, -1, :]
        return F.normalize(embs, p=2, dim=1)

    for i in tqdm(range(0, len(test_rows), BS), desc="Inference"):
        batch = test_rows[i:i+BS]
        demos   = [r["demographic_and_preferences"] for r in batch]
        queries = [r["target_query"] for r in batch]
        gts     = [r["gt"] for r in batch]

                               
        full_texts = [f"{d}. {q}" for d, q in zip(demos, queries)]

        q_embs = encode_e5v_texts(full_texts)

                     
        if args.eval_gt_similarity:
            gt_embs = encode_e5v_texts(gts)
            gt_sims = F.cosine_similarity(q_embs, gt_embs, dim=1).tolist()
        else:
            gt_sims = [None] * len(batch)

        for j, (qe, row, gt_sim) in enumerate(zip(q_embs, batch, gt_sims)):
            topk = retrieve_top_k(qe, img_names, img_mat, k=args.top_k)
            results.append({
                "index": i + j,
                "user_id": row["user_id"],
                "combined_input_text": full_texts[j],
                "demographic_and_preferences": row["demographic_and_preferences"],
                "target_query": row["target_query"],
                "domain": row["domain"],
                "subdomain": row["subdomain"],
                "ground_truth_text": row["gt"],
                "gt_text_similarity": gt_sim,
                "top_k_results": [
                    {"rank": rnk, "image_name": name, "similarity": sim}
                    for rnk, (name, sim) in enumerate(topk, 1)
                ],
            })

    out_file = Path(args.output_dir) / "retrieval_results_e5v_direct.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_file}")
    print(f"[STATS] Total={len(results)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
        
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--image_embedding_dir", type=str, required=True)

        
    ap.add_argument("--model_name", type=str, default="royokong/e5-v")
    ap.add_argument("--output_dir", type=str, default="./inference_output_e5v_direct")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

            
    ap.add_argument("--use_plain_text", action="store_true")
    ap.add_argument("--eval_gt_similarity", action="store_true")

                           
    ap.add_argument("--target_policy", type=str, default="all", choices=["all","last","filter_last"])
    ap.add_argument("--allowed_subs", type=str,
                    default="Preferred Game Name;Preferred Game Genres;Multiplayer Preference;Gaming Frequency")
    ap.add_argument("--skip_on_mismatch", action="store_true")

                                          
    ap.add_argument("--allow_game_names_in_context", action="store_true")

    args = ap.parse_args()
    main(args)
