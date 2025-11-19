                      
                       

import json, argparse, re, math
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

def parse_appid_from_image_name(stem: str) -> str:
    if not stem:
        return ""
    first = stem.split("/", 1)[0]
    return first if first.isdigit() else ""


def load_cf_gt_appids(
    cf_gt_path: str,
    prefer_added_only: bool = False,
    fm_original_only: bool = False
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    cf = json.loads(Path(cf_gt_path).read_text(encoding="utf-8"))
    user_rel: Dict[str, Set[str]] = {}
    user_orig: Dict[str, Set[str]] = {} 
    empty_count = 0
    
    keys_added = ["fm_only_added_appId", "norm_final_without_original_appId"]
    keys_final = ["fm_final_appId", "norm_final_appId"]
    keys_orig = ["fm_orig_appId"]

    if fm_original_only:
        print(f"[INFO] GT loader is restricted to {keys_orig} key(s) ONLY (due to --fm_original flag).")

    for uid, obj in cf.items():
        rel_list = None
        

        if fm_original_only:
            for key in keys_orig:
                if obj.get(key):
                    rel_list = [str(x) for x in obj[key]]
                    break
        

        elif prefer_added_only:
            for key in keys_added:
                if obj.get(key):
                    rel_list = [str(x) for x in obj[key]]
                    break
        

        else:
            for key in keys_final:
                if obj.get(key):
                    rel_list = [str(x) for x in obj[key]]
                    break

            if rel_list is None:
                 for key in keys_orig:
                    if obj.get(key):
                        rel_list = [str(x) for x in obj[key]]
                        break


        rel_set = {str(x).strip() for x in (rel_list or []) if str(x).strip()}
        if not rel_set:
            empty_count += 1
        user_rel[str(uid)] = rel_set


        orig_rel_list = None
        for key in keys_orig:
             if obj.get(key):
                orig_rel_list = [str(x) for x in obj[key]]
                break
        user_orig[str(uid)] = {str(x).strip() for x in (orig_rel_list or []) if str(x).strip()}

    if empty_count > 0:
        print(f"[INFO] {empty_count} users have empty rel sets (fm_original={fm_original_only}, prefer_added_only={prefer_added_only}).")
    
    return user_rel, user_orig

def load_user_restriction(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] restrict file not found: {path}")
        return None
    ids = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    print(f"[INFO] restricting to {len(ids)} users from {path}")
    return ids

                                                
def recall_at_k(rel_set: Set[str], ranked: List[str], k: int) -> float:
    if not rel_set:
        return 0.0
    for t in ranked[:k]:
        if t in rel_set:
            return 1.0
    return 0.0

def mrr_at_k(rel_set: Set[str], ranked: List[str], k: int) -> float:
    if not rel_set:
        return 0.0
    for i, t in enumerate(ranked[:k], 1):
        if t in rel_set:
            return 1.0 / i
    return 0.0

def ap_at_k(rel_set: Set[str], ranked: List[str], k: int) -> float:
    if not rel_set:
        return 0.0
    hits = 0
    s = 0.0
    for i, t in enumerate(ranked[:k], 1):
        if t in rel_set:
            hits += 1
            s += hits / i
    return s / max(1, len(rel_set))

def ndcg_at_k(rel_set: Set[str], ranked: List[str], k: int) -> float:
    if not rel_set:
        return 0.0
    def dcg(lst):
        return sum((1.0 / math.log2(i+2)) for i, t in enumerate(lst[:k]) if t in rel_set)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i+2) for i in range(ideal_hits))
    return (dcg(ranked) / idcg) if idcg > 0 else 0.0


def get_ranked_appids(row: Dict[str, Any],
                      use_image_appid_from_results: bool = False,
                      use_image_title_norm_from_results: bool = False) -> List[Tuple[str, str]]: 

    ranked_ids_with_source: List[Tuple[str, str]] = []
    
    for item in row.get("top_k_results", []):
        appid = ""
        source = ""
        
        if use_image_appid_from_results and ("image_appid" in item):
            appid = str(item.get("image_appid") or "").strip()
            source = f"[image_appid: {appid}]"
        elif use_image_title_norm_from_results and ("image_title_norm" in item):
            appid = str(item.get("image_title_norm") or "").strip()
            source = f"[image_title_norm: {appid}]" 
        else:
            source = str(item.get("image_name", "")).strip()
            appid = parse_appid_from_image_name(source)
            
        if appid:
            ranked_ids_with_source.append((appid, source))
        else:

            ranked_ids_with_source.append(("", source)) 
            
    return ranked_ids_with_source

                                          
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_json", required=True)
    ap.add_argument("--cf_gt_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=str, default="1,3,5,10")
    ap.add_argument("--debug_users", type=int, default=10)
    
    ap.add_argument("--use_image_appid_from_results", action="store_true",
                    help="")
    ap.add_argument("--use_image_title_norm_from_results", action="store_true",
                    help="use `image_title_norm` from retrieval JSON (as ID, not recommended for games)")

    ap.add_argument("--prefer_added_only", action="store_true",
                    help="evaluate only CF-added items AND filter original items from predictions")
    ap.add_argument("--fm_original", action="store_true",
                    help="evaluate only 'fm_orig_appId' items (overrides --prefer_added_only)")
    ap.add_argument("--restrict_to_users", type=str, default=None,
                    help="(optional) newline-separated user_id list; evaluate only these users")

    args = ap.parse_args()
    Ks = [int(x) for x in args.ks.split(",") if x.strip()]
    
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    
    user_rel, user_orig = load_cf_gt_appids(
        args.cf_gt_json,
        prefer_added_only=args.prefer_added_only,
        fm_original_only=args.fm_original
    )
    restrict = load_user_restriction(args.restrict_to_users)

    per_sample: List[Dict[str,Any]] = []
    agg_strict = {f"{m}@{k}": 0.0 for m in ["R", "MRR", "MAP", "nDCG"] for k in Ks}
    debug_buf = [] 

    n = 0
    nonempty_rel_users = 0

    for row in results:
        uid = str(row.get("user_id",""))
        if restrict and uid not in restrict:
            continue

        rel_set = user_rel.get(uid, set())
        if rel_set:
            nonempty_rel_users += 1
            
        orig_set = user_orig.get(uid, set())

        ranked_ids_with_source = get_ranked_appids(
            row,
            use_image_appid_from_results=args.use_image_appid_from_results,
            use_image_title_norm_from_results=args.use_image_title_norm_from_results
        )
        
        ranked_ids_full = [appid for appid, src in ranked_ids_with_source if appid]

        if args.prefer_added_only:
            if orig_set: 

                ranked_ids = [item_id for item_id in ranked_ids_full if item_id not in orig_set]
            else:
                ranked_ids = ranked_ids_full 
        else:

            ranked_ids = ranked_ids_full 
            

        rec = {"user_id": uid, "index": row.get("index"), "rel_count": len(rel_set), "strict": {}}

        for k in Ks:
            r = recall_at_k(rel_set, ranked_ids, k)
            m = mrr_at_k(rel_set, ranked_ids, k)
            a = ap_at_k(rel_set, ranked_ids, k)
            d = ndcg_at_k(rel_set, ranked_ids, k)
            rec["strict"][f"R@{k}"]   = r
            rec["strict"][f"MRR@{k}"] = m
            rec["strict"][f"MAP@{k}"] = a
            rec["strict"][f"nDCG@{k}"]= d

            agg_strict[f"R@{k}"]   += r
            agg_strict[f"MRR@{k}"] += m
            agg_strict[f"MAP@{k}"] += a
            agg_strict[f"nDCG@{k}"]+= d


        if args.debug_users > 0 and len(debug_buf) < args.debug_users:
            top_10_predictions_debug = []

            for i, (item_id, original_source) in enumerate(ranked_ids_with_source[:10], 1):

                parsed_id_or_fail = item_id if item_id else "PARSE_FAILED"

                is_hit_added = (item_id in rel_set) if item_id else False
                
                is_hit_orig = (item_id in orig_set) if item_id else False

                hit_status = "NO_HIT"
                if is_hit_added:
                    hit_status = "HIT_ADDED (GT)" 
                elif is_hit_orig:
                    hit_status = "HIT_ORIGINAL"


                is_filtered_out = (args.prefer_added_only and is_hit_orig)

                top_10_predictions_debug.append({
                    "rank": i,
                    "original_source": original_source, 
                    "parsed_id": parsed_id_or_fail, 
                    "hit_status": hit_status, 
                    "is_filtered_out_by_prefer_added_only": is_filtered_out 
                })
            

            any_hit_10_strict = any(
                item["hit_status"] == "HIT_ADDED (GT)" 
                for item in top_10_predictions_debug
            )
            
            dentry = {
                "user_id": uid,
                "rel_count (GT)": len(rel_set),
                "is_non_empty_gt": bool(rel_set),
                "strict_hit_in_top_10 (GT)": any_hit_10_strict,
                "ground_truth_ids (GT)": sorted(list(rel_set)), 
                "original_ids (filtered_out)": sorted(list(orig_set)), 
                "top_10_predictions": top_10_predictions_debug
            }
            debug_buf.append(dentry)

        per_sample.append(rec)
        n += 1 

                   
    for k in list(agg_strict.keys()):
        agg_strict[k] = (agg_strict[k]/n) if n else 0.0

    summary = {
        "num_samples": n,
        "users_with_nonempty_rel": nonempty_rel_users,
        "prefer_added_only": args.prefer_added_only,
        "fm_original_only": args.fm_original,
        "ks": Ks,
        "metrics_strict": agg_strict,
        "notes": (
            "STRICT: exact match on parsed AppIDs (APPID/... pattern or --use_image_appid_from_results). "
            f"If prefer_added_only=True, original items were FILTERED from ranked lists before eval."
        )
    }

    (out_dir/"per_sample_eval.json").write_text(json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir/"debug_samples.json").write_text(json.dumps(debug_buf, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[DEBUG] wrote {len(debug_buf)} debug samples to {out_dir/'debug_samples.json'}")
    if n == 0 and len(results) > 0:
         print("[WARN] num_samples is 0. All users might have been filtered out by --restrict_to_users.")
    if nonempty_rel_users == 0:
        print("[WARN] All users have empty rel sets. Check GT field names (e.g., fm_only_added_appId) or flags.")


if __name__ == "__main__":
    main()