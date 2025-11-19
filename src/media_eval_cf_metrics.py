                       
""""""

import json, argparse, re, math
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

                                   
def image_name_to_movieid(image_name: str) -> str:
    """"""
    if not image_name:
        return ""
    
                         
    match = re.match(r"movieId_(\d+)_", image_name)
    if match:
        return match.group(1)                      
    
                                                                       
    return ""

                                    
def load_cf_gt_rel_sets(
    cf_gt_path: str, 
    prefer_added_only: bool = False, 
    fm_original_only: bool = False
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:                
    """"""
    cf = json.loads(Path(cf_gt_path).read_text(encoding="utf-8"))
    user_rel: Dict[str, Set[str]] = {}
    user_orig: Dict[str, Set[str]] = {}                       
    fallback_count = 0
    empty_count = 0

    if fm_original_only:
        print("[INFO] GT loader is restricted to 'fm_orig_movieId' key ONLY (due to --fm_original flag).")

    for uid, obj in cf.items():
        rel = None
        
                                           
        if fm_original_only:
            if obj.get("fm_orig_movieId"):
                rel = [str(x) for x in obj["fm_orig_movieId"]]
        
                                              
        elif prefer_added_only:
                               
            if obj.get("fm_only_added_movieId"):
                rel = [str(x) for x in obj["fm_only_added_movieId"]]
            elif obj.get("norm_final_without_original"):
                rel = [str(x) for x in obj["norm_final_without_original"]]
            elif obj.get("fm_only_added"):
                            
                rel = [str(x) for x in obj["fm_only_added"]]
                fallback_count += 1
        else:
                               
            for key in ("fm_final_movieId", "norm_final", "norm_cf"):
                if obj.get(key):
                    rel = [str(x) for x in obj[key]]
                    break
            if rel is None:
                                               
                for key in ("fm_orig_movieId", "fm_final", "fm_cf", "fm_orig"):
                    if obj.get(key):
                                         
                        rel = [str(x) for x in obj[key]]
                        fallback_count += 1
                        break

                              
        rel_set = {str(x).strip() for x in (rel or []) if str(x).strip()}
        if not rel_set:
            empty_count += 1
        user_rel[str(uid)] = rel_set

                                             
        orig_rel_list = None
        if obj.get("fm_orig_movieId"):
             orig_rel_list = [str(x) for x in obj["fm_orig_movieId"]]
        user_orig[str(uid)] = {str(x).strip() for x in (orig_rel_list or []) if str(x).strip()}


                                 
    if fm_original_only:
        if empty_count:
            print(f"[INFO] {empty_count} users have empty rel sets after restricting to 'fm_orig_movieId'.")
    elif fallback_count:
        print(f"[WARN] {fallback_count} users used fallback normalization from fm_* (non-ID) fields.")
    elif empty_count:
        print(f"[INFO] {empty_count} users have empty rel sets after prefer_added_only={prefer_added_only}.")
    
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

                                                                       
                     
                    
                   
                                                                                           
                                       
                                                                 
                                                      

                            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_json", required=True, help="Path to retrieval_results.json")
    ap.add_argument("--cf_gt_json", required=True, help="Path to CF GT JSON (user_cf_movie_gt_*.json)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks", type=str, default="1,3,5,10")
    ap.add_argument("--debug_users", type=int, default=10, help="save top N users (file order) for debug file")
    ap.add_argument("--prefer_added_only", action="store_true",
                    help="evaluate only CF-added items (exclude user's originals)")
                                                                                                                                      
                                                                                        
    ap.add_argument("--fm_original", action="store_true",
                    help="evaluate only 'fm_orig_movieId' items (overrides --prefer_added_only)")
    ap.add_argument("--restrict_to_users", type=str, default=None,
                    help="(optional) newline-separated user_id list; evaluate only these users")

    ap.add_argument("--use_image_title_norm_from_results", action="store_true",
                    help="use `image_title_norm` from retrieval JSON (as ID) instead of parsing stems")
    
    args = ap.parse_args()

    Ks = [int(x) for x in args.ks.split(",") if x.strip()]

    results = json.loads(Path(args.retrieval_json).read_text(encoding="utf-8"))
    user_rel, user_orig = load_cf_gt_rel_sets(
        args.cf_gt_json, 
        prefer_added_only=args.prefer_added_only,
        fm_original_only=args.fm_original
    )
    restrict = load_user_restriction(args.restrict_to_users)

    per_sample: List[Dict[str,Any]] = []
                                                                                     
    agg_strict = {f"{m}@{k}": 0.0 for m in ["R", "MAP"] for k in Ks}

    n = 0
    nonempty_rel_users = 0
    debug_buf = []                   

    for row in results:
        uid = str(row.get("user_id",""))
        if restrict and uid not in restrict:
            continue

                                                                            
        rel_set = user_rel.get(uid, set())
        if rel_set:
            nonempty_rel_users += 1

                              
        ranked_ids: List[str] = []
        for item in row.get("top_k_results", []):
            item_id = ""         
            
            if args.use_image_title_norm_from_results and ("image_title_norm" in item):
                item_id = str(item["image_title_norm"]).strip()
            else:
                raw_stem = str(item.get("image_name", ""))
                item_id = image_name_to_movieid(raw_stem)              
            
            if item_id:
                ranked_ids.append(item_id)

                 
                                                                  
                                                                                 
        if args.prefer_added_only:
                                                           
            orig_set = user_orig.get(uid, set())
            
            if orig_set:                                   
                                                               
                ranked_ids = [item_id for item_id in ranked_ids if item_id not in orig_set]
            
                                       
                                                
                                                                        
                                                                        
                                                         
                          


        if not ranked_ids:
                                                
                                                               
             pass                                            


        rec = {"user_id": uid, "index": row.get("index"), "rel_count": len(rel_set), "strict": {}}

                        
                                                                
                                                     
        for k in Ks:
            r = recall_at_k(rel_set, ranked_ids, k)
                                                  
            a = ap_at_k(rel_set, ranked_ids, k)
                                                   
            rec["strict"][f"R@{k}"]   = r
                                           
            rec["strict"][f"MAP@{k}"] = a
                                           

            agg_strict[f"R@{k}"]   += r
                                         
            agg_strict[f"MAP@{k}"] += a
                                         

                                                     
        if args.debug_users > 0 and len(debug_buf) < args.debug_users:
            
                                                                           
            top_10_predictions_debug = []
            
                                                             
            for i, item in enumerate(row.get("top_k_results", [])[:10]): 
                item_id = ""
                original_source = ""                              
                
                if args.use_image_title_norm_from_results and ("image_title_norm" in item):
                    item_id = str(item["image_title_norm"]).strip()
                    original_source = f"[from image_title_norm: {item_id}]"
                else:
                    original_source = str(item.get("image_name", ""))
                    item_id = image_name_to_movieid(original_source)        
                
                                                    
                parsed_id_or_fail = item_id if item_id else "PARSE_FAILED"
                
                                                            
                                         
                is_hit_added = (item_id in rel_set) if item_id else False
                
                                         
                orig_set_for_debug = user_orig.get(uid, set())
                is_hit_orig = (item_id in orig_set_for_debug) if item_id else False

                hit_status = "NO_HIT"
                if is_hit_added:
                    hit_status = "HIT_ADDED (GT)"                
                elif is_hit_orig:
                    hit_status = "HIT_ORIGINAL"                          

                                                                              
                                                                    
                is_filtered_out = (args.prefer_added_only and is_hit_orig)

                top_10_predictions_debug.append({
                    "rank": i + 1,
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
                "rel_count (added)": len(rel_set),
                "is_non_empty_gt (added)": bool(rel_set),
                "strict_hit_in_top_10 (added)": any_hit_10_strict,
                "ground_truth_ids (added)": sorted(list(rel_set)), 
                "original_ids (filtered_out)": sorted(list(user_orig.get(uid, set()))),
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
            "STRICT: exact match on parsed Movie IDs (movieId_ pattern only). "
            f"If prefer_added_only=True, original items were FILTERED from ranked lists before eval."
        )
    }

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"per_sample_eval.json").write_text(json.dumps(per_sample, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir/"debug_samples.json").write_text(json.dumps(debug_buf, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[DEBUG] wrote {len(debug_buf)} debug samples to {out_dir/'debug_samples.json'}")
    if n == 0 and len(results) > 0:
         print("[WARN] num_samples is 0. All users might have been filtered out.")
    if nonempty_rel_users == 0:
        print("[WARN] All users have empty rel sets. "
              "Likely no CF-added items (with --prefer_added_only) or wrong GT field names/IDs.")

if __name__ == "__main__":
    main()