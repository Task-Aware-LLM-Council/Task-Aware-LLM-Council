import json, sys                                                                                                                                                            
from collections import Counter
                                                                                                                                                                            
c = Counter()                                             
for line in open(sys.argv[1]):
    r = json.loads(line)                                                                                                                                                    
    src = r["source_dataset"]
    if src not in ("HARDMATH", "HumanEvalPlus"):                                                                                                                            
        continue                                                                                                                                                            
    c[(src, tuple(r.get("predicted_route") or []), r.get("force_role") or "free")] += 1
                                                                                                                                                                            
for (src, route, forced), n in c.most_common():                                                                                                                             
    print(f"{n:4}  src={src:14}  route={route}  force={forced}")                                                                                                            
                                                                                                                                                                              
