import json, sys                                                                                                                                                            
out, target = sys.argv[1], int(sys.argv[2])                                                                                                                                 
for line in open(out):                                                                                                                                                      
    r = json.loads(line)                                  
    if r["source_dataset"] != "MuSiQue" or r["index"] != target:                                                                                                            
        continue                                          
    print("=== INDEX", target, "===")                                                                                                                                       
    print("ROUTE:", r.get("predicted_route"), " synth:", r.get("synthesis_used"))                                                                                           
    print("GOLD :", r.get("gold_answer"))                                                                                                                                   
    print("---- p4_answer ----")                                                                                                                                            
    print(r.get("p4_answer"))                                                                                                                                               
    break 