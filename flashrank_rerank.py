#!/usr/local/bin/python3.11
import sys
import json
from flashrank import Ranker, RerankRequest

if len(sys.argv) < 3:
    print("Usage: flashrank_rerank.py <model_name> <top_n>", file=sys.stderr)
    sys.exit(1)

model_name = sys.argv[1]
top_n = int(sys.argv[2])

data = json.load(sys.stdin)
query = data["query"]
passages = data["passages"]

ranker = Ranker(model_name=model_name)
rerank_request = RerankRequest(query=query, passages=passages)
results = ranker.rerank(rerank_request)

output = []
for i, item in enumerate(results[:top_n]):
    output.append({
        "index": i,
        "id": item.get("id", i),
        "text": item["text"],
        "score": float(item.get("score", 0.0))
    })

print(json.dumps({"results": output}))