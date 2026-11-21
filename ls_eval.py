import time 

from langsmith import Client, evaluate

from eval import load_questions, is_hit, PAUSE
from lc_search import search 

DATASET = "ctm-questions-50"
client = Client()

if not client.has_dataset(dataset_name=DATASET):
    dataset = client.create_dataset(DATASET, description="CTM eval questions with expected verses")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": q}, "outputs": {"expected": e}}
            for q, e in load_questions()
        ],
    ) 
    print(f"Uploaded {len(load_questions())} questions to '{DATASET}'")

def target(inputs):
    docs = search(inputs["question"])
    time.sleep(PAUSE[True])
    return {"refs": [d.metadata["ref"] for d in docs]}

def hit_at_5(outputs, reference_outputs): 
    return any(is_hit(r, reference_outputs["expected"]) for r in outputs["refs"])

if __name__ == "__main__": 
    evaluate(
        target,
        data=DATASET,
        evaluators=[hit_at_5],
        experiment_prefix="rerank-numpy-ar-50",
        max_concurrency=1,
    )
    