# Model comparison

| Model | Recall@10 overall | NDCG@10 overall | Recall@10 cross-category | NDCG@10 cross-category |
|---|---:|---:|---:|---:|
| ALS | 0.5680 | 0.2819 | 0.5361 | 0.2696 |
| Two-Tower | 0.6880 | 0.3198 | 0.6084 | 0.2799 |

The cross-category test contains only users whose held-out item category differs from their dominant training-history category. Results on simulated data validate the pipeline and controlled patterns, not real business impact.
