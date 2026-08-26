# Problem statement

Build a ranking system that recommends a user's next item or service from their
multi-category interaction history. The primary objective is **cross-category
recommendation**: evidence in category A should help rank relevant items in category
B. This differs from a conventional recommender whose easiest success is retrieving
more items from the user's already dominant category.

The offline protocol uses a temporal leave-one-out split. It reports Recall@10 and
NDCG@10 for all eligible users, then separately for cases where the held-out item's
category differs from the user's dominant training-history category. Accuracy is not
used because the output is an ordered candidate list, not a class label. RMSE is not
the primary metric because the objective is retrieval/ranking rather than exact rating
prediction.

Stage A uses public Amazon Reviews 2023 categories to establish a measurable
algorithmic baseline. Stage B merges multiple public datasets and uses explicitly
labeled, controlled synthetic journeys to demonstrate source decoupling and
service-oriented scenarios. Stage B does not touch or simulate any specific company's
real customer data, and its metrics do not claim real customer behavior or business
insight for any company.

