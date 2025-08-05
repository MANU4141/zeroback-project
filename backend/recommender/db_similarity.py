def compute_similarity(query_attrs, db_attrs, weights=None):
    score = 0
    for task in ["category", "color", "material", "detail"]:
        w = weights[task] if weights and task in weights else 1
        q = set(query_attrs.get(task, []))
        d = set(db_attrs.get(task, []))
        score += w * len(q & d)
    return score


def recommend_similar_images(query_attributes, db_images, top_n=3, weights=None):
    results = []
    for img_info in db_images:
        sim = compute_similarity(query_attributes, img_info["label"], weights)
        results.append((sim, img_info["img_path"], img_info["label"]))
    results.sort(reverse=True)
    return results[:top_n]
