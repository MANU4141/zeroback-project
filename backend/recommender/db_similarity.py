def compute_similarity(query_attrs, db_attrs, weights=None):
    score = 0
    # style에 가장 높은 가중치 부여 (예: 10)
    style_weight = 10
    style_query = set(query_attrs.get("style", []))
    style_db = set(db_attrs.get("style", []))
    # style 완전 일치 시 특수 플래그 반환
    if style_query and style_db and style_query & style_db:
        # style이 하나라도 겹치면 최우선
        return 10000  # 매우 큰 값으로 style 매칭을 최상위로
    # style이 겹치지 않으면 나머지 속성 가중치로 점수 계산
    for task in ["category", "color", "material", "detail"]:
        w = weights.get(task, 1) if weights else 1
        q = set(query_attrs.get(task, []))
        d = set(db_attrs.get(task, []))
        score += w * len(q & d)
    # style도 점수에 반영 (겹치는 style 개수 * style_weight)
    score += style_weight * len(style_query & style_db)
    return score


def recommend_similar_images(query_attributes, db_images, top_n=3, weights=None):
    results = []
    for img_info in db_images:
        sim = compute_similarity(query_attributes, img_info["label"], weights)
        results.append((sim, img_info["img_path"], img_info["label"]))
    results.sort(reverse=True)
    return results[:top_n]
