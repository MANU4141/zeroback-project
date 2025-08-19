def compute_similarity(query_attrs, db_attrs, weights=None):
    """
    속성 기반 유사도 계산

    Args:
        query_attrs: 쿼리 속성 딕셔너리
        db_attrs: DB 이미지 속성 딕셔너리
        weights: 각 속성별 가중치 딕셔너리

    Returns:
        float: 유사도 점수
    """
    # 가중치 기본값 처리
    default_weights = {"category": 1, "color": 1, "material": 1, "detail": 1}
    if weights:
        default_weights.update(weights)
    weights = default_weights

    score = 0
    for task in ["category", "color", "material", "detail"]:
        w = weights.get(task, 1)

        # 입력 안전성: 리스트가 아닌 경우 방어 변환
        q_raw = query_attrs.get(task, [])
        d_raw = db_attrs.get(task, [])

        # 단일 문자열이면 리스트로 변환
        if isinstance(q_raw, str):
            q_raw = [q_raw]
        if isinstance(d_raw, str):
            d_raw = [d_raw]

        q = set(q_raw) if q_raw else set()
        d = set(d_raw) if d_raw else set()

        score += w * len(q & d)
    return score


def recommend_similar_images(
    query_attributes, db_images, top_n=3, weights=None, exclude_zero_score=False
):
    """
    유사한 이미지 추천

    Args:
        query_attributes: 쿼리 속성 딕셔너리
        db_images: DB 이미지 정보 리스트
        top_n: 반환할 상위 N개 결과
        weights: 속성별 가중치 딕셔너리
        exclude_zero_score: True면 0점 항목 제외

    Returns:
        list: (점수, 이미지경로, 라벨) 튜플 리스트
    """
    results = []
    for img_info in db_images:
        sim = compute_similarity(query_attributes, img_info["label"], weights)

        # 빈 점수 제거 옵션
        if exclude_zero_score and sim <= 0:
            continue

        results.append((sim, img_info["img_path"], img_info["label"]))

    # 정렬 안정성 보장: 점수만을 기준으로 정렬 (동점 시 원본 순서 유지)
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]
