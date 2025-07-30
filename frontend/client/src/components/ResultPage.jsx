import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  if (!state) {
    return (
      <div className="result-container">
        <h2>결과 데이터를 찾을 수 없습니다.</h2>
        <button className="back-btn" onClick={() => navigate("/")}>
          다시 시도하기
        </button>
      </div>
    );
  }

  const { location, latitude, longitude, style_select, user_request, recommended_images } = state;

  return (
    <div className="result-container">
      <h1 className="title">추천 결과</h1>
      <p className="subtitle">당신을 위한 AI 코디 추천</p>

      {/* ? 선택한 정보 표시 */}
      <div className="info-box">
        <h3>? 선택한 위치</h3>
        <p>{location}</p>
        <p>위도: {latitude} | 경도: {longitude}</p>

        <h3>? 선택한 스타일</h3>
        <p>{style_select.join(", ")}</p>

        {user_request && (
          <>
            <h3>? 추가 요청사항</h3>
            <p>{user_request}</p>
          </>
        )}
      </div>

      {/* ? 사용자가 업로드한 이미지 미리보기 */}
      <div className="user-images">
        <h3>? 업로드한 이미지</h3>
        <div className="image-preview-grid">
          {state.user_images && state.user_images.length > 0 ? (
            state.user_images.map((img, idx) => (
              <img key={idx} src={img} alt="uploaded" />
            ))
          ) : (
            <p>업로드된 이미지가 없습니다.</p>
          )}
        </div>
      </div>

      {/* ? AI 추천 이미지 표시 */}
      <div className="recommended-images">
        <h3>? AI 추천 코디</h3>
        <div className="image-preview-grid">
          {recommended_images && recommended_images.length > 0 ? (
            recommended_images.map((img, idx) => (
              <img key={idx} src={img} alt="recommend" />
            ))
          ) : (
            <p>추천 이미지를 불러오는 중...</p>
          )}
        </div>
      </div>

      {/* ? 다시 요청 버튼 */}
      <button className="back-btn" onClick={() => navigate("/")}>
        다시 요청하기
      </button>
    </div>
  );
}
