import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation(); // OOTDForm에서 전달한 데이터
  const data = state || {};

  if (!data.success) {
    return (
      <div className="result-container">
        <h2>추천 결과를 불러올 수 없습니다.</h2>
        <button className="back-btn" onClick={() => navigate("/")}>
          다시 시도하기
        </button>
      </div>
    );
  }

  return (
    <div className="result-container">
      <h1 className="title">오늘의 추천 코디</h1>

      {/* ? 날씨 정보 */}
      <div className="weather-box">
        <h3>현재 날씨</h3>
        <p>?? 온도: {data.weather.temperature}°C</p>
        <p>?? 상태: {data.weather.condition}</p>
        <p>? 습도: {data.weather.humidity}%</p>
        <p>? 풍속: {data.weather.wind_speed} m/s</p>
      </div>

      {/* ? 추천 코디 */}
      <div className="recommend-box">
        <h3>추천 스타일</h3>
        <div className="recommend-grid">
          {data.recommendations && data.recommendations.length > 0 ? (
            data.recommendations.map((item, index) => (
              <div key={index} className="recommend-card">
                <img src={item.image} alt={item.description} />
                <p>{item.description}</p>
              </div>
            ))
          ) : (
            <p>추천 스타일이 없습니다.</p>
          )}
        </div>
      </div>

      <button className="back-btn" onClick={() => navigate("/")}>
        다시 코디 받기
      </button>
    </div>
  );
}
