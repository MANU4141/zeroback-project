import React from "react";
import { useNavigate } from "react-router-dom";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  
  // ✅ 날씨 아이콘
  const getWeatherIcon = (desc) => {
    const d = desc.toLowerCase();
    if (d.includes("맑음") || d.includes("clear")) return "☀️";
    if (d.includes("구름") || d.includes("cloud")) return "☁️";
    if (d.includes("비") || d.includes("rain")) return "🌧️";
    if (d.includes("눈") || d.includes("snow")) return "❄️";
    return "🌤️";
  };

  return (
    <div className="result-wrapper">
      {/* ✅ 상단 제목 */}
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI가 추천하는 오늘의 완벽한 룩</p>

      <div className="result-layout">
        {/* ✅ 왼쪽 추천 이미지 영역 */}
        <div className="image-card">
          <div className="image-header">
            <h2 className="section-title">오늘의 추천 룩</h2>
            <span className="location">{state.location}</span>
          </div>
          <div className="image-grid">
            {state.recommended_images.map((img, idx) => (
              <img key={idx} src={img} alt={`추천 ${idx}`} />
            ))}
          </div>
        </div>

        {/* ✅ 오른쪽 정보 카드 */}
        <div className="sidebar">
          {/* 날씨 정보 */}
          <div className="info-card weather-card">
            <h3 className="card-title">날씨 정보</h3>
            <div className="weather-content">
              <div className="weather-icon">{getWeatherIcon(state.weather.description)}</div>
              <div className="weather-details">
                <p className="weather-desc">{state.weather.description}</p>
                <p className="weather-temp">{state.weather.temp}°C</p>
              </div>
            </div>
            <div className="weather-extra">
              <p>습도: {state.weather.humidity}%</p>
              <p>바람: {state.weather.wind} m/s</p>
            </div>
          </div>

          {/* 추천 아이템 */}
          <div className="info-card">
            <h3 className="card-title">추천 아이템</h3>
            <ul className="item-list">
              {state.recommended_items.map((item, idx) => (
                <li key={idx}>
                  <span className="dot"></span>
                  {item}
                </li>
              ))}
            </ul>
          </div>

          {/* 스타일링 팁 */}
          <div className="info-card">
            <h3 className="card-title">💡 스타일링 팁</h3>
            <p className="styling-tip">{state.styling_tip}</p>
          </div>

          {/* 다시 요청 버튼 */}
          <button className="retry-btn" onClick={() => navigate("/")}>
            새로운 추천 받기
          </button>
        </div>
      </div>
    </div>
  );
}
