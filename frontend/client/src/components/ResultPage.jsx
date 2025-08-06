import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaCheckCircle } from "react-icons/fa";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  // 더미 데이터 (백엔드 연결 전 테스트)
  const data = state || {
    location: "서울 강남구",
    weather: {
      description: "비",
      temp: 19,
      humidity: 61,
      wind: 9,
    },
    recommended_images: [
      "https://images.pexels.com/photos/3756042/pexels-photo-3756042.jpeg",
      "https://images.pexels.com/photos/2983464/pexels-photo-2983464.jpeg",
      "https://images.pexels.com/photos/1552242/pexels-photo-1552242.jpeg",
      "https://images.pexels.com/photos/3774933/pexels-photo-3774933.jpeg",
    ],
    recommended_items: [
      "상의: 반팔 티셔츠",
      "하의: 청바지",
      "겉옷: 방수 재킷",
      "신발: 방수 부츠",
      "액세서리: 우산",
    ],
    styling_tip: "오늘은 비가 오므로 방수 재킷과 방수 부츠를 추천합니다.",
  };

  const getWeatherIcon = (desc) => {
    const d = desc.toLowerCase();
    if (d.includes("맑음") || d.includes("clear")) return "☀️";
    if (d.includes("구름") || d.includes("cloud")) return "☁️";
    if (d.includes("비") || d.includes("rain")) return "🌧️";
    if (d.includes("눈") || d.includes("snow")) return "❄️";
    return "🌤️";
  };

  return (
    <div className="result-container">
      <h1 className="title">오늘의 추천 코디</h1>
      <p className="subtitle">당신을 위한 AI 기반 스타일링</p>

      <div className="result-content">
        {/* 왼쪽: 이미지 */}
        <div className="image-section">
          <h2>추천 룩</h2>
          <div className="image-grid">
            {data.recommended_images.map((img, idx) => (
              <img key={idx} src={img} alt={`추천 ${idx + 1}`} />
            ))}
          </div>
        </div>

        {/* 오른쪽: 정보 */}
        <div className="info-section">
          {/* 날씨 정보 */}
          <div className="info-card weather-card">
            <h3>날씨 정보</h3>
            <div className="weather-main">
              <span className="weather-icon">{getWeatherIcon(data.weather.description)}</span>
              <div>
                <p>{data.weather.description}</p>
                <p>{data.weather.temp}°C</p>
              </div>
            </div>
            <p>습도: {data.weather.humidity}%</p>
            <p>바람: {data.weather.wind} m/s</p>
          </div>

          {/* 추천 아이템 */}
          <div className="info-card">
            <h3>추천 아이템</h3>
            <ul>
              {data.recommended_items.map((item, idx) => (
                <li key={idx}>
                  <FaCheckCircle className="item-icon" /> {item}
                </li>
              ))}
            </ul>
          </div>

          {/* 스타일링 팁 */}
          <div className="info-card">
            <h3>💡 스타일링 팁</h3>
            <p>{data.styling_tip}</p>
          </div>

          {/* 버튼 */}
          <button className="retry-btn" onClick={() => navigate("/")}>
            새로운 추천 받기
          </button>
        </div>
      </div>
    </div>
  );
}
