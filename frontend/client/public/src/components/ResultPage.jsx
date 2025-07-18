import "../css/ResultPage.css";
import { useLocation, useNavigate } from "react-router-dom";
import { FaMapMarkerAlt } from "react-icons/fa";
import { useState } from "react";

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { location: userLocation, styles, request } = location.state || {};

  const [currentIndex, setCurrentIndex] = useState(2); // 총 6개 중 3번째로 시작

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev === 0 ? 5 : prev - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev === 5 ? 0 : prev + 1));
  };

  return (
    <div className="result-container">
      <h1 className="result-title">OOTD-AI</h1>
      <p className="result-subtitle">AI가 추천하는 오늘의 완벽한 룩</p>

      <div className="result-card">
        {/* 오늘의 추천 헤더 */}
        <div className="result-header">
          <div className="location-info">
            <FaMapMarkerAlt className="location-icon" />
            <span>{userLocation}</span>
          </div>
          <div className="weather-info">
            <div>
              <div className="weather-icon">☀️</div>
              <div className="weather-text">맑음</div>
              <div className="weather-temp">nn°C</div>
            </div>
            <div className="weather-sub">
              <div>습도: n%</div>
              <div>바람: n m/s</div>
            </div>
          </div>
        </div>

        {/* 캐러셀 */}
        <div className="carousel">
          <button onClick={handlePrev} className="carousel-btn">◀</button>
          <div className="carousel-image">
            <img src={`https://via.placeholder.com/250x200?text=코디${currentIndex + 1}`} alt="코디 이미지" />
          </div>
          <button onClick={handleNext} className="carousel-btn">▶</button>
        </div>
        <div className="carousel-indicator">{currentIndex + 1}/6</div>

        {/* 스타일링 팁 */}
        <div className="style-tip">
          <div className="tip-title">📦 스타일링 팁</div>
          <p>
            오늘 같은 날씨에는 가벼운 레이어드 룩을 추천드려요.<br />
            아침에는 쌀쌀할 수 있으니 가디건을 준비하시고,<br />
            낮에는 벗으셔도 좋을 것 같아요.
          </p>
        </div>

        {/* 버튼 */}
        <button className="back-btn" onClick={() => navigate("/")}>
          ← 뒤로가기
        </button>
      </div>
    </div>
  );
}
