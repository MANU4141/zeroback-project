// src/components/OOTDForm.jsx
import "../css/OOTDForm.css";
import { useState } from "react";
import { FaMapMarkerAlt, FaSearch } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";

export default function OOTDForm() {
  const navigate = useNavigate();
  const [location, setLocation] = useState("");          // 지도 또는 도시 선택으로 저장
  const [selectedCity, setSelectedCity] = useState("");  // 버튼 클릭으로 고른 도시명
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [styles, setStyles] = useState([]);
  const [request, setRequest] = useState("");

  const styleOptions = [
    "스트릿", "미니멀", "빈티지", "캐주얼", "러블리", "오피스", "댄디", "아메카지",
  ];

  const toggleStyle = (style) => {
    setStyles((prev) =>
      prev.includes(style)
        ? prev.filter((s) => s !== style)
        : [...prev, style]
    );
  };

  const handleSubmit = () => {
    const finalLocation = location || selectedCity;
    if (finalLocation && styles.length > 0 && request.trim() !== "") {
      navigate("/result", { state: { location: finalLocation, styles, request } });
    } else {
      alert("모든 항목을 입력해주세요.");
    }
  };

  return (
    <div className={`ootd-container ${isModalOpen ? "blurred" : ""}`}>
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI가 추천하는 오늘의 완벽한 룩</p>

      <div className="form-card">
        {/* 위치 선택 */}
        <div className="form-section">
          <label className="form-label">위치 선택</label>
          <div className="location-box" onClick={() => setIsModalOpen(true)}>
            <FaMapMarkerAlt className="icon" />
            <span className="location-label">
              {location || selectedCity || "위치 선택"}
            </span>
          </div>
        </div>

        {/* 스타일 선택 */}
        <div className="form-section">
          <label className="form-label">스타일 (중복 선택 가능)</label>
          <div className="search-bar">
            <input placeholder="search text" />
            <FaSearch className="icon" />
          </div>
          <div className="style-grid">
            {styleOptions.map((style, idx) => (
              <button
                key={idx}
                className={`style-btn ${styles.includes(style) ? "selected" : ""}`}
                onClick={() => toggleStyle(style)}
              >
                {style}
              </button>
            ))}
          </div>
        </div>

        {/* 요청사항 */}
        <div className="form-section">
          <label className="form-label">추가 요청사항</label>
          <textarea
            placeholder="원하는 스타일이나 요청사항을 입력해주세요."
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
        </div>

        {/* 제출 버튼 */}
        <button className="submit-btn" onClick={handleSubmit}>
          추천받기
        </button>
      </div>

      {/* 지도 모달 */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">위치 선택</h3>
            <input
              type="text"
              placeholder="🔍 위치 검색"
              className="modal-search"
            />
            <div className="modal-map">
              <KakaoMapModal onSelect={(addr) => setLocation(addr)} />
            </div>
            <div className="city-grid">
              {["서울", "인천", "부산", "대구", "광주", "대전"].map((city) => (
                <button
                  key={city}
                  className={`city-btn ${selectedCity === city ? "active" : ""}`}
                  onClick={() => {
                    setSelectedCity(city);
                    setLocation(city);
                  }}
                >
                  {city}
                </button>
              ))}
            </div>
            <button className="modal-confirm" onClick={() => setIsModalOpen(false)}>
              확인
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
