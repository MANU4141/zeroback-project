import "../css/OOTDForm.css";
import { useState } from "react";
import { FaMapMarkerAlt, FaSearch } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";
import axios from "axios";

export default function OOTDForm() {
  const navigate = useNavigate();
  const [location, setLocation] = useState(""); // 선택한 주소
  const [coords, setCoords] = useState({ lat: null, lng: null }); // ✅ 위도, 경도
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [styles, setStyles] = useState([]);
  const [request, setRequest] = useState("");

  const styleOptions = [
    "스트릿", "미니멀", "빈티지", "캐주얼", "러블리", "오피스", "하이틴", "아메카지",
  ];

  const toggleStyle = (style) => {
    setStyles((prev) =>
      prev.includes(style) ? prev.filter((s) => s !== style) : [...prev, style]
    );
  };

  const handleSubmit = async () => {
    if (!location || styles.length === 0) {
      alert("위치와 스타일을 선택해주세요.");
      return;
    }

    const requestData = {
      location: location,
      latitude: coords.lat,
      longitude: coords.lng,
      style_select: styles,
      user_request: request || "" // 요청사항 없으면 빈 값
    };

    const formData = new FormData();
    formData.append("data", JSON.stringify(requestData));
    // formData.append("image", imageFile); // 이미지 선택 시 추가 가능

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/recommend", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("✅ 백엔드 응답:", response.data);
      navigate("/result", { state: response.data }); // 결과 페이지로 이동
    } catch (error) {
      console.error("❌ API 요청 오류:", error);
      alert("추천 요청 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className={`ootd-container ${isModalOpen ? "blurred" : ""}`}>
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI가 추천하는 오늘의 완벽한 코디</p>

      <div className="form-card">
        {/* ✅ 위치 선택 */}
        <div className="form-section">
          <label className="form-label">위치 선택</label>
          <div className="location-box" onClick={() => setIsModalOpen(true)}>
            <FaMapMarkerAlt className="icon" />
            <span className="location-label">
              {location || "위치를 선택하세요"}
            </span>
          </div>
        </div>

        {/* ✅ 스타일 선택 */}
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

        {/* ✅ 추가 요청사항 */}
        <div className="form-section">
          <label className="form-label">추가 요청사항</label>
          <textarea
            placeholder="원하는 스타일이나 요청사항을 입력하세요."
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
        </div>

        {/* ✅ 제출 버튼 */}
        <button className="submit-btn" onClick={handleSubmit}>
          추천받기
        </button>
      </div>

      {/* ✅ 모달 */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">위치 선택</h3>
            <div className="modal-map">
              <KakaoMapModal
                onSelect={(data) => {
                  setLocation(data.address);
                  setCoords({ lat: data.lat, lng: data.lng }); // ✅ 좌표 저장
                }}
                onClose={() => setIsModalOpen(false)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
