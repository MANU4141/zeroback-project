import "../css/OOTDForm.css";
import { useState } from "react";
import { FaMapMarkerAlt, FaSearch } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";
import axios from "axios";

export default function OOTDForm() {
  const navigate = useNavigate();
  const [location, setLocation] = useState(""); // 선택한 주소
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
    if (!location || styles.length === 0 || request.trim() === "") {
      alert("모든 항목을 입력해주세요.");
      return;
    }

    // ✅ 백엔드로 전송할 데이터 구성
    const formData = new FormData();
    const requestData = {
      location: location,
      style_select: styles,
      user_request: request,
    };

    formData.append("data", JSON.stringify(requestData));
    // 이미지 추가 가능 (선택): formData.append("image", imageFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/recommend", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("추천 결과:", response.data);

      // ✅ API 응답 데이터를 ResultPage로 전달
      navigate("/result", { state: response.data });

    } catch (error) {
      console.error("API 요청 오류:", error);
      alert("추천 요청 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className={`ootd-container ${isModalOpen ? "blurred" : ""}`}>
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI가 추천하는 오늘의 완벽한 코디</p>

      <div className="form-card">
        {/* 위치 선택 */}
        <div className="form-section">
          <label className="form-label">위치 선택</label>
          <div className="location-box" onClick={() => setIsModalOpen(true)}>
            <FaMapMarkerAlt className="icon" />
            <span className="location-label">
              {location || "위치를 선택하세요"}
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
            placeholder="원하는 스타일이나 요청사항을 입력하세요."
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
        </div>

        {/* 제출 버튼 */}
        <button className="submit-btn" onClick={handleSubmit}>
          추천받기
        </button>
      </div>

      {/* 모달 */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">위치 선택</h3>
            <input
              type="text"
              placeholder="원하는 위치 검색"
              className="modal-search"
            />
            <div className="modal-map">
              <KakaoMapModal onSelect={(addr) => setLocation(addr)} />
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
