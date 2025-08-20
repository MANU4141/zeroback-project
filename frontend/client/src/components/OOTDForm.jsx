// src/components/OOTDForm.jsx
import "../css/OOTDForm.css";
import { useMemo, useState } from "react";
import { FaMapMarkerAlt, FaCloudUploadAlt } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";
import axios from "axios";

/**
 * OOTDForm
 * - 위치 선택(카카오 맵 모달)
 * - 스타일 입력(자동완성 + 칩)
 * - 추가 요청사항
 * - 이미지 업로드(드래그&드롭/미리보기/개별삭제/전체삭제)
 * - 제출 → /api/recommend 로 FormData 전송
 * - 결과 페이지로 이동 시 재요청 대비용 requestPayload(state) 함께 전달
 */
export default function OOTDForm() {
  const navigate = useNavigate();

  // 위치/좌표/모달
  const [location, setLocation] = useState("");
  const [coords, setCoords] = useState({ lat: null, lng: null });
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 스타일 입력/선택
  const [styles, setStyles] = useState([]);
  const [styleInput, setStyleInput] = useState("");
  const [suggestOpen, setSuggestOpen] = useState(false);

  // 요청사항
  const [request, setRequest] = useState("");

  // 이미지
  const [images, setImages] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  // 스타일 마스터 (자동완성 소스)
  const STYLE_MASTER = useMemo(
    () => [
      "레트로","로맨틱","리소트","매니시","모던","밀리터리","섹시","소피스트케이티드",
      "스트리트","스포티","아방가르드","오리엔탈","웨스턴","젠더리스","컨트리","클래식",
      "키치","톰보이","펑크","페미닌","프레피","히피","힙합",
    ],
    []
  );

  // 기본 추천 세트 (전체선택)
  const COMMON_DEFAULTS = [...STYLE_MASTER];

  // 스타일 토글
  const toggleStyle = (style) => {
    setStyles((prev) =>
      prev.includes(style) ? prev.filter((s) => s !== style) : [...prev, style]
    );
  };

  // 스타일 입력 엔터 → 칩 추가
  const handleStyleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const v = styleInput.trim();
      if (!v) return;
      if (!styles.includes(v)) setStyles((p) => [...p, v]);
      setStyleInput("");
      setSuggestOpen(false);
    }
  };

  // 자동완성 필터
  const filteredSuggestions = useMemo(() => {
    if (!styleInput.trim()) return [];
    const q = styleInput.trim().toLowerCase();
    return STYLE_MASTER.filter(
      (s) => s.toLowerCase().includes(q) && !styles.includes(s)
    ).slice(0, 8);
  }, [STYLE_MASTER, styleInput, styles]);

  // 업로드 핸들러
  const handleImageUpload = (fileList) => {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    setImages((prev) => [...prev, ...files]);
  };
  const onInputChange = (e) => handleImageUpload(e.target.files);
  const clearImages = () => setImages([]);
  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleImageUpload(e.dataTransfer.files);
  };
  const removeImageAt = (idx) => {
    setImages((prev) => prev.filter((_, i) => i !== idx));
  };

  // 제출
  const handleSubmit = async () => {
    if (!location || styles.length === 0) {
      alert("위치와 스타일을 선택/입력해주세요.");
      return;
    }

    // JSON 데이터
    const requestData = {
      location,
      latitude: coords.lat,
      longitude: coords.lng,
      style_select: styles,
      user_request: request || "",
    };

    // FormData 구성 (이미지 + JSON)
    const formData = new FormData();
    formData.append("data", JSON.stringify(requestData));
    images.forEach((img) => formData.append("images", img));

    try {
      const apiUrl = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
      const res = await axios.post(`${apiUrl}/api/recommend`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // 결과 페이지로 이동
      // ?? FormData는 새로고침 시 소실될 수 있음(클라이언트 메모리) → 재요청 전용으로만 사용
      navigate("/result", { state: { ...res.data, requestPayload: requestData } });
    } catch (err) {
      console.error("API 요청 오류:", err);
      alert("추천 요청 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className={`ootd-container ${isModalOpen ? "blurred" : ""}`}>
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI가 추천하는 오늘의 완벽한 룩</p>

      <div className="form-card">
        {/* 위치 */}
        <div className="form-section">
          <label className="form-label">위치 선택</label>
          <div className="location-box" onClick={() => setIsModalOpen(true)}>
            <FaMapMarkerAlt className="icon" />
            <span className="location-label">{location || "위치를 선택해주세요"}</span>
          </div>
        </div>

        {/* 스타일 */}
        <div className="form-section">
          <div className="style-header">
            <label className="form-label">스타일 (중복 선택 가능)</label>
            <div className="style-actions">
              <button
                type="button"
                className="link-btn"
                onClick={() =>
                  setStyles((prev) => [...new Set([...prev, ...COMMON_DEFAULTS])])
                }
              >
                전체 선택
              </button>
              <span className="dot">·</span>
              <button type="button" className="link-btn" onClick={() => setStyles([])}>
                초기화
              </button>
            </div>
          </div>

          {/* 입력 → 칩/자동완성 */}
          <div
            className="chip-input only-input"
            onFocus={() => setSuggestOpen(true)}
            onBlur={() => setTimeout(() => setSuggestOpen(false), 120)}
          >
            <input
              value={styleInput}
              onChange={(e) => setStyleInput(e.target.value)}
              onKeyDown={handleStyleKeyDown}
              placeholder="스타일을 입력해주세요 (예: 스포티, 모던…)"
            />
          </div>

          {suggestOpen && filteredSuggestions.length > 0 && (
            <ul className="suggest-list">
              {filteredSuggestions.map((s) => (
                <li
                  key={s}
                  className="suggest-item"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    toggleStyle(s);
                    setStyleInput("");
                    setSuggestOpen(false);
                  }}
                >
                  <span className="suggest-dot" />
                  <span className="suggest-text">{s}</span>
                </li>
              ))}
            </ul>
          )}

          {styles.length > 0 && (
            <div className="selected-styles">
              {styles.map((tag, i) => (
                <span className="style-tag" key={`${tag}-${i}`}>
                  {tag}
                  <button
                    type="button"
                    className="style-tag-x"
                    onClick={() => toggleStyle(tag)}
                    aria-label={`${tag} 제거`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* 요청사항 */}
        <div className="form-section">
          <label className="form-label">추가 요청사항</label>
          <textarea
            className="text-input text-input--short"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
        </div>

        {/* 업로드 */}
        <div className="form-section">
          <label className="form-label">이미지/스냅 업로드 (여러 개 가능)</label>

          <div
            className={`upload-card ${dragOver ? "drag" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <FaCloudUploadAlt className="upload-icon" />
            <div className="upload-title">파일을 끌어다 놓거나 클릭하여 업로드</div>
            <div className="upload-sub">JPG · PNG · WEBP 지원, 여러 장 가능</div>

            <label htmlFor="image-upload" className="file-upload-btn">
              이미지 선택
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              multiple
              onChange={onInputChange}
              className="image-upload"
            />
          </div>

          {/* 미리보기 */}
          {images.length > 0 && (
            <div className="preview-grid">
              {images.map((file, idx) => {
                const src = URL.createObjectURL(file);
                return (
                  <div className="preview-box" key={`${file.name}-${idx}`}>
                    <img
                      src={src}
                      alt={`preview-${idx}`}
                      onLoad={() => URL.revokeObjectURL(src)}
                    />
                    <button
                      className="remove-btn"
                      onClick={() => removeImageAt(idx)}
                      title="삭제"
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          <div className="file-upload-info">
            {images.length > 0 ? (
              <>
                {images.length}개 선택됨{" "}
                <button type="button" className="link-btn" onClick={clearImages}>
                  모두 제거
                </button>
              </>
            ) : (
              "선택된 파일 없음"
            )}
          </div>
        </div>

        {/* 제출 */}
        <button className="submit-btn" onClick={handleSubmit}>
          추천받기
        </button>
      </div>

      {/* 위치 선택 모달 */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">위치 선택</h3>
            <div className="modal-map">
              <KakaoMapModal
                onSelect={(data) => {
                  setLocation(data.address);
                  setCoords({ lat: data.lat, lng: data.lng });
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