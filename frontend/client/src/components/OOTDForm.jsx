import "../css/OOTDForm.css";
import { useMemo, useState } from "react";
import { FaMapMarkerAlt, FaCloudUploadAlt } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";
import axios from "axios";

export default function OOTDForm() {
  const navigate = useNavigate();
  const [location, setLocation] = useState("");
  const [coords, setCoords] = useState({ lat: null, lng: null });
  const [isModalOpen, setIsModalOpen] = useState(false);

  // ✅ 스타일
  const [styles, setStyles] = useState([]);
  const [styleInput, setStyleInput] = useState("");
  const [suggestOpen, setSuggestOpen] = useState(false);

  // ✅ 요청사항
  const [request, setRequest] = useState("");

  // ✅ 이미지
  const [images, setImages] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  // ✅ 스타일 마스터
  const STYLE_MASTER = useMemo(
    () => [
      "스트릿","미니멀","빈티지","캐주얼","러블리","오피스","하이틴","아메카지","스포츠",
      "포멀","시크","댄디","클래식","레트로","럭셔리","고프코어","Y2K","프레피",
      "아웃도어","실용주의","이지웨어","펑크","로맨틱","모던","걸리시","보헤미안",
      "그런지","캠퍼스룩","리조트룩","골프룩","테크웨어","원마일웨어","힙합","스케이터",
      "페미닌","젠더리스","아방가르드","미니멀테일러드","리넨룩","모노톤","톤온톤",
      "컬러포인트","오버핏","슬림핏","레더","데님","밀리터리","워크웨어"
    ],
    []
  );

  const COMMON_DEFAULTS = ["스트릿","미니멀","빈티지","캐주얼","러블리","오피스","하이틴","아메카지","스포츠",
      "포멀","시크","댄디","클래식","레트로","럭셔리","고프코어","Y2K","프레피",
      "아웃도어","실용주의","이지웨어","펑크","로맨틱","모던","걸리시","보헤미안",
      "그런지","캠퍼스룩","리조트룩","골프룩","테크웨어","원마일웨어","힙합","스케이터",
      "페미닌","젠더리스","아방가르드","미니멀테일러드","리넨룩","모노톤","톤온톤",
      "컬러포인트","오버핏","슬림핏","레더","데님","밀리터리","워크웨어"];

  const toggleStyle = (style) => {
    setStyles((prev) =>
      prev.includes(style) ? prev.filter((s) => s !== style) : [...prev, style]
    );
  };

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

  const filteredSuggestions = useMemo(() => {
    if (!styleInput.trim()) return [];
    const q = styleInput.trim().toLowerCase();
    return STYLE_MASTER.filter(
      (s) => s.toLowerCase().includes(q) && !styles.includes(s)
    ).slice(0, 8);
  }, [STYLE_MASTER, styleInput, styles]);

  // ✅ 업로드
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

  // ✅ 제출
  const handleSubmit = async () => {
    if (!location || styles.length === 0) {
      alert("위치와 스타일을 선택/입력해주세요.");
      return;
    }

    const requestData = {
      location,
      latitude: coords.lat,
      longitude: coords.lng,
      style_select: styles,
      user_request: request || ""
    };

    const formData = new FormData();
    formData.append("data", JSON.stringify(requestData));
    images.forEach((img) => formData.append("images", img));

    try {
      const res = await axios.post("http://127.0.0.1:5000/api/recommend", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      navigate("/result", { state: res.data });
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
            <span className="location-label">
              {location || "위치를 선택해주세요"}
            </span>
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

          {/* 입력 */}
          <div
            className="chip-input only-input"
            onFocus={() => setSuggestOpen(true)}
            onBlur={() => setTimeout(() => setSuggestOpen(false), 120)}
          >
            <input
              value={styleInput}
              onChange={(e) => setStyleInput(e.target.value)}
              onKeyDown={handleStyleKeyDown}
              placeholder="스타일을 입력해주세요 (예: 캐주얼, 미니멀…)"
            />
          </div>

          {/* 자동완성 */}
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

          {/* 선택된 스타일 표시 */}
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

          {/* ✅ 미리보기 그리드 */}
          {images.length > 0 && (
            <div className="preview-grid">
              {images.map((file, idx) => {
                const src = URL.createObjectURL(file);
                return (
                  <div className="preview-box" key={`${file.name}-${idx}`}>
                    <img src={src} alt={`preview-${idx}`} onLoad={() => URL.revokeObjectURL(src)} />
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

        {/* 테스트 이동 */}
        <button
          className="test-btn"
          onClick={() => navigate("/result")}
          title="백엔드 연결 전 레이아웃만 확인"
        >
          🔍 테스트용 결과 페이지 열기
        </button>
      </div>

      {/* 모달 */}
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
