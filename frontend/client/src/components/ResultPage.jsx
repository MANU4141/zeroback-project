// src/components/ResultPage.jsx
import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "../css/ResultPage.css";

/**
 * ResultPage
 * - location, weather, recommended_images, styling_tip 은 반드시 백엔드/이전 페이지에서 전달된 값 사용
 * - recommended_images: "파일명"/"객체"/"절대URL" 모두 수용 → 화면용 절대URL로 변환
 * - 다시 받기: requestPayload(폼에서 넘긴 FormData/JSON)로 /api/recommend 재요청
 */
export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  // API 베이스 (환경변수 > 로컬 기본)
  const API_BASE = (process.env.REACT_APP_API_URL || "http://127.0.0.1:5000").replace(/\/$/, "");

  // 문자열/객체/파일명을 화면 표시용 절대 URL로 변환
  const toImageUrl = (item) => {
    if (!item) return "";
    if (typeof item === "string") {
      if (/^https?:\/\//i.test(item)) return item;        // 절대 URL
      return `${API_BASE}/api/images/${item}`;            // 파일명 → /api/images/{파일명}
    }
    if (typeof item === "object") {
      const p = item.img_path || item.path || item.filename || item.file || item.url || "";
      if (!p) return "";
      if (/^https?:\/\//i.test(p)) return p;
      return `${API_BASE}/api/images/${p}`;
    }
    return "";
  };
  const normalizeImages = (arr) => (Array.isArray(arr) ? arr.map(toImageUrl).filter(Boolean) : []);

  // 재요청 payload (없으면 홈으로)
  const requestPayload = state?.requestPayload ?? null;

  const [data, setData] = useState(() => ({
    location: state?.location ?? "",
    weather: state?.weather ?? {},                // {description, temp, humidity, wind}
    recommended_images: normalizeImages(state?.recommended_images ?? []),
    styling_tip: state?.styling_tip ?? "",
  }));
  const { location, weather, recommended_images, styling_tip } = data;

  // 뷰어 상태
  const [index, setIndex] = useState(0);
  const stripRef = useRef(null);
  const thumbRefs = useRef([]);

  useEffect(() => {
    const el = thumbRefs.current[index];
    if (el && stripRef.current) {
      el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }
  }, [index]);

  const next = () =>
    setIndex((i) => (recommended_images.length ? (i + 1) % recommended_images.length : 0));
  const prev = () =>
    setIndex((i) =>
      recommended_images.length ? (i - 1 + recommended_images.length) % recommended_images.length : 0
    );

  const icon = (d) => {
    const s = String(d || "").toLowerCase();
    if (s.includes("rain") || s.includes("비")) return "??";
    if (s.includes("snow") || s.includes("눈")) return "??";
    if (s.includes("cloud") || s.includes("구름")) return "??";
    if (s.includes("맑") || s.includes("clear")) return "??";
    return "??";
  };

  // 다시 받기
  const [loading, setLoading] = useState(false);
  const onRetry = async () => {
    if (!requestPayload) return navigate("/");

    try {
      setLoading(true);

      // 재요청 시에도 FormData 사용 (이미지는 없음)
      const formData = new FormData();
      formData.append("data", JSON.stringify(requestPayload));

      const res = await axios.post(`${API_BASE}/api/recommend`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const r = res.data || {};
      setData({
        location: r.location ?? location,
        weather: r.weather ?? weather,
        recommended_images: normalizeImages(r.recommended_images ?? []),
        styling_tip: r.styling_tip ?? "",
      });
      setIndex(0);
    } catch (e) {
      console.error(e);
      alert("다시 받기 요청 중 오류가 발생했어요.");
    } finally {
      setLoading(false);
    }
  };

  const hasImages = recommended_images.length > 0;
  const heroSrc = hasImages ? recommended_images[index] : "";

  return (
    <div className="result-shell">
      <div className="result-card">
        {/* 헤더 */}
        <div className="result-head">
          <h1 className="result-title">OOTD-AI</h1>
          <button className="retry-link" onClick={onRetry} disabled={loading}>
            {loading ? "요청 중..." : "다시 받기"}
          </button>
        </div>

        {/* 위치 */}
        <div className="result-sub">
          <div className="pin">?</div>
          <div className="where">{location || "위치 미지정"}</div>
        </div>

        {/* 날씨 */}
        <div className="weather-chip">
          <div className="weather-left">
            <span className="w-emoji">{icon(weather?.description)}</span>
            <div className="w-main">
              <div className="w-desc">{weather?.description ?? "-"}</div>
              <div className="w-temp">
                {weather?.temp !== undefined && weather?.temp !== null ? weather.temp : "-"}°C
              </div>
            </div>
          </div>
          <div className="weather-right">
            <div className="w-meta">
              습도:{" "}
              {weather?.humidity !== undefined && weather?.humidity !== null
                ? weather.humidity
                : "-"}
              %
            </div>
            <div className="w-meta">
              바람: {weather?.wind !== undefined && weather?.wind !== null ? weather.wind : "-"} m/s
            </div>
          </div>
        </div>

        {/* 메인 이미지 & 네비 */}
        <div className="viewer">
          {hasImages ? (
            <>
              <button className="nav-btn left" onClick={prev} aria-label="previous">
                ?
              </button>
              <div className="hero">
                <img src={heroSrc} alt={`추천 ${index + 1}`} className="hero-img" />
                <div className="counter">
                  {index + 1}/{recommended_images.length}
                </div>
              </div>
              <button className="nav-btn right" onClick={next} aria-label="next">
                ?
              </button>
            </>
          ) : (
            <div className="hero">
              <div className="hero-empty">추천 이미지가 없습니다.</div>
            </div>
          )}
        </div>

        {/* 썸네일 */}
        {hasImages && (
          <div className="thumb-strip" ref={stripRef}>
            {recommended_images.map((src, i) => (
              <button
                key={i}
                ref={(el) => (thumbRefs.current[i] = el)}
                className={`thumb ${i === index ? "active" : ""}`}
                onClick={() => setIndex(i)}
                aria-label={`썸네일 ${i + 1}`}
              >
                <img src={src} alt={`썸네일 ${i + 1}`} />
              </button>
            ))}
          </div>
        )}

        {/* 스타일링 팁 */}
        {styling_tip && (
          <div className="tip-card">
            <div className="tip-title">? 스타일링 팁</div>
            <p className="tip-body">{styling_tip}</p>
          </div>
        )}

        <button className="back-btn" onClick={() => navigate("/")}>
          ? 뒤로가기
        </button>
      </div>
    </div>
  );
}