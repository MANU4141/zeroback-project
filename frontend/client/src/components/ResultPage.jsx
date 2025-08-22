import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "../css/ResultPage.css";
import {
  FaMapMarkerAlt,
  FaSun,
  FaCloud,
  FaCloudRain,
  FaSnowflake,
  FaChevronLeft,
  FaChevronRight,
  FaArrowLeft,
  FaLightbulb,
} from "react-icons/fa";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  // API 베이스 (환경변수 사용, 기본값은 /api)
  const API_BASE = process.env.REACT_APP_API_BASE || "/api";

  // 이미지 URL 변환 함수
  const toImageUrl = (item) => {
    if (!item) return "";

    let p = "";
    if (typeof item === "string") {
      p = item;
    } else if (typeof item === "object") {
      p = item.img_path || item.path || item.filename || item.file || item.url || "";
    }

    if (!p) return "";
    if (/^https?:\/\//i.test(p)) return p;

    const filename = p.split(/[\\/]/).pop();
    if (!filename) return "";

    return `${API_BASE}/api/images/${filename}`;
  };
  const normalizeImages = (arr) => (Array.isArray(arr) ? arr.map(toImageUrl).filter(Boolean) : []);

  // 재요청 payload 상태화 (업데이트 persist)
  const [requestPayload, setRequestPayload] = useState(state?.requestPayload ?? null);

  // 페이지네이션 상태
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // state에서 페이지 정보 초기화 (비동기 반영 보장)
  useEffect(() => {
    setCurrentPage(state?.current_page ?? 1);
    setTotalPages(state?.total_pages ?? 1);
  }, [state]);

  const [data, setData] = useState(() => ({
    location: state?.location ?? state?.requestPayload?.location ?? "",
    weather: state?.weather ?? {},
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
    if (s.includes("rain") || s.includes("비")) return <FaCloudRain />;
    if (s.includes("snow") || s.includes("눈")) return <FaSnowflake />;
    if (s.includes("cloud") || s.includes("구름") || s.includes("흐림")) return <FaCloud />;
    if (s.includes("맑") || s.includes("clear")) return <FaSun />;
    return <FaSun />;
  };

  // 다시 받기 (다음 페이지) – 디버깅 로그 추가, 비활성화 로직 강화
  const [loading, setLoading] = useState(false);
  const onRetry = async () => {
    if (!requestPayload || totalPages <= 1) {
      console.log("No more pages: totalPages =", totalPages);
      return;
    }

    try {
      setLoading(true);
      const nextPage = currentPage + 1 > totalPages ? 1 : currentPage + 1;
      const updatedPayload = { ...requestPayload, page: nextPage };
      setRequestPayload(updatedPayload);
      console.log("Requesting page:", nextPage, "Current totalPages:", totalPages);

      const formData = new FormData();
      formData.append("data", JSON.stringify(updatedPayload));

      const res = await axios.post(`${API_BASE}/api/recommend`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const r = res.data || {};
      console.log("Response total_pages:", r.total_pages);  // 디버깅: 응답 값 확인

      setData({
        location: r.location ?? location,
        weather: r.weather ?? weather,
        recommended_images: normalizeImages(r.recommended_images ?? []),
        styling_tip: r.styling_tip ?? "",
      });
      setCurrentPage(r.current_page ?? nextPage);
      setTotalPages(r.total_pages ?? 1);  // 응답에서 업데이트 보장
      setIndex(0);
    } catch (e) {
      console.error(e);
      alert("오류 발생");
      setCurrentPage(currentPage);  // 롤백
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
          <button className="retry-link" onClick={onRetry} disabled={loading || totalPages <= 1}>
            {loading ? "요청 중..." : (totalPages <= 1 ? "더 이상 없음" : "다음 페이지")}
          </button>
        </div>
        {/* 페이지 표시 */}
        <div style={{ marginBottom: 8, fontWeight: 500 }}>페이지: {currentPage} / {totalPages}</div>
        {/* 위치 */}
        <div className="result-sub">
          <div className="pin">
            <FaMapMarkerAlt />
          </div>
          <div className="where">{location || "위치 미지정"}</div>
        </div>

        {/* 날씨 */}
        <div className="weather-chip">
          <div className="weather-left">
            <span className="w-emoji">{icon(weather?.condition)}</span>
            <div className="w-main">
              <div className="w-desc">{weather?.condition ?? "-"}</div>
              <div className="w-temp">
                {weather?.temperature !== undefined && weather?.temperature !== null
                  ? weather.temperature
                  : "-"}
                °C
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
              바람:{" "}
              {weather?.wind_speed !== undefined && weather?.wind_speed !== null
                ? weather.wind_speed
                : "-"}{" "}
              m/s
            </div>
          </div>
        </div>

        {/* 메인 이미지 & 네비 */}
        <div className="viewer">
          {hasImages ? (
            <>
              <button className="nav-btn left" onClick={prev} aria-label="previous">
                <FaChevronLeft />
              </button>
              <div className="hero">
                <img src={heroSrc} alt={`추천 ${index + 1}`} className="hero-img" />
                <div className="counter">
                  {index + 1}/{recommended_images.length}
                </div>
              </div>
              <button className="nav-btn right" onClick={next} aria-label="next">
                <FaChevronRight />
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
            <div className="tip-title">
              <FaLightbulb /> 스타일링 팁
            </div>
            <p className="tip-body">{styling_tip}</p>
          </div>
        )}

        <button className="back-btn" onClick={() => navigate("/")}>
          <FaArrowLeft /> 뒤로가기
        </button>
      </div>
    </div>
  );
}
