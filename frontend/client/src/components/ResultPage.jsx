import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  const requestPayload = state?.requestPayload ?? null;

  const [data, setData] = useState(() => ({
    location: state?.location ?? "서울",
    weather: state?.weather ?? { description: "맑음", temp: 23, humidity: 43, wind: 9 },
    recommended_images:
      state?.recommended_images ??
      [
        "https://images.pexels.com/photos/2705759/pexels-photo-2705759.jpeg",
        "https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg",
        "https://images.pexels.com/photos/3771069/pexels-photo-3771069.jpeg",
        "https://images.pexels.com/photos/270408/pexels-photo-270408.jpeg",
        "https://images.pexels.com/photos/2983464/pexels-photo-2983464.jpeg",
      ],
    styling_tip:
      state?.styling_tip ??
      "오늘 같은 날씨에는 가벼운 레이어드 룩을 추천합니다. 오후에 온도가 올라갈 수 있으니 겉옷을 준비하세요.",
  }));

  const { location, weather, recommended_images, styling_tip } = data;

  // 슬라이더
  const [index, setIndex] = useState(0);
  const stripRef = useRef(null);
  const thumbRefs = useRef([]);

  useEffect(() => {
    const el = thumbRefs.current[index];
    if (el && stripRef.current) {
      el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }
  }, [index]);

  const next = () => setIndex((i) => (i + 1) % recommended_images.length);
  const prev = () => setIndex((i) => (i - 1 + recommended_images.length) % recommended_images.length);

  const icon = (d) => {
    const s = (d || "").toLowerCase();
    if (s.includes("rain") || s.includes("비")) return "🌧️";
    if (s.includes("snow") || s.includes("눈")) return "❄️";
    if (s.includes("cloud") || s.includes("구름")) return "☁️";
    if (s.includes("맑") || s.includes("clear")) return "☀️";
    return "🌤️";
  };

  // 다시 받기: 백엔드로 재요청
  const [loading, setLoading] = useState(false);
  const onRetry = async () => {
    if (!requestPayload) return navigate("/");

    try {
      setLoading(true);
      const url = (process.env.REACT_APP_API_URL || "http://127.0.0.1:5000").replace(/\/$/, "");
      const res = await axios.post(`${url}/api/recommend`, requestPayload, {
        headers:
          requestPayload instanceof FormData
            ? { "Content-Type": "multipart/form-data" }
            : { "Content-Type": "application/json" },
      });
      const r = res.data || {};
      setData({
        location: r.location ?? location,
        weather: r.weather ?? weather,
        recommended_images: r.recommended_images ?? recommended_images,
        styling_tip: r.styling_tip ?? styling_tip,
      });
      setIndex(0);
    } catch (e) {
      console.error(e);
      alert("다시 받기 요청 중 오류가 발생했어요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="result-shell">
      <div className="result-card">
        <div className="result-head">
          <h1 className="result-title">OOTD-AI</h1>
          <button className="retry-link" onClick={onRetry} disabled={loading}>
            {loading ? "요청 중..." : "다시 받기"}
          </button>
        </div>

        <div className="result-sub">
          <div className="pin">📍</div>
          <div className="where">{location || "위치 미지정"}</div>
        </div>

        <div className="weather-chip">
          <div className="weather-left">
            <span className="w-emoji">{icon(weather?.description)}</span>
            <div className="w-main">
              <div className="w-desc">{weather?.description ?? "-"}</div>
              <div className="w-temp">{weather?.temp ?? "-"}°C</div>
            </div>
          </div>
          <div className="weather-right">
            <div className="w-meta">습도: {weather?.humidity ?? "-"}%</div>
            <div className="w-meta">바람: {weather?.wind ?? "-"} m/s</div>
          </div>
        </div>

        <div className="viewer">
          <button className="nav-btn left" onClick={prev} aria-label="prev">‹</button>
          <div className="hero">
            <img src={recommended_images[index]} alt={`추천 ${index + 1}`} className="hero-img" />
            <div className="counter">{index + 1}/{recommended_images.length}</div>
          </div>
          <button className="nav-btn right" onClick={next} aria-label="next">›</button>
        </div>

        <div className="thumb-strip" ref={stripRef}>
          {recommended_images.map((src, i) => (
            <button
              key={i}
              ref={(el) => (thumbRefs.current[i] = el)}
              className={`thumb ${i === index ? "active" : ""}`}
              onClick={() => setIndex(i)}
              aria-label={`thumb ${i + 1}`}
            >
              <img src={src} alt={`thumb ${i + 1}`} />
            </button>
          ))}
        </div>

        <div className="tip-card">
          <div className="tip-title">💡 스타일링 팁</div>
          <p className="tip-body">{styling_tip}</p>
        </div>

        <button className="back-btn" onClick={() => navigate("/")}>‹ 뒤로가기</button>
      </div>
    </div>
  );
}
