// src/components/KakaoMapModal.jsx
import React, { useEffect, useState } from "react";
import "../css/KakaoMapModal.css";

/**
 * KakaoMapModal
 * - 카카오 JS SDK 동적 로드
 * - 현위치(실패 시 서울시청) 기준 지도/마커 생성
 * - 지도 클릭, 주소 검색, 주요 도시 버튼 이동 지원
 * - 확인 시 부모에 {address, lat, lng} 전달
 */
export default function KakaoMapModal({ onSelect, onClose }) {
  const [map, setMap] = useState(null);
  const [marker, setMarker] = useState(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    let kakaoScript;

    // 1) Kakao SDK 로드 (이미 로드되어 있으면 재사용)
    const loadKakao = () =>
      new Promise((resolve) => {
        if (window.kakao && window.kakao.maps) return resolve(window.kakao);
        kakaoScript = document.createElement("script");
        kakaoScript.src =
          "https://dapi.kakao.com/v2/maps/sdk.js?appkey=2380b0c6e08137415b69715b13439001&autoload=false&libraries=services";
        kakaoScript.onload = () => resolve(window.kakao);
        document.head.appendChild(kakaoScript);
      });

    // 2) 지도 초기화
    loadKakao().then((kakao) => {
      kakao.maps.load(() => {
        const container = document.getElementById("kakao-map");
        if (!container) return;

        // 기본 중심(서울시청). 현위치 사용 시 재설정
        let center = new kakao.maps.LatLng(37.5665, 126.978);

        const init = () => {
          const m = new kakao.maps.Map(container, { center, level: 3 });
          const mk = new kakao.maps.Marker({ map: m, position: center });
          setMap(m);
          setMarker(mk);

          const geocoder = new kakao.maps.services.Geocoder();

          // 초기 선택 정보 세팅
          geocoder.coord2Address(center.getLng(), center.getLat(), (res, st) => {
            if (st === kakao.maps.services.Status.OK && res?.[0]?.address) {
              setSelected({
                address: res[0].address.address_name,
                lat: center.getLat(),
                lng: center.getLng(),
              });
            }
          });

          // 지도 클릭 → 마커 이동 + 역지오코딩
          const handleClick = (e) => {
            const latlng = e.latLng;
            mk.setPosition(latlng);
            geocoder.coord2Address(latlng.getLng(), latlng.getLat(), (res, st) => {
              if (st === kakao.maps.services.Status.OK && res?.[0]?.address) {
                setSelected({
                  address: res[0].address.address_name,
                  lat: latlng.getLat(),
                  lng: latlng.getLng(),
                });
              }
            });
          };

          kakao.maps.event.addListener(m, "click", handleClick);

          // 정리 함수: 리스너 제거 (카카오 SDK는 명시적 remove가 없어 지도 DOM 해제만 고려)
          return () => {
            // eslint-disable-next-line no-unused-expressions
            m && kakao.maps.event.removeListener && kakao.maps.event.removeListener(m, "click", handleClick);
          };
        };

        // 현위치 → 실패 시 기본 중심
        let cleanup;
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              center = new kakao.maps.LatLng(pos.coords.latitude, pos.coords.longitude);
              cleanup = init();
            },
            () => {
              cleanup = init();
            }
          );
        } else {
          cleanup = init();
        }

        // Unmount 시 정리
        return () => {
          cleanup && cleanup();
        };
      });
    });

    return () => {
      // 스크립트 태그 제거 (이미 전역 로드됐으면 남겨도 무방하나, 모달 마운트/언마운트 온전히 관리하려면 제거)
      if (kakaoScript && kakaoScript.parentNode) {
        kakaoScript.parentNode.removeChild(kakaoScript);
      }
    };
  }, []);

  // 주요 도시 좌표
  const cityCenters = {
    서울: { lat: 37.5665, lng: 126.978 },
    부산: { lat: 35.1796, lng: 129.0756 },
    대구: { lat: 35.8714, lng: 128.6014 },
    인천: { lat: 37.4563, lng: 126.7052 },
    광주: { lat: 35.1595, lng: 126.8526 },
    대전: { lat: 36.3504, lng: 127.3845 },
  };

  // 도시 버튼 클릭 → 센터/마커 이동 + 선택 갱신
  const moveTo = (lat, lng, label) => {
    if (!map || !marker) return;
    const kakao = window.kakao;
    const pos = new kakao.maps.LatLng(lat, lng);
    map.setCenter(pos);
    marker.setPosition(pos);
    setSelected({ address: `${label} 시청`, lat, lng });
  };

  // 확인
  const handleConfirm = () => {
    if (!map || !marker) return;

    // 검색 우선
    if (search.trim()) {
      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.addressSearch(search, (res, st) => {
        if (st === window.kakao.maps.services.Status.OK && res?.[0]) {
          const lat = parseFloat(res[0].y);
          const lng = parseFloat(res[0].x);
          onSelect({ address: res[0].address.address_name, lat, lng });
          onClose();
        } else {
          alert("검색 결과가 없습니다.");
        }
      });
      return;
    }

    // 지도에서 선택된 위치 사용
    if (selected) {
      onSelect(selected);
      onClose();
    } else {
      alert("지도를 클릭하거나 검색어를 입력하세요.");
    }
  };

  return (
    <div>
      {/* 검색 입력 */}
      <div className="search-wrapper">
        <input
          className="modal-search"
          placeholder="원하는 위치 검색"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* 지도 */}
      <div id="kakao-map" className="kakao-map" />

      {/* 도시 버튼 */}
      <div className="city-grid">
        {Object.keys(cityCenters).map((c) => (
          <button
            key={c}
            className="city-btn"
            onClick={() => moveTo(cityCenters[c].lat, cityCenters[c].lng, c)}
          >
            {c}
          </button>
        ))}
      </div>

      <button className="modal-confirm" onClick={handleConfirm}>
        확인
      </button>
    </div>
  );
}