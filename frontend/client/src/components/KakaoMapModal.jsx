// src/components/KakaoMapModal.jsx
import React, { useEffect, useState } from "react";
import "../css/KakaoMapModal.css";

export default function KakaoMapModal({ onSelect, onClose }) {
  const [map, setMap] = useState(null);
  const [marker, setMarker] = useState(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);

  // SDK 로드 & 지도 초기화
  useEffect(() => {
    const load = () =>
      new Promise((resolve) => {
        if (window.kakao && window.kakao.maps) return resolve(window.kakao);
        const s = document.createElement("script");
        s.src =
          "https://dapi.kakao.com/v2/maps/sdk.js?appkey=2380b0c6e08137415b69715b13439001&autoload=false&libraries=services";
        s.onload = () => resolve(window.kakao);
        document.head.appendChild(s);
      });

    load().then((kakao) => {
      kakao.maps.load(() => {
        const container = document.getElementById("kakao-map");
        if (!container) return;

        // 기본: 현위치 → 실패 시 서울시청
        let center = new kakao.maps.LatLng(37.5665, 126.978);
        const init = () => {
          const m = new kakao.maps.Map(container, { center, level: 3 });
          const mk = new kakao.maps.Marker({ map: m, position: center });
          setMap(m);
          setMarker(mk);

          const geocoder = new kakao.maps.services.Geocoder();
          geocoder.coord2Address(center.getLng(), center.getLat(), (res, st) => {
            if (st === kakao.maps.services.Status.OK) {
              setSelected({
                address: res[0].address.address_name,
                lat: center.getLat(),
                lng: center.getLng(),
              });
            }
          });

          kakao.maps.event.addListener(m, "click", (e) => {
            const latlng = e.latLng;
            mk.setPosition(latlng);
            geocoder.coord2Address(latlng.getLng(), latlng.getLat(), (res, st) => {
              if (st === kakao.maps.services.Status.OK) {
                setSelected({
                  address: res[0].address.address_name,
                  lat: latlng.getLat(),
                  lng: latlng.getLng(),
                });
              }
            });
          });
        };

        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              center = new kakao.maps.LatLng(pos.coords.latitude, pos.coords.longitude);
              init();
            },
            init
          );
        } else {
          init();
        }
      });
    });
  }, []);

  const cityCenters = {
    서울: { lat: 37.5665, lng: 126.978 },
    부산: { lat: 35.1796, lng: 129.0756 },
    대구: { lat: 35.8714, lng: 128.6014 },
    인천: { lat: 37.4563, lng: 126.7052 },
    광주: { lat: 35.1595, lng: 126.8526 },
    대전: { lat: 36.3504, lng: 127.3845 },
  };

  const moveTo = (lat, lng, label) => {
    if (!map || !marker) return;
    const kakao = window.kakao;
    const pos = new kakao.maps.LatLng(lat, lng);
    map.setCenter(pos);
    marker.setPosition(pos);
    setSelected({ address: `${label} 시청`, lat, lng });
  };

  const handleConfirm = () => {
    if (!map || !marker) return;

    if (search.trim()) {
      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.addressSearch(search, (res, st) => {
        if (st === window.kakao.maps.services.Status.OK) {
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

    if (selected) {
      onSelect(selected);
      onClose();
    } else {
      alert("지도를 클릭하거나 검색어를 입력하세요.");
    }
  };

  return (
    <div>
      <div className="search-wrapper">
        <input
          className="modal-search"
          placeholder="원하는 위치 검색"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div id="kakao-map" className="kakao-map" />

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

      <button className="modal-confirm" onClick={handleConfirm}>확인</button>
    </div>
  );
}
