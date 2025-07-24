import React, { useEffect, useState } from "react";
import "../css/KakaoMapModal.css"; // ✅ CSS 분리

const loadKakaoSDK = () => {
  return new Promise((resolve) => {
    if (window.kakao && window.kakao.maps) {
      resolve(window.kakao);
    } else {
      const script = document.createElement("script");
      script.src =
        "https://dapi.kakao.com/v2/maps/sdk.js?appkey=2380b0c6e08137415b69715b13439001&autoload=false&libraries=services";
      script.async = true;
      document.head.appendChild(script);
      script.onload = () => resolve(window.kakao);
    }
  });
};

export default function KakaoMapModal({ onSelect }) {
  const [mapInstance, setMapInstance] = useState(null);
  const [marker, setMarker] = useState(null);

  useEffect(() => {
    loadKakaoSDK().then((kakao) => {
      kakao.maps.load(() => {
        const container = document.getElementById("kakao-map");
        if (!container) return;

        // ✅ 기본 좌표 (서울시청)
        let defaultLatLng = new kakao.maps.LatLng(37.5665, 126.9780);

        // ✅ Geolocation으로 현재 위치 가져오기
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              defaultLatLng = new kakao.maps.LatLng(
                position.coords.latitude,
                position.coords.longitude
              );
              initMap(defaultLatLng);
            },
            () => {
              initMap(defaultLatLng);
            }
          );
        } else {
          initMap(defaultLatLng);
        }

        function initMap(centerLatLng) {
          const map = new kakao.maps.Map(container, { center: centerLatLng, level: 3 });
          const newMarker = new kakao.maps.Marker({ position: centerLatLng, map });
          setMapInstance(map);
          setMarker(newMarker);

          kakao.maps.event.addListener(map, "click", (mouseEvent) => {
            const latlng = mouseEvent.latLng;
            newMarker.setPosition(latlng);
            const geocoder = new kakao.maps.services.Geocoder();
            geocoder.coord2Address(latlng.getLng(), latlng.getLat(), (result, status) => {
              if (status === kakao.maps.services.Status.OK) {
                onSelect(result[0].address.address_name);
              }
            });
          });
        }
      });
    });
  }, [onSelect]);

  // ✅ 주요 도시 버튼 클릭 시 지도 이동
  const moveToCity = (lat, lng) => {
    if (mapInstance && marker) {
      const moveLatLng = new window.kakao.maps.LatLng(lat, lng);
      mapInstance.setCenter(moveLatLng);
      marker.setPosition(moveLatLng);
    }
  };

  const cityCenters = {
    서울: { lat: 37.5665, lng: 126.9780 },
    부산: { lat: 35.1796, lng: 129.0756 },
    대구: { lat: 35.8714, lng: 128.6014 },
    인천: { lat: 37.4563, lng: 126.7052 },
    광주: { lat: 35.1595, lng: 126.8526 },
    대전: { lat: 36.3504, lng: 127.3845 },
  };

  return (
    <div>
      <div id="kakao-map" className="kakao-map"></div>
      <div className="city-grid">
        {Object.keys(cityCenters).map((city) => (
          <button
            key={city}
            className="city-btn"
            onClick={() => moveToCity(cityCenters[city].lat, cityCenters[city].lng)}
          >
            {city}
          </button>
        ))}
      </div>
    </div>
  );
}
