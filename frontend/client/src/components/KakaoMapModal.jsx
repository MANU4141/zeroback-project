import React, { useEffect, useState } from "react";
import "../css/KakaoMapModal.css";

export default function KakaoMapModal({ onSelect, onClose }) {
  const [mapInstance, setMapInstance] = useState(null);
  const [marker, setMarker] = useState(null);
  const [searchQuery, setSearchQuery] = useState(""); // 검색어 상태
  const [selectedLocation, setSelectedLocation] = useState(null);

  useEffect(() => {
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

    loadKakaoSDK().then((kakao) => {
      kakao.maps.load(() => {
        const container = document.getElementById("kakao-map");
        if (!container) return;

        // ✅ 기본 위치: 현위치
        let defaultLatLng = new kakao.maps.LatLng(37.5665, 126.9780); // 서울시청
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(
            (pos) => {
              defaultLatLng = new kakao.maps.LatLng(pos.coords.latitude, pos.coords.longitude);
              initMap(defaultLatLng);
            },
            () => initMap(defaultLatLng)
          );
        } else {
          initMap(defaultLatLng);
        }

        function initMap(centerLatLng) {
          const map = new kakao.maps.Map(container, { center: centerLatLng, level: 3 });
          const newMarker = new kakao.maps.Marker({ position: centerLatLng, map });
          setMapInstance(map);
          setMarker(newMarker);

          // ✅ 초기 선택 위치 저장
          const geocoder = new kakao.maps.services.Geocoder();
          geocoder.coord2Address(centerLatLng.getLng(), centerLatLng.getLat(), (result, status) => {
            if (status === kakao.maps.services.Status.OK) {
              setSelectedLocation({
                address: result[0].address.address_name,
                lat: centerLatLng.getLat(),
                lng: centerLatLng.getLng(),
              });
            }
          });

          // ✅ 지도 클릭 시 선택 변경
          kakao.maps.event.addListener(map, "click", (mouseEvent) => {
            const latlng = mouseEvent.latLng;
            newMarker.setPosition(latlng);
            geocoder.coord2Address(latlng.getLng(), latlng.getLat(), (result, status) => {
              if (status === kakao.maps.services.Status.OK) {
                setSelectedLocation({
                  address: result[0].address.address_name,
                  lat: latlng.getLat(),
                  lng: latlng.getLng(),
                });
              }
            });
          });
        }
      });
    });
  }, []);

  // ✅ 주요 도시 좌표
  const cityCenters = {
    서울: { lat: 37.5665, lng: 126.9780 },
    부산: { lat: 35.1796, lng: 129.0756 },
    대구: { lat: 35.8714, lng: 128.6014 },
    인천: { lat: 37.4563, lng: 126.7052 },
    광주: { lat: 35.1595, lng: 126.8526 },
    대전: { lat: 36.3504, lng: 127.3845 },
  };

  // ✅ 도시 버튼 클릭 시 이동
  const moveToCity = (lat, lng, cityName) => {
    if (mapInstance && marker) {
      const coords = new window.kakao.maps.LatLng(lat, lng);
      mapInstance.setCenter(coords);
      marker.setPosition(coords);

      // ✅ 선택한 도시 이름을 주소로 설정
      setSelectedLocation({
        address: `${cityName} 시청`,
        lat: lat,
        lng: lng,
      });
    }
  };

  // ✅ 확인 버튼 클릭 시 처리
  const handleConfirm = () => {
    if (!mapInstance || !marker) return;

    if (searchQuery.trim()) {
      // ✅ 검색어 입력 → 검색 후 위치 이동
      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.addressSearch(searchQuery, (result, status) => {
        if (status === window.kakao.maps.services.Status.OK) {
          const coords = new window.kakao.maps.LatLng(result[0].y, result[0].x);
          mapInstance.setCenter(coords);
          marker.setPosition(coords);
          const address = result[0].address.address_name;

          onSelect({ address, lat: parseFloat(result[0].y), lng: parseFloat(result[0].x) });
          onClose(); // 모달 닫기
        } else {
          alert("검색 결과를 찾을 수 없습니다.");
        }
      });
    } else if (selectedLocation) {
      // ✅ 지도에서 클릭한 위치 사용
      onSelect(selectedLocation);
      onClose();
    } else {
      alert("위치를 선택하거나 검색하세요.");
    }
  };

  return (
    <div>
      {/* 검색 입력 */}
      <div className="search-wrapper">
        <input
          type="text"
          placeholder="원하는 위치 검색"
          className="modal-search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      {/* 지도 */}
      <div id="kakao-map" className="kakao-map"></div>

      {/* ✅ 주요 도시 버튼 */}
      <div className="city-grid">
        {Object.keys(cityCenters).map((city) => (
          <button
            key={city}
            className="city-btn"
            onClick={() => moveToCity(cityCenters[city].lat, cityCenters[city].lng, city)}
          >
            {city}
          </button>
        ))}
      </div>

      {/* 확인 버튼 */}
      <button className="modal-confirm" onClick={handleConfirm}>
        확인
      </button>
    </div>
  );
}
