import React, { useEffect } from "react";

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
      script.onload = () => {
        resolve(window.kakao);
      };
    }
  });
};

export default function KakaoMapModal({ onSelect }) {
  useEffect(() => {
    loadKakaoSDK().then((kakao) => {
      kakao.maps.load(() => {
        const container = document.getElementById("kakao-map");
        if (!container) return;

        const options = {
          center: new kakao.maps.LatLng(37.5665, 126.9780),
          level: 3,
        };

        const map = new kakao.maps.Map(container, options);
        const geocoder = new kakao.maps.services.Geocoder();
        const marker = new kakao.maps.Marker({ map });

        kakao.maps.event.addListener(map, "click", (mouseEvent) => {
          const latlng = mouseEvent.latLng;
          marker.setPosition(latlng);

          geocoder.coord2Address(latlng.getLng(), latlng.getLat(), (result, status) => {
            if (status === kakao.maps.services.Status.OK) {
              onSelect(result[0].address.address_name);
            }
          });
        });
      });
    });
  }, [onSelect]);

  return (
    <div
      id="kakao-map"
      style={{ width: "100%", height: "300px", background: "#f1f1f1" }}
    ></div>
  );
}
