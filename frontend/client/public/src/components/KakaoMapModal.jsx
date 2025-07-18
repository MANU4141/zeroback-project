import { useEffect } from "react";

export default function KakaoMap({ onLocationSelect }) {
  useEffect(() => {
    const scriptCheck = () => {
      if (window.kakao && window.kakao.maps) {
        window.kakao.maps.load(() => {
          const container = document.getElementById("map");
          const options = {
            center: new window.kakao.maps.LatLng(37.5665, 126.9780), // 서울 좌표
            level: 3,
          };

          const map = new window.kakao.maps.Map(container, options);

          // 지도 클릭 이벤트
          window.kakao.maps.event.addListener(map, "click", function (mouseEvent) {
            const latlng = mouseEvent.latLng;
            onLocationSelect(latlng);
          });
        });
      } else {
        console.error("Kakao map library not loaded yet.");
      }
    };

    // 스크립트 로딩 이후 실행
    if (document.readyState === "complete") {
      scriptCheck();
    } else {
      window.addEventListener("load", scriptCheck);
      return () => window.removeEventListener("load", scriptCheck);
    }
  }, [onLocationSelect]);

  return (
    <div
      id="map"
      style={{
        width: "100%",
        height: "300px",
        borderRadius: "12px",
        marginBottom: "16px",
      }}
    />
  );
}
