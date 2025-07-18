import { useEffect } from "react";

export default function KakaoMap({ onLocationSelect }) {
  useEffect(() => {
    const scriptCheck = () => {
      if (window.kakao && window.kakao.maps) {
        window.kakao.maps.load(() => {
          const container = document.getElementById("map");
          const options = {
            center: new window.kakao.maps.LatLng(37.5665, 126.9780), // ���� ��ǥ
            level: 3,
          };

          const map = new window.kakao.maps.Map(container, options);

          // ���� Ŭ�� �̺�Ʈ
          window.kakao.maps.event.addListener(map, "click", function (mouseEvent) {
            const latlng = mouseEvent.latLng;
            onLocationSelect(latlng);
          });
        });
      } else {
        console.error("Kakao map library not loaded yet.");
      }
    };

    // ��ũ��Ʈ �ε� ���� ����
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
