import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation(); // OOTDForm���� ������ ������
  const data = state || {};

  if (!data.success) {
    return (
      <div className="result-container">
        <h2>��õ ����� �ҷ��� �� �����ϴ�.</h2>
        <button className="back-btn" onClick={() => navigate("/")}>
          �ٽ� �õ��ϱ�
        </button>
      </div>
    );
  }

  return (
    <div className="result-container">
      <h1 className="title">������ ��õ �ڵ�</h1>

      {/* ? ���� ���� */}
      <div className="weather-box">
        <h3>���� ����</h3>
        <p>?? �µ�: {data.weather.temperature}��C</p>
        <p>?? ����: {data.weather.condition}</p>
        <p>? ����: {data.weather.humidity}%</p>
        <p>? ǳ��: {data.weather.wind_speed} m/s</p>
      </div>

      {/* ? ��õ �ڵ� */}
      <div className="recommend-box">
        <h3>��õ ��Ÿ��</h3>
        <div className="recommend-grid">
          {data.recommendations && data.recommendations.length > 0 ? (
            data.recommendations.map((item, index) => (
              <div key={index} className="recommend-card">
                <img src={item.image} alt={item.description} />
                <p>{item.description}</p>
              </div>
            ))
          ) : (
            <p>��õ ��Ÿ���� �����ϴ�.</p>
          )}
        </div>
      </div>

      <button className="back-btn" onClick={() => navigate("/")}>
        �ٽ� �ڵ� �ޱ�
      </button>
    </div>
  );
}
