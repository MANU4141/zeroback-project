import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../css/ResultPage.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const { state } = useLocation();

  if (!state) {
    return (
      <div className="result-container">
        <h2>��� �����͸� ã�� �� �����ϴ�.</h2>
        <button className="back-btn" onClick={() => navigate("/")}>
          �ٽ� �õ��ϱ�
        </button>
      </div>
    );
  }

  const { location, latitude, longitude, style_select, user_request, recommended_images } = state;

  return (
    <div className="result-container">
      <h1 className="title">��õ ���</h1>
      <p className="subtitle">����� ���� AI �ڵ� ��õ</p>

      {/* ? ������ ���� ǥ�� */}
      <div className="info-box">
        <h3>? ������ ��ġ</h3>
        <p>{location}</p>
        <p>����: {latitude} | �浵: {longitude}</p>

        <h3>? ������ ��Ÿ��</h3>
        <p>{style_select.join(", ")}</p>

        {user_request && (
          <>
            <h3>? �߰� ��û����</h3>
            <p>{user_request}</p>
          </>
        )}
      </div>

      {/* ? ����ڰ� ���ε��� �̹��� �̸����� */}
      <div className="user-images">
        <h3>? ���ε��� �̹���</h3>
        <div className="image-preview-grid">
          {state.user_images && state.user_images.length > 0 ? (
            state.user_images.map((img, idx) => (
              <img key={idx} src={img} alt="uploaded" />
            ))
          ) : (
            <p>���ε�� �̹����� �����ϴ�.</p>
          )}
        </div>
      </div>

      {/* ? AI ��õ �̹��� ǥ�� */}
      <div className="recommended-images">
        <h3>? AI ��õ �ڵ�</h3>
        <div className="image-preview-grid">
          {recommended_images && recommended_images.length > 0 ? (
            recommended_images.map((img, idx) => (
              <img key={idx} src={img} alt="recommend" />
            ))
          ) : (
            <p>��õ �̹����� �ҷ����� ��...</p>
          )}
        </div>
      </div>

      {/* ? �ٽ� ��û ��ư */}
      <button className="back-btn" onClick={() => navigate("/")}>
        �ٽ� ��û�ϱ�
      </button>
    </div>
  );
}
