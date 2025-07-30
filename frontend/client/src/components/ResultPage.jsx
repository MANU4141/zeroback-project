import "../css/OOTDForm.css";
import { useState } from "react";
import { FaMapMarkerAlt } from "react-icons/fa";
import { useNavigate } from "react-router-dom";
import KakaoMapModal from "./KakaoMapModal";
import axios from "axios";

export default function OOTDForm() {
  const navigate = useNavigate();
  const [location, setLocation] = useState("");
  const [coords, setCoords] = useState({ lat: null, lng: null });
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [styles, setStyles] = useState([]);
  const [request, setRequest] = useState("");
  const [images, setImages] = useState([]); // ? �̹��� ����

  const styleOptions = [
    "��Ʈ��", "�̴ϸ�", "��Ƽ��", "ĳ�־�", "����", "���ǽ�", "����ƾ", "�Ƹ�ī��"
  ];

  const toggleStyle = (style) => {
    setStyles((prev) =>
      prev.includes(style) ? prev.filter((s) => s !== style) : [...prev, style]
    );
  };

  // ? �̹��� ���ε� �ڵ鷯
  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    setImages((prev) => [...prev, ...files]);
  };

  // ? �̹��� ���� �ڵ鷯
  const handleRemoveImage = (index) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    if (!location || styles.length === 0) {
      alert("��ġ�� ��Ÿ���� �������ּ���.");
      return;
    }

    const requestData = {
      location,
      latitude: coords.lat,
      longitude: coords.lng,
      style_select: styles,
      user_request: request || ""
    };

    const formData = new FormData();
    formData.append("data", JSON.stringify(requestData));

    // ? �̹��� ���� �߰�
    images.forEach((img) => formData.append("images", img));

    try {
      const response = await axios.post("http://127.0.0.1:8000/api/recommend", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("? �鿣�� ����:", response.data);
      navigate("/result", { state: response.data });
    } catch (error) {
      console.error("? API ��û ����:", error);
      alert("��õ ��û �� ������ �߻��߽��ϴ�.");
    }
  };

  return (
    <div className={`ootd-container ${isModalOpen ? "blurred" : ""}`}>
      <h1 className="title">OOTD-AI</h1>
      <p className="subtitle">AI�� ��õ�ϴ� ������ �Ϻ��� �ڵ�</p>

      <div className="form-card">
        {/* ? ��ġ ���� */}
        <div className="form-section">
          <label className="form-label">��ġ ����</label>
          <div className="location-box" onClick={() => setIsModalOpen(true)}>
            <FaMapMarkerAlt className="icon" />
            <span className="location-label">
              {location || "��ġ�� �����ϼ���"}
            </span>
          </div>
        </div>

        {/* ? ��Ÿ�� ���� */}
        <div className="form-section">
          <label className="form-label">��Ÿ�� (�ߺ� ���� ����)</label>
          <div className="style-grid">
            {styleOptions.map((style, idx) => (
              <button
                key={idx}
                className={`style-btn ${styles.includes(style) ? "selected" : ""}`}
                onClick={() => toggleStyle(style)}
              >
                {style}
              </button>
            ))}
          </div>
        </div>

        {/* ? �߰� ��û���� */}
        <div className="form-section">
          <label className="form-label">�߰� ��û����</label>
          <textarea
            placeholder="���ϴ� ��Ÿ���̳� ��û������ �Է��ϼ���."
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
        </div>

        {/* ? �̹��� ���ε� */}
        <div className="form-section">
          <label className="form-label">�̹���/���� ���ε� (���� �� ����)</label>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleImageUpload}
            className="image-upload"
          />
          <div className="image-preview">
            {images.map((img, index) => (
              <div key={index} className="preview-box">
                <img src={URL.createObjectURL(img)} alt="preview" />
                <button className="remove-btn" onClick={() => handleRemoveImage(index)}>X</button>
              </div>
            ))}
          </div>
        </div>

        {/* ? ���� ��ư */}
        <button className="submit-btn" onClick={handleSubmit}>
          ��õ�ޱ�
        </button>
      </div>

      {/* ? ��� */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal">
            <h3 className="modal-title">��ġ ����</h3>
            <div className="modal-map">
              <KakaoMapModal
                onSelect={(data) => {
                  setLocation(data.address);
                  setCoords({ lat: data.lat, lng: data.lng });
                }}
                onClose={() => setIsModalOpen(false)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
