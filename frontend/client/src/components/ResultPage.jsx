import { useLocation } from "react-router-dom";

export default function ResultPage() {
  const { state } = useLocation();

  return (
    <div>
      <h2>추천 결과</h2>
      <p><strong>위치:</strong> {state?.location}</p>
      <p><strong>스타일:</strong> {state?.styles?.join(", ")}</p>
      <p><strong>요청사항:</strong> {state?.request}</p>
    </div>
  );
}
