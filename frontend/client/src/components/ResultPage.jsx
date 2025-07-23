import { useLocation } from "react-router-dom";

export default function ResultPage() {
  const { state } = useLocation();

  return (
    <div>
      <h2>��õ ���</h2>
      <p><strong>��ġ:</strong> {state?.location}</p>
      <p><strong>��Ÿ��:</strong> {state?.styles?.join(", ")}</p>
      <p><strong>��û����:</strong> {state?.request}</p>
    </div>
  );
}
