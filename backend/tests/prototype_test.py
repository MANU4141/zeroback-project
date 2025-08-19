#!/usr/bin/env python3
"""
Backend 통합 프로토타입 테스트 파일 (최적화 버전)

이 파일은 backend의 모든 주요 기능을 하나의 파일에서 테스트할 수 있습니다:
1. 이미지 업로드 및 AI 분석 (YOLO + ResNet)
2. 날씨 정보 조회
3. 최종 의상 추천

사용법:
python prototype_test.py --image "path/to/image.jpg" --lat 37.5665 --lng 126.9780 --request "캐주얼한 옷차림"
"""

import os
import sys
import json
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# 현재 파일의 디렉토리와 상위 디렉토리를 sys.path에 추가 (중복 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BackendPrototypeTester:
    """Backend 기능을 통합 테스트하는 클래스 (최적화 버전)"""

    def __init__(self):
        self.models_ready = False
        self.weather_ready = False
        self.db_ready = False
        self._initialize_components()

    def _initialize_components(self):
        """컴포넌트들을 초기화합니다."""
        try:
            from config import Config
            from app.services import model_manager
            from app.weather import KoreaWeatherAPI
            from app.utils import build_db_images

            # 설정 검증
            config_errors = Config.validate_config()
            if config_errors:
                logger.warning(f"설정 검증 경고: {len(config_errors)}개 문제 발견")
                for error in config_errors[:3]:  # 처음 3개만 표시
                    logger.warning(f"  - {error}")

            # AI 모델 초기화
            logger.info("AI 모델 초기화 중...")
            self.model_manager = model_manager

            # Flask 앱 없이 모델 초기화를 위한 mock 앱 생성
            try:
                from flask import Flask

                test_app = Flask(__name__)
                test_app.config.from_object(Config)

                with test_app.app_context():
                    # 모델 초기화 실행
                    success = self.model_manager.initialize_models(test_app)
                    if success:
                        logger.info("✅ AI 모델 초기화 성공")
                        self.models_ready = True
                    else:
                        logger.warning("⚠️ AI 모델 초기화 실패")
                        self.models_ready = False

            except Exception as e:
                logger.exception("AI 모델 초기화 중 오류")
                self.models_ready = False

            # 날씨 API 초기화
            logger.info("날씨 API 초기화 중...")
            self.weather_api = KoreaWeatherAPI()
            self.weather_ready = True

            # 이미지 DB 초기화
            logger.info("이미지 데이터베이스 로드 중...")
            self.db_images = build_db_images(Config.LABELS_DIR, Config.IMAGE_DIR)
            self.db_ready = True

            logger.info(f"✅ 초기화 완료: DB 이미지 {len(self.db_images)}개")

        except Exception as e:
            logger.exception("초기화 실패")
            # 기본값 설정으로 부분적으로라도 테스트 진행 가능하도록
            self.db_images = []
            self.db_ready = False

    def run_comprehensive_test(self, args) -> Dict[str, Any]:
        """종합적인 테스트를 실행합니다."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_config": vars(args),
            "component_status": self._check_component_status(),
            "test_results": {},
        }

        try:
            # 1. 이미지 분석 테스트 (경로 또는 파일명 사용)
            if args.image:
                image_path = None

                # 전체 경로인 경우
                if os.path.exists(args.image):
                    image_path = args.image
                    logger.info(
                        f"🖼️ 전체 경로 이미지 분석 테스트 시작: {os.path.basename(args.image)}"
                    )
                else:
                    # 파일명만 제공된 경우 DATA/images에서 찾기
                    from config import Config

                    potential_path = os.path.join(Config.IMAGE_DIR, args.image)

                    if os.path.exists(potential_path):
                        image_path = potential_path
                        logger.info(
                            f"🖼️ 파일명으로 이미지 분석 테스트 시작: {args.image}"
                        )
                    else:
                        logger.error(f"❌ 이미지를 찾을 수 없습니다: {args.image}")
                        logger.info(f"   경로 확인: {potential_path}")
                        test_results["test_results"]["image_analysis"] = {
                            "skipped": True,
                            "reason": f"Image not found: {args.image}",
                        }
                        image_path = None

                if image_path:
                    test_results["test_results"]["image_analysis"] = (
                        self._test_image_analysis(image_path)
                    )
                    # args.image를 실제 경로로 업데이트 (저장용)
                    args.image = image_path
            else:
                logger.info("⏭️ 이미지 분석 테스트 생략 (이미지 파일 지정되지 않음)")
                test_results["test_results"]["image_analysis"] = {"skipped": True}

            # 2. 날씨 정보 테스트
            logger.info("🌤️ 날씨 정보 테스트 시작")
            test_results["test_results"]["weather"] = self._test_weather_api(
                args.lat, args.lng
            )

            # 3. 추천 생성 테스트
            logger.info("💡 추천 생성 테스트 시작")
            test_results["test_results"]["recommendation"] = (
                self._test_recommendation_generation(args, test_results["test_results"])
            )

            # 4. 성능 요약
            test_results["performance_summary"] = self._get_performance_summary()

            # 5. 이미지 저장 (원본 + 추천 상위 5개)
            if args.save_images:
                logger.info("💾 테스트 이미지 저장 중...")
                test_results["saved_images"] = self._save_test_images(
                    args, test_results
                )

            logger.info("✅ 모든 테스트 완료")

        except Exception as e:
            logger.exception("테스트 실행 중 오류")
            test_results["test_error"] = str(e)

        return test_results

    def _check_component_status(self) -> Dict[str, Any]:
        """컴포넌트 상태를 확인합니다."""
        return {
            "models_ready": self.models_ready,
            "weather_ready": self.weather_ready,
            "db_ready": self.db_ready,
            "db_image_count": len(self.db_images) if hasattr(self, "db_images") else 0,
        }

    def _test_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """이미지 분석을 테스트합니다."""
        try:
            import mimetypes
            from app.services import analyze_single_image
            from werkzeug.datastructures import FileStorage

            # MIME 타입 안전 추정
            mime_type, _ = mimetypes.guess_type(image_path)
            content_type = mime_type or "application/octet-stream"

            # 파일을 FileStorage 객체로 래핑
            with open(image_path, "rb") as f:
                file_storage = FileStorage(
                    stream=f,
                    filename=os.path.basename(image_path),
                    content_type=content_type,
                )

                analysis_result, debug_info = analyze_single_image(file_storage, 0)

            return {
                "success": True,
                "analysis_result": analysis_result,
                "debug_info": debug_info,
                "image_path": image_path,
            }

        except Exception as e:
            logger.exception("이미지 분석 테스트 실패")
            return {"success": False, "error": str(e), "image_path": image_path}

    def _test_weather_api(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """날씨 API를 테스트합니다."""
        try:
            from app.services import get_weather_info

            weather_info, is_fallback = get_weather_info(latitude, longitude)

            return {
                "success": True,
                "weather_info": weather_info,
                "is_fallback": is_fallback,
                "coordinates": {"latitude": latitude, "longitude": longitude},
            }

        except Exception as e:
            logger.exception("날씨 API 테스트 실패")
            return {
                "success": False,
                "error": str(e),
                "coordinates": {"latitude": latitude, "longitude": longitude},
            }

    def _test_recommendation_generation(
        self, args, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """추천 생성을 테스트합니다."""
        try:
            from app.services import get_final_recommendation
            from app.utils import convert_image_paths_to_filenames

            # 요청 데이터 구성
            user_preferences = {
                "location": "테스트 위치",
                "latitude": args.lat,
                "longitude": args.lng,
                "style_select": args.style.split(",") if args.style else ["캐주얼"],
                "user_request": args.request or "편안한 옷차림을 추천해주세요",
            }

            # 날씨 정보 가져오기
            weather_info = test_results.get("weather", {}).get("weather_info", {})

            # AI 분석 결과 가져오기 (선택적) 및 형태 변환
            ai_analysis = test_results.get("image_analysis", {}).get("analysis_result")

            # AI 분석 결과 변환 - 추천 시스템이 기대하는 형태로 변환
            if ai_analysis and isinstance(ai_analysis, list) and len(ai_analysis) > 0:
                # 모든 분석 결과를 합쳐서 처리
                converted_analysis = {}

                for result in ai_analysis:
                    attributes = result.get("attributes", {})
                    category = result.get("category", "")

                    # 카테고리 추가
                    if category:
                        if "category" not in converted_analysis:
                            converted_analysis["category"] = []
                        converted_analysis["category"].append(category)

                    # attributes가 있는 경우 각 속성을 문자열 리스트로 변환
                    if attributes:
                        for attr_name, attr_values in attributes.items():
                            if attr_name not in converted_analysis:
                                converted_analysis[attr_name] = []

                            # 확률이 포함된 딕셔너리 형태인 경우 class_name만 추출
                            if isinstance(attr_values, list):
                                for item in attr_values[:3]:  # 상위 3개만 사용
                                    if isinstance(item, dict) and "class_name" in item:
                                        converted_analysis[attr_name].append(
                                            item["class_name"]
                                        )
                                    elif isinstance(item, str):
                                        converted_analysis[attr_name].append(item)
                            elif isinstance(attr_values, str):
                                converted_analysis[attr_name].append(attr_values)

                ai_analysis = converted_analysis if converted_analysis else None
            else:
                ai_analysis = None

            # DB 이미지 준비
            converted_db_images = convert_image_paths_to_filenames(self.db_images)

            # 추천 생성 (더 많은 이미지 요청해서 원본 제외 후 5개 확보)
            recommendation_result = get_final_recommendation(
                weather=weather_info,
                user_prompt=user_preferences.get("user_request", "편안한 옷차림"),
                style_preferences=user_preferences.get("style_select", ["캐주얼"]),
                ai_attributes=ai_analysis,
                db_images=converted_db_images[:500],  # 더 많은 이미지에서 선택
                per_page=8,  # 8개 이미지 반환 (원본 제외하고 5개 확보)
                page=1,
            )

            return {
                "success": True,
                "recommendation_result": recommendation_result,
                "user_preferences": user_preferences,
                "weather_used": weather_info,
                "ai_analysis_used": bool(ai_analysis),
            }

        except Exception as e:
            logger.exception("추천 생성 테스트 실패")
            return {"success": False, "error": str(e)}

    def _get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약을 가져옵니다."""
        try:
            from app.monitoring import performance_monitor

            return {
                "performance_metrics": performance_monitor.get_summary(),
                "current_session": performance_monitor.current_session,
            }
        except Exception:
            return {"error": "성능 모니터링 정보를 가져올 수 없습니다."}

    def _save_test_images(self, args, results: Dict[str, Any]) -> Dict[str, Any]:
        """테스트에 사용된 원본 이미지와 추천된 상위 5개 이미지를 저장합니다."""
        try:
            # 저장 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"test_results_{timestamp}")
            save_dir.mkdir(exist_ok=True)

            saved_images = {
                "save_directory": str(save_dir),
                "original_image": None,
                "recommended_images": [],
                "total_saved": 0,
            }

            # 1. 원본 이미지 저장 (있는 경우)
            if args.image and os.path.exists(args.image):
                try:
                    original_image_path = Path(args.image)
                    saved_original = save_dir / f"original_{original_image_path.name}"
                    shutil.copy2(args.image, saved_original)
                    saved_images["original_image"] = str(saved_original)
                    saved_images["total_saved"] += 1
                    logger.info(f"✅ 원본 이미지 저장: {saved_original}")
                except Exception as e:
                    logger.exception(f"원본 이미지 저장 실패: {e}")

            # 2. 추천된 상위 5개 이미지 저장 (원본 제외)
            rec_result = results.get("test_results", {}).get("recommendation", {})
            if rec_result.get("success"):
                recommendation_data = rec_result.get("recommendation_result", {})
                recommended_images = recommendation_data.get("images", [])

                # 원본 이미지 파일명 추출 (제외용)
                original_filename = None
                if args.image:
                    original_filename = os.path.basename(args.image)

                # 원본 이미지를 제외한 추천 이미지들 필터링
                filtered_recommendations = []
                for item in recommended_images:
                    image_filename = (
                        item.get("image_filename")
                        or item.get("image_name")
                        or item.get("img_path")
                    )

                    # 원본 이미지가 아닌 경우만 추가
                    if image_filename != original_filename:
                        filtered_recommendations.append(item)

                    # 5개가 될 때까지 수집
                    if len(filtered_recommendations) >= 5:
                        break

                # 상위 5개 추천 이미지 저장
                for idx, item in enumerate(filtered_recommendations[:5], 1):
                    try:
                        # 추천 아이템에서 이미지 경로 추출 (다양한 키 시도)
                        image_filename = (
                            item.get("image_filename")
                            or item.get("image_name")
                            or item.get("img_path")
                        )
                        similarity_score = item.get("similarity_score", 0.0)

                        if image_filename:
                            # DB에서 실제 이미지 경로 찾기
                            from config import Config

                            source_image_path = Path(Config.IMAGE_DIR) / image_filename

                            if source_image_path.exists():
                                # 저장할 파일명 생성 (순위와 유사도 점수 포함)
                                file_ext = source_image_path.suffix
                                saved_filename = f"recommended_{idx:02d}_score_{similarity_score:.3f}_{image_filename}"
                                saved_path = save_dir / saved_filename

                                # 이미지 복사
                                shutil.copy2(source_image_path, saved_path)

                                saved_images["recommended_images"].append(
                                    {
                                        "rank": idx,
                                        "filename": image_filename,
                                        "saved_path": str(saved_path),
                                        "similarity_score": similarity_score,
                                        "attributes": item.get("attributes", {}),
                                    }
                                )
                                saved_images["total_saved"] += 1
                                logger.info(f"✅ 추천 이미지 {idx} 저장: {saved_path}")
                            else:
                                logger.warning(
                                    f"추천 이미지 파일을 찾을 수 없음: {source_image_path}"
                                )

                    except Exception as e:
                        logger.exception(f"추천 이미지 {idx} 저장 실패: {e}")

            # 3. 테스트 결과 JSON도 함께 저장
            try:
                results_json_path = save_dir / "test_results.json"
                with open(results_json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 테스트 결과 JSON 저장: {results_json_path}")
            except Exception as e:
                logger.exception(f"테스트 결과 JSON 저장 실패: {e}")

            return saved_images

        except Exception as e:
            logger.exception("이미지 저장 중 오류 발생")
            return {"error": str(e), "total_saved": 0}

    def print_test_summary(self, results: Dict[str, Any]):
        """테스트 결과 요약을 출력합니다."""
        print("\n" + "=" * 60)
        print("📋 BACKEND PROTOTYPE TEST SUMMARY")
        print("=" * 60)

        # 컴포넌트 상태
        status = results["component_status"]
        print(f"🔧 Component Status:")
        print(f"   Models Ready: {'✅' if status['models_ready'] else '❌'}")
        print(f"   Weather Ready: {'✅' if status['weather_ready'] else '❌'}")
        print(
            f"   DB Ready: {'✅' if status['db_ready'] else '❌'} ({status['db_image_count']} images)"
        )

        # 테스트 결과
        test_res = results["test_results"]
        print(f"\n📊 Test Results:")

        # 이미지 분석
        img_test = test_res.get("image_analysis", {})
        if img_test.get("skipped"):
            print(f"   Image Analysis: ⏭️ Skipped")
        else:
            print(f"   Image Analysis: {'✅' if img_test.get('success') else '❌'}")
            if img_test.get("success") and "analysis_result" in img_test:
                analysis = img_test["analysis_result"]
                if analysis:
                    print(f"      → Detected {len(analysis)} objects")

        # 날씨 테스트
        weather_test = test_res.get("weather", {})
        print(f"   Weather API: {'✅' if weather_test.get('success') else '❌'}")
        if weather_test.get("success"):
            weather_info = weather_test.get("weather_info", {})
            is_fallback = weather_test.get("is_fallback", False)
            print(
                f"      → {weather_info.get('temperature', 'N/A')}°C, {weather_info.get('condition', 'N/A')}"
            )
            print(f"      → Source: {'Fallback' if is_fallback else 'API'}")

        # 추천 테스트
        rec_test = test_res.get("recommendation", {})
        print(f"   Recommendation: {'✅' if rec_test.get('success') else '❌'}")
        if rec_test.get("success"):
            rec_result = rec_test.get("recommendation_result", {})
            rec_text = rec_result.get("recommendation_text", "")
            if rec_text:
                print(f"      → {rec_text[:100]}...")

        # 성능 요약
        perf = results.get("performance_summary", {})
        if "performance_metrics" in perf:
            metrics = perf["performance_metrics"]
            print(f"\n⚡ Performance Summary:")
            for operation, data in list(metrics.items())[:5]:  # 상위 5개만 표시
                if isinstance(data, dict) and "avg_ms" in data:
                    print(
                        f"   {operation}: {data['avg_ms']:.2f}ms avg ({data['count']} calls)"
                    )

        # 저장된 이미지 정보
        saved_images = results.get("saved_images", {})
        if saved_images and saved_images.get("total_saved", 0) > 0:
            print(f"\n💾 Saved Images:")
            print(f"   Directory: {saved_images.get('save_directory', 'N/A')}")
            print(f"   Total Saved: {saved_images.get('total_saved', 0)}")

            if saved_images.get("original_image"):
                print(f"   Original: ✅")

            recommended_count = len(saved_images.get("recommended_images", []))
            if recommended_count > 0:
                print(f"   Recommended: {recommended_count} images")

                # 상위 3개 추천 이미지의 점수 표시
                for img_info in saved_images.get("recommended_images", [])[:3]:
                    rank = img_info.get("rank", 0)
                    score = img_info.get("similarity_score", 0.0)
                    print(f"      #{rank}: {score:.3f} similarity")

        print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Backend Prototype Tester")
    parser.add_argument(
        "--image",
        type=str,
        help="테스트할 이미지 파일 경로 또는 파일명 (DATA/images 폴더에서 자동 검색)",
    )
    parser.add_argument(
        "--lat", type=float, default=37.5665, help="위도 (기본값: 서울)"
    )
    parser.add_argument(
        "--lng", type=float, default=126.9780, help="경도 (기본값: 서울)"
    )
    parser.add_argument(
        "--request", type=str, default="편안한 옷차림", help="사용자 요청"
    )
    parser.add_argument(
        "--style", type=str, default="캐주얼", help="스타일 (쉼표로 구분)"
    )
    parser.add_argument("--output", type=str, help="결과를 저장할 JSON 파일 경로")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="원본 이미지와 추천 상위 5개 이미지를 저장",
    )
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 테스트 실행
    tester = BackendPrototypeTester()
    results = tester.run_comprehensive_test(args)

    # 결과 출력
    tester.print_test_summary(results)

    # 결과 저장 (선택적)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 결과가 {args.output}에 저장되었습니다.")
        except Exception as e:
            logger.exception("결과 저장 실패")
            print(f"❌ 결과 저장 실패: {e}")


if __name__ == "__main__":
    main()
