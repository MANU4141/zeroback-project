# Nginx Dockerfile for Frontend Serving (Simple Copy)
# 이 Dockerfile은 로컬에서 'npm run build'를 먼저 실행하여 'frontend/client/build' 디렉토리가 생성된 상태를 가정합니다.
FROM nginx:alpine

# 커스텀 Nginx 설정 파일 복사
COPY nginx/frontend.conf /etc/nginx/conf.d/default.conf

# 로컬에서 빌드된 React 정적 파일들을 Nginx 웹 루트로 복사
COPY frontend/client/build /usr/share/nginx/html

# curl 설치 (헬스체크용)
RUN apk add --no-cache curl

# 포트 노출
EXPOSE 80

# nginx 실행
CMD ["nginx", "-g", "daemon off;"]