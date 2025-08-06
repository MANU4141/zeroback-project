# Nginx Dockerfile for Frontend Serving
FROM nginx:alpine

# 기본 설정 파일 제거
RUN rm /etc/nginx/conf.d/default.conf

# 커스텀 설정 파일만 복사
COPY nginx/frontend.conf /etc/nginx/conf.d/default.conf

# React 빌드 파일 복사
COPY frontend/client/build /usr/share/nginx/html

# 권한 설정
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# curl 설치 (헬스체크용)
RUN apk add --no-cache curl

# 포트 노출
EXPOSE 80

# nginx 실행
CMD ["nginx", "-g", "daemon off;"]