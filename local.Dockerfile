# 로컬 시험용입니다. 제출하지 않으며 서버는 이 파일을 쓰지 않습니다(서버는 자체 Dockerfile로 같은 규칙을 적용).
# 서버와 같은 규칙: Python 3.12, requirements.txt는 name==version, PyPI의 미리 빌드된 휠(.whl)만, 실행 중 인터넷 없음.
#
#   docker build -f local.Dockerfile -t nova-agent .
#   docker run --rm --network=none --read-only --tmpfs /tmp nova-agent \
#       python -c "from run import prepare; prepare(); print('prepare OK')"
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /build/requirements.txt
# 1) 인터넷 사용: PyPI에서 휠만 받습니다. 소스만 배포되는 패키지는 여기서 실패하며, 서버에서도 실패합니다.
RUN python -m pip download --only-binary=:all: --index-url https://pypi.org/simple --no-cache-dir \
        --dest /build/wheels -r /build/requirements.txt
# 2) 인터넷 없이 받은 휠만 설치합니다.
RUN python -m pip install --no-index --only-binary=:all: --no-cache-dir --find-links=/build/wheels \
        --target=/opt/dependencies -r /build/requirements.txt
ENV PYTHONPATH=/opt/dependencies
COPY --chown=65532:65532 . /app/
USER 65532:65532
WORKDIR /app
CMD ["python", "run.py"]
