# backend/app/monitoring.py
"""
성능 모니터링 (프로토타입 단순화 버전)
- 간단 타이밍 측정 컨텍스트/데코레이터
- 기본 요약 통계
"""

import time
import logging
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """아주 간단한 타이밍 수집기"""

    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self.metrics = defaultdict(lambda: deque(maxlen=self.history_size))

    def record_timing(self, operation: str, duration_ms: float):
        self.metrics[operation].append(duration_ms)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Perf %s: %.2fms", operation, duration_ms)

    def get_summary(self, operation: str = None):
        if operation:
            return self._summary_for(operation)
        return {op: self._summary_for(op) for op in self.metrics.keys()}

    def _summary_for(self, operation: str):
        timings = self.metrics.get(operation, [])
        if not timings:
            return None
        n = len(timings)
        sorted_vals = sorted(timings)
        return {
            "operation": operation,
            "count": n,
            "avg_ms": sum(timings) / n,
            "min_ms": sorted_vals[0],
            "max_ms": sorted_vals[-1],
            "latest_ms": timings[-1],
        }

    def reset(self):
        self.metrics.clear()


# 전역 인스턴스
performance_monitor = PerformanceMonitor()


@contextmanager
def performance_timer(operation_name: str):
    """간단 타이밍 측정 컨텍스트"""
    t0 = time.perf_counter()
    try:
        yield {"operation": operation_name, "start_time": t0}
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000
        performance_monitor.record_timing(operation_name, duration_ms)


def monitor_performance(operation_name: str):
    """간단 타이밍 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with performance_timer(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class TimingContext:
    """멀티스텝 타이밍 (간단판)"""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.timings = {}

    def record_step(self, step_name: str):
        now = time.perf_counter()
        dur_ms = (now - self._last) * 1000
        self.timings[step_name] = dur_ms
        performance_monitor.record_timing(step_name, dur_ms)
        self._last = now
        return dur_ms

    def finalize(self, total_operation_name: str = "total_request"):
        total_ms = (time.perf_counter() - self._t0) * 1000
        self.timings[total_operation_name] = total_ms
        performance_monitor.record_timing(total_operation_name, total_ms)
        return self.timings


# 하위 호환 헬퍼들
def create_timing_context():
    return TimingContext()


def record_timing_step(context: TimingContext, step_name: str):
    return context.record_step(step_name)


def finalize_timing_context(
    context: TimingContext, total_operation_name: str = "total_request"
):
    return context.finalize(total_operation_name)


def check_performance_slo(timings=None):
    """
    SLO 체크는 프로토타입에서는 제거.
    호환성 유지용으로 항상 빈 리스트 반환.
    """
    return []
