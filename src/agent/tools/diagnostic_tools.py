"""Bộ 3 Diagnostic Tools (Read-Only) dạng hàm Python chuẩn cho IT Helpdesk Agent.

Sẵn sàng 100% để tích hợp vào LangGraph (@tool) hoặc gọi trực tiếp trong pipeline.

Đặc tả dựa trên:
- documents/07_TOOLS.md (Mục 1: Tool Schema)
- implementation_plan/13_TOOLS_AND_GUARDRAILS_IMPLEMENTATION_PLAN.md
"""

from src.agent.tools.diagnostic_adapter import MockDiagnosticBackendAdapter
from src.agent.tools.interfaces import DiagnosticBackendAdapter

# Singleton adapter mặc định
_DEFAULT_ADAPTER = MockDiagnosticBackendAdapter()


def check_vpn_gateway_status(
    gateway_id: str = "default",
    adapter: DiagnosticBackendAdapter | None = None,
) -> dict:
    """Tra cứu trạng thái gateway VPN hiện tại để phân biệt lỗi hệ thống vs lỗi phía người dùng.

    Args:
        gateway_id: Mã định danh gateway VPN (mặc định 'default').
        adapter: Backend adapter thực hiện kết nối tra cứu (mặc định dùng MockAdapter).

    Returns:
        dict chứa gateway_id, status (up/down/maintenance), last_incident_at và is_healthy (bool).
    """
    active_adapter = adapter or _DEFAULT_ADAPTER
    raw = active_adapter.get_vpn_gateway_status(gateway_id=gateway_id)
    status = raw.get("status", "down").lower()

    return {
        "gateway_id": raw.get("gateway_id", gateway_id),
        "status": status,
        "last_incident_at": raw.get("last_incident_at"),
        "is_healthy": status == "up",
    }


def check_printer_status(
    printer_id: str,
    adapter: DiagnosticBackendAdapter | None = None,
) -> dict:
    """Tra cứu trạng thái máy in văn phòng để phân biệt lỗi máy in vs lỗi cấu hình máy người dùng.

    Args:
        printer_id: Mã định danh của máy in (VD: PRINTER_01, PRINTER_FLOOR_3).
        adapter: Backend adapter thực hiện kết nối tra cứu.

    Returns:
        dict chứa printer_id, status (online/offline/jammed/low_toner), queue_length, last_error và is_operational (bool).
    """
    active_adapter = adapter or _DEFAULT_ADAPTER
    raw = active_adapter.get_printer_status(printer_id=printer_id)
    status = raw.get("status", "offline").lower()

    return {
        "printer_id": raw.get("printer_id", printer_id),
        "status": status,
        "queue_length": int(raw.get("queue_length", 0)),
        "last_error": raw.get("last_error"),
        "is_operational": status == "online",
    }


def check_license_availability(
    software_name: str,
    adapter: DiagnosticBackendAdapter | None = None,
) -> dict:
    """Tra cứu số lượng bản quyền (license) phần mềm còn trống trước khi hướng dẫn quy trình cấp phát.

    Args:
        software_name: Tên phần mềm cần kiểm tra bản quyền (VD: Figma, Photoshop, IntelliJ).
        adapter: Backend adapter thực hiện kết nối tra cứu.

    Returns:
        dict chứa software_name, total_seats, used_seats, available_seats và has_available_seat (bool).
    """
    active_adapter = adapter or _DEFAULT_ADAPTER
    raw = active_adapter.get_license_availability(software_name=software_name)

    total = int(raw.get("total_seats", 0))
    used = int(raw.get("used_seats", 0))
    available = max(0, total - used)

    return {
        "software_name": raw.get("software_name", software_name),
        "total_seats": total,
        "used_seats": used,
        "available_seats": available,
        "has_available_seat": available > 0,
    }

