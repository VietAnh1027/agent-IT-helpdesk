"""Triển khai DiagnosticBackendAdapter cho môi trường giả lập (Mock) và thực tế (Production).

Đặc tả dựa trên:
- documents/07_TOOLS.md (Mục 2: Chiến lược Mock Backend)
- implementation_plan/13_TOOLS_AND_GUARDRAILS_IMPLEMENTATION_PLAN.md
"""

import logging
import sys

from src.agent.tools.interfaces import DiagnosticBackendAdapter

logger = logging.getLogger(__name__)


class MockDiagnosticBackendAdapter(DiagnosticBackendAdapter):
    """Adapter giả lập đọc dữ liệu hạ tầng IT từ PostgreSQL mock tables hoặc bộ nhớ (in-memory).

    Kế thừa tường minh từ DiagnosticBackendAdapter Protocol.
    Hỗ trợ 2 cơ chế:
    1. Đọc trực tiếp từ các bảng PostgreSQL mock: mock_vpn_gateways, mock_printers, mock_software_licenses.
    2. Fallback sang bộ nhớ in-memory khi chạy unit test offline hoặc khi được seed dữ liệu mẫu.
    """

    def __init__(self, use_db: bool = False) -> None:
        self.use_db = use_db
        self._vpn_gateways: dict[str, dict] = {}
        self._printers: dict[str, dict] = {}
        self._software_licenses: dict[str, dict] = {}

    # --- Seed Helpers phục vụ Unit Tests và Demo kịch bản ---

    def seed_vpn_gateway(self, gateway_id: str, status: str = "up", last_incident_at: str | None = None) -> None:
        """Seed trạng thái gateway VPN (up, down, maintenance)."""
        self._vpn_gateways[gateway_id] = {
            "gateway_id": gateway_id,
            "status": status,
            "last_incident_at": last_incident_at,
        }

    def seed_printer(
        self,
        printer_id: str,
        status: str = "online",
        queue_length: int = 0,
        last_error: str | None = None,
    ) -> None:
        """Seed trạng thái máy in (online, offline, jammed, low_toner)."""
        self._printers[printer_id] = {
            "printer_id": printer_id,
            "status": status,
            "queue_length": queue_length,
            "last_error": last_error,
        }

    def seed_license(self, software_name: str, total_seats: int, used_seats: int = 0) -> None:
        """Seed số lượng license phần mềm."""
        self._software_licenses[software_name.casefold()] = {
            "software_name": software_name,
            "total_seats": total_seats,
            "used_seats": used_seats,
        }

    # --- Triển khai các phương thức của DiagnosticBackendAdapter ---

    def get_vpn_gateway_status(self, gateway_id: str = "default") -> dict:
        """Tra cứu trạng thái của VPN gateway từ PostgreSQL mock_vpn_gateways (hoặc in-memory)."""
        # 1. Kiểm tra cache/seed in-memory trước
        if gateway_id in self._vpn_gateways:
            return dict(self._vpn_gateways[gateway_id])

        # 2. Truy vấn từ bảng PostgreSQL mock_vpn_gateways
        try:
            from src.core.database import get_connection

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, status, last_incident_at FROM mock_vpn_gateways WHERE id = %s",
                        (gateway_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        last_inc = (
                            row[2].isoformat()
                            if hasattr(row[2], "isoformat")
                            else str(row[2])
                            if row[2]
                            else None
                        )
                        return {
                            "gateway_id": row[0],
                            "status": row[1],
                            "last_incident_at": last_inc,
                        }
        except Exception as exc:
            print(
                f"[MockDiagnosticAdapter] Lỗi truy vấn mock_vpn_gateways (gateway_id='{gateway_id}'): {exc}",
                file=sys.stderr,
            )
            logger.warning("Lỗi truy vấn mock_vpn_gateways (gateway_id='%s'): %s", gateway_id, exc)

        # 3. Fallback mặc định nếu không tìm thấy dữ liệu
        return {
            "gateway_id": gateway_id,
            "status": "down",
            "last_incident_at": None,
        }

    def get_printer_status(self, printer_id: str) -> dict:
        """Tra cứu trạng thái của máy in từ PostgreSQL mock_printers (hoặc in-memory)."""
        if printer_id in self._printers:
            return dict(self._printers[printer_id])

        try:
            from src.core.database import get_connection

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, status, queue_length, last_error FROM mock_printers WHERE id = %s",
                        (printer_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "printer_id": row[0],
                            "status": row[1],
                            "queue_length": int(row[2]),
                            "last_error": row[3],
                        }
        except Exception as exc:
            print(
                f"[MockDiagnosticAdapter] Lỗi truy vấn mock_printers (printer_id='{printer_id}'): {exc}",
                file=sys.stderr,
            )
            logger.warning("Lỗi truy vấn mock_printers (printer_id='%s'): %s", printer_id, exc)

        return {
            "printer_id": printer_id,
            "status": "offline",
            "queue_length": 0,
            "last_error": "Printer not found or unreachable",
        }

    def get_license_availability(self, software_name: str) -> dict:
        """Tra cứu số lượng bản quyền phần mềm từ PostgreSQL mock_software_licenses (hoặc in-memory)."""
        key = software_name.casefold()
        if key in self._software_licenses:
            return dict(self._software_licenses[key])

        try:
            from src.core.database import get_connection

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT software_name, total_seats, used_seats FROM mock_software_licenses WHERE software_name ILIKE %s",
                        (software_name,),
                    )
                    row = cur.fetchone()
                    if row:
                        return {
                            "software_name": row[0],
                            "total_seats": int(row[1]),
                            "used_seats": int(row[2]),
                        }
        except Exception as exc:
            print(
                f"[MockDiagnosticAdapter] Lỗi truy vấn mock_software_licenses (software_name='{software_name}'): {exc}",
                file=sys.stderr,
            )
            logger.warning(
                "Lỗi truy vấn mock_software_licenses (software_name='%s'): %s", software_name, exc
            )

        return {
            "software_name": software_name,
            "total_seats": 0,
            "used_seats": 0,
        }


class ProductionDiagnosticBackendAdapter(DiagnosticBackendAdapter):
    """Khung Adapter dành cho môi trường Production thực tế.

    Kế thừa tường minh từ DiagnosticBackendAdapter Protocol.

    LƯU Ý DÀNH CHO DEVELOPER:
    Đây là khung sườn (skeleton/stub) để bạn hoặc đội ngũ triển khai hoàn thiện khi đưa
    hệ thống vào môi trường doanh nghiệp thực thụ. Thay vì query các bảng mock, các hàm ở đây sẽ:
    - get_vpn_gateway_status: Gọi API của Firewall / VPN Gateway (Cisco AnyConnect, Fortinet, Palo Alto).
    - get_printer_status: Gọi Windows Print Server API (WMI / PowerShell) hoặc giao thức SNMP máy in.
    - get_license_availability: Gọi Microsoft 365 Admin Graph API, Okta API hoặc Jamf Pro.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def get_vpn_gateway_status(self, gateway_id: str = "default") -> dict:
        """TODO: Kết nối API Gateway VPN thật tại đây."""
        pass

    def get_printer_status(self, printer_id: str) -> dict:
        """TODO: Kết nối Print Server / SNMP thật tại đây."""
        pass

    def get_license_availability(self, software_name: str) -> dict:
        """TODO: Kết nối Identity & License Management API thật tại đây."""
        pass
