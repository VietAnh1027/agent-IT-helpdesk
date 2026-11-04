"""Interfaces và Protocols cho hệ thống Diagnostic Tools (Read-Only).

Đặc tả dựa trên:
- documents/07_TOOLS.md (Mục 1 & 2)
- implementation_plan/13_TOOLS_AND_GUARDRAILS_IMPLEMENTATION_PLAN.md
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class DiagnosticBackendAdapter(Protocol):
    """Protocol chuẩn cho Adapter kết nối hạ tầng IT để chẩn đoán hệ thống."""

    def get_vpn_gateway_status(self, gateway_id: str = "default") -> dict:
        """Tra cứu trạng thái của VPN gateway."""
        ...

    def get_printer_status(self, printer_id: str) -> dict:
        """Tra cứu trạng thái của máy in."""
        ...

    def get_license_availability(self, software_name: str) -> dict:
        """Tra cứu số lượng bản quyền phần mềm còn trống."""
        ...

