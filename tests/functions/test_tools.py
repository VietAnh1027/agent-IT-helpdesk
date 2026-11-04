"""Unit tests cho Bộ 3 Diagnostic Tools (Read-Only) & Guardrails theo implementation plan 13."""

import unittest

from unittest.mock import MagicMock, patch

from src.agent.tools.diagnostic_adapter import (
    MockDiagnosticBackendAdapter,
    ProductionDiagnosticBackendAdapter,
)
from src.agent.tools.diagnostic_tools import (
    check_license_availability,
    check_printer_status,
    check_vpn_gateway_status,
)
from src.agent.tools.guardrails import (
    CATEGORY_TOOL_WHITELIST,
    validate_tool_call,
)
from src.agent.tools.interfaces import DiagnosticBackendAdapter


class TestDiagnosticTools(unittest.TestCase):
    """Kiểm tra chức năng của 3 diagnostic tools qua Adapter giả lập."""

    def setUp(self):
        self.adapter = MockDiagnosticBackendAdapter()
        # Seed dữ liệu mẫu cho adapter
        self.adapter.seed_vpn_gateway("default", status="up", last_incident_at=None)
        self.adapter.seed_vpn_gateway("gw_hcm", status="maintenance", last_incident_at="2026-11-01T10:00:00Z")
        self.adapter.seed_printer("PRINTER_01", status="online", queue_length=0, last_error=None)
        self.adapter.seed_printer("PRINTER_02", status="jammed", queue_length=4, last_error="Paper jam in tray 2")
        self.adapter.seed_license("Figma", total_seats=10, used_seats=8)
        self.adapter.seed_license("Photoshop", total_seats=5, used_seats=5)

    def test_check_vpn_gateway_healthy(self):
        result = check_vpn_gateway_status(gateway_id="default", adapter=self.adapter)
        self.assertEqual(result["gateway_id"], "default")
        self.assertEqual(result["status"], "up")
        self.assertTrue(result["is_healthy"])
        self.assertIsNone(result["last_incident_at"])

    def test_check_vpn_gateway_maintenance(self):
        result = check_vpn_gateway_status(gateway_id="gw_hcm", adapter=self.adapter)
        self.assertEqual(result["gateway_id"], "gw_hcm")
        self.assertEqual(result["status"], "maintenance")
        self.assertFalse(result["is_healthy"])
        self.assertEqual(result["last_incident_at"], "2026-11-01T10:00:00Z")

    def test_check_vpn_gateway_not_found_fallback(self):
        result = check_vpn_gateway_status(gateway_id="unknown_gw", adapter=self.adapter)
        self.assertEqual(result["gateway_id"], "unknown_gw")
        self.assertEqual(result["status"], "down")
        self.assertFalse(result["is_healthy"])

    def test_check_printer_operational(self):
        result = check_printer_status(printer_id="PRINTER_01", adapter=self.adapter)
        self.assertEqual(result["printer_id"], "PRINTER_01")
        self.assertEqual(result["status"], "online")
        self.assertTrue(result["is_operational"])
        self.assertEqual(result["queue_length"], 0)

    def test_check_printer_jammed_not_operational(self):
        result = check_printer_status(printer_id="PRINTER_02", adapter=self.adapter)
        self.assertEqual(result["printer_id"], "PRINTER_02")
        self.assertEqual(result["status"], "jammed")
        self.assertFalse(result["is_operational"])
        self.assertEqual(result["queue_length"], 4)
        self.assertEqual(result["last_error"], "Paper jam in tray 2")

    def test_check_printer_not_found(self):
        result = check_printer_status(printer_id="NON_EXISTENT", adapter=self.adapter)
        self.assertEqual(result["status"], "offline")
        self.assertFalse(result["is_operational"])

    def test_check_license_available(self):
        result = check_license_availability(software_name="Figma", adapter=self.adapter)
        self.assertEqual(result["software_name"], "Figma")
        self.assertEqual(result["total_seats"], 10)
        self.assertEqual(result["used_seats"], 8)
        self.assertEqual(result["available_seats"], 2)
        self.assertTrue(result["has_available_seat"])

    def test_check_license_out_of_seats(self):
        result = check_license_availability(software_name="Photoshop", adapter=self.adapter)
        self.assertEqual(result["software_name"], "Photoshop")
        self.assertEqual(result["total_seats"], 5)
        self.assertEqual(result["used_seats"], 5)
        self.assertEqual(result["available_seats"], 0)
        self.assertFalse(result["has_available_seat"])

    def test_check_license_not_found(self):
        result = check_license_availability(software_name="UnknownSoft", adapter=self.adapter)
        self.assertEqual(result["total_seats"], 0)
        self.assertEqual(result["available_seats"], 0)
        self.assertFalse(result["has_available_seat"])


class TestToolGuardrails(unittest.TestCase):
    """Kiểm tra lớp Guardrail phân vùng Category (Category-to-Tool Whitelist)."""

    def test_whitelist_allows_valid_category_tool(self):
        self.assertTrue(validate_tool_call(category="VPN", tool_name="check_vpn_gateway_status"))
        self.assertTrue(validate_tool_call(category="PRINTER", tool_name="check_printer_status"))
        self.assertTrue(validate_tool_call(category="SOFTWARE", tool_name="check_license_availability"))

    def test_whitelist_blocks_category_tool_mismatch(self):
        # Ticket VPN cố tình gọi tool máy in -> Bị chặn
        self.assertFalse(validate_tool_call(category="VPN", tool_name="check_printer_status"))
        # Ticket Printer cố tình gọi tool license -> Bị chặn
        self.assertFalse(validate_tool_call(category="PRINTER", tool_name="check_license_availability"))

    def test_whitelist_blocks_categories_without_tools(self):
        # PASSWORD và MAIL không có tool, xử lý qua KB -> Bị chặn nếu cố gọi tool
        self.assertFalse(validate_tool_call(category="PASSWORD", tool_name="check_vpn_gateway_status"))
        self.assertFalse(validate_tool_call(category="MAIL", tool_name="check_printer_status"))
        self.assertFalse(validate_tool_call(category="OUT_OF_SCOPE", tool_name="check_license_availability"))

    def test_category_case_insensitivity(self):
        self.assertTrue(validate_tool_call(category="vpn", tool_name="check_vpn_gateway_status"))
        self.assertTrue(validate_tool_call(category="Printer", tool_name="check_printer_status"))


class TestMockAdapterDatabaseIntegration(unittest.TestCase):
    """Kiểm tra MockDiagnosticBackendAdapter truy vấn các bảng PostgreSQL mock."""

    def test_query_vpn_gateway_from_postgres(self):
        adapter = MockDiagnosticBackendAdapter()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = ("default", "up", None)

        with patch("src.core.database.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            result = adapter.get_vpn_gateway_status("default")

            self.assertEqual(result["gateway_id"], "default")
            self.assertEqual(result["status"], "up")
            mock_cur.execute.assert_called_once()
            sql_called = mock_cur.execute.call_args[0][0]
            self.assertIn("mock_vpn_gateways", sql_called)

    def test_query_printer_from_postgres(self):
        adapter = MockDiagnosticBackendAdapter()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = ("PRINTER_01", "online", 2, None)

        with patch("src.core.database.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            result = adapter.get_printer_status("PRINTER_01")

            self.assertEqual(result["printer_id"], "PRINTER_01")
            self.assertEqual(result["status"], "online")
            self.assertEqual(result["queue_length"], 2)
            sql_called = mock_cur.execute.call_args[0][0]
            self.assertIn("mock_printers", sql_called)

    def test_query_license_from_postgres(self):
        adapter = MockDiagnosticBackendAdapter()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = ("Figma", 20, 15)

        with patch("src.core.database.get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            result = adapter.get_license_availability("Figma")

            self.assertEqual(result["software_name"], "Figma")
            self.assertEqual(result["total_seats"], 20)
            self.assertEqual(result["used_seats"], 15)
            sql_called = mock_cur.execute.call_args[0][0]
            self.assertIn("mock_software_licenses", sql_called)


class TestProductionDiagnosticBackendAdapter(unittest.TestCase):
    """Kiểm tra khung sườn của ProductionDiagnosticBackendAdapter."""

    def test_production_adapter_satisfies_protocol(self):
        prod_adapter = ProductionDiagnosticBackendAdapter()
        self.assertIsInstance(prod_adapter, DiagnosticBackendAdapter)

    def test_production_adapter_methods_pass(self):
        prod_adapter = ProductionDiagnosticBackendAdapter()
        self.assertIsNone(prod_adapter.get_vpn_gateway_status("default"))
        self.assertIsNone(prod_adapter.get_printer_status("P1"))
        self.assertIsNone(prod_adapter.get_license_availability("Figma"))


if __name__ == "__main__":
    unittest.main()

