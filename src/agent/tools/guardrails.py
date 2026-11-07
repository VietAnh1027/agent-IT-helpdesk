"""Tool Guardrails: Kiểm soát quyền hạn và phân vùng công cụ theo Category.

Đặc tả dựa trên:
- documents/07_TOOLS.md (Mục 5: Cập nhật Guardrails)
- documents/09_GUARDRAILS.md (Mục 4: Output Guardrails)
- implementation_plan/13_TOOLS_AND_GUARDRAILS_IMPLEMENTATION_PLAN.md
"""

# Whitelist ánh xạ Category -> Tập hợp các Tool được phép gọi
CATEGORY_TOOL_WHITELIST: dict[str, set[str]] = {
    "VPN": {"check_vpn_gateway_status"},
    "PRINTER": {"check_printer_status"},
    "SOFTWARE": {"check_license_availability"},
    "PASSWORD": set(),
    "MAIL": set(),
    "OUT_OF_SCOPE": set(),
}


def validate_tool_call(category: str, tool_name: str) -> bool:
    """Kiểm tra xem ticket thuộc category này có được phép kích hoạt tool_name hay không.

    Quy tắc cứng: Ngăn chặn triệt để kịch bản LLM bị lừa gọi chéo tool giữa các category.
    """
    cat_upper = category.strip().upper()
    allowed_tools = CATEGORY_TOOL_WHITELIST.get(cat_upper, set())
    return tool_name.strip() in allowed_tools

