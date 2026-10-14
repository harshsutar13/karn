"""
tools/xlsx_writer.py

Generate .xlsx spreadsheets from structured data.

Supported inputs:
- Simple table: {"headers": [...], "rows": [[...], ...]}
- Multiple sheets: [{"sheet_name": "...", "headers": [...], "rows": [...]}, ...]
- Parameter list: [{"parameter": "rho", "value": 994, "unit": "kg/m³"}, ...]
"""

from __future__ import annotations

import os
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


_HEADER_FILL = PatternFill("solid", fgColor="305496")
_HEADER_FONT = Font(color="FFFFFF", bold=True)
_TITLE_FONT = Font(size=14, bold=True)


def _autosize(ws, min_width: int = 10, max_width: int = 60) -> None:
    """Auto-fit column widths based on content."""
    for col in ws.columns:
        max_len = 0
        for cell in col:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        width = max(min_width, min(max_len + 2, max_width))
        ws.column_dimensions[get_column_letter(col[0].column)].width = width


def write_xlsx(
    title: str,
    output_path: str,
    sheets: list[dict],
    footer_note: str | None = None,
) -> str:
    """
    Write an Excel workbook.

    Parameters
    ----------
    title : str
        Workbook title, added as a bold line above the first sheet's data.
    output_path : str
        Where to save the .xlsx.
    sheets : list[dict]
        Each dict has:
          - sheet_name (optional, default "Sheet1")
          - headers: list of str (optional)
          - rows: list of list (each row's cell values)
    footer_note : str | None
        Optional note added at the bottom of the first sheet.

    Returns
    -------
    str : absolute path written
    """
    if not sheets:
        sheets = [{"sheet_name": "Sheet1", "headers": [], "rows": []}]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb = Workbook()
    # Remove default sheet; we'll create them
    wb.remove(wb.active)

    for idx, spec in enumerate(sheets):
        sheet_name = (spec.get("sheet_name") or f"Sheet{idx+1}")[:31]  # Excel max 31 chars
        ws = wb.create_sheet(title=sheet_name)

        row_cursor = 1

        # Title row on first sheet
        if idx == 0:
            ws.cell(row=row_cursor, column=1, value=title).font = _TITLE_FONT
            row_cursor += 1
            ws.cell(
                row=row_cursor, column=1,
                value=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            row_cursor += 2  # blank line

        headers = spec.get("headers") or []
        if headers:
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=row_cursor, column=c, value=str(h))
                cell.fill = _HEADER_FILL
                cell.font = _HEADER_FONT
                cell.alignment = Alignment(horizontal="center")
            row_cursor += 1

        rows = spec.get("rows") or []
        for r in rows:
            if isinstance(r, dict):
                # Convert {k: v} row into ordered list using headers if given
                if headers:
                    values = [r.get(h, "") for h in headers]
                else:
                    values = list(r.values())
            elif isinstance(r, (list, tuple)):
                values = list(r)
            else:
                values = [r]

            for c, v in enumerate(values, start=1):
                ws.cell(row=row_cursor, column=c, value=v)
            row_cursor += 1

        _autosize(ws)

        # Footer note on the first sheet only
        if idx == 0 and footer_note:
            ws.cell(row=row_cursor + 1, column=1, value=footer_note).font = Font(
                italic=True, size=9, color="888888"
            )

    wb.save(output_path)
    return os.path.abspath(output_path)