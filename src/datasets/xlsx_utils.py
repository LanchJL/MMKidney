import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _col_letters(ref: str) -> str:
    m = re.match(r"([A-Z]+)", ref or "")
    return m.group(1) if m else ""


def _cell_value(c: ET.Element, shared: List[str]) -> str:
    t = c.attrib.get("t")
    v = c.find("m:v", NS)
    if v is not None and v.text is not None:
        if t == "s":
            idx = int(v.text)
            return shared[idx] if 0 <= idx < len(shared) else v.text
        return v.text
    isel = c.find("m:is/m:t", NS)
    return isel.text if isel is not None and isel.text is not None else ""


def read_sheet_rows(xlsx_path: str, sheet_name: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Return (header_by_col_letter, list_of_rows_by_col_letter)."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                txt = "".join(t.text or "" for t in si.findall(".//m:t", NS))
                shared.append(txt)

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {x.attrib["Id"]: x.attrib["Target"] for x in rels.findall("pr:Relationship", NS)}

        sheet_path = None
        for s in wb.findall("m:sheets/m:sheet", NS):
            if s.attrib.get("name") == sheet_name:
                rid = s.attrib.get("{%s}id" % NS["r"])
                sheet_path = "xl/" + rel_map[rid]
                break

        if sheet_path is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        root = ET.fromstring(z.read(sheet_path))
        rows = root.findall("m:sheetData/m:row", NS)
        if not rows:
            return {}, []

        header_row = rows[0]
        header = {}
        for c in header_row.findall("m:c", NS):
            header[_col_letters(c.attrib.get("r", ""))] = _cell_value(c, shared)

        out = []
        for r in rows[1:]:
            row = {}
            for c in r.findall("m:c", NS):
                row[_col_letters(c.attrib.get("r", ""))] = _cell_value(c, shared)
            out.append(row)

        return header, out
