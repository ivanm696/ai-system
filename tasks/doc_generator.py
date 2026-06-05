"""
Task: generate_pdf
Генерация PDF-документов (ISO, отчёты, договоры).
Поддерживает: заголовок, секции, штамп (watermark), мокрая печать.
"""
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from core.task_registry import TaskRegistry, BaseTask
from core.config import DIRS


@TaskRegistry.register("generate_pdf")
class DocGeneratorTask(BaseTask):
    """Генерация PDF-документов с разделами и штампом."""

    def run(self, input_data: dict) -> dict:
        filename = input_data.get("filename", "document")
        title = input_data.get("title", "Document")
        sections = input_data.get("sections", [])
        stamp = input_data.get("stamp", None)          # текст штампа
        stamp_color = input_data.get("stamp_color", "red")

        out_path = os.path.join(DIRS["docs"], f"{filename}.pdf")
        c = canvas.Canvas(out_path, pagesize=A4)
        width, height = A4

        # Header line
        c.setStrokeColor(colors.HexColor("#4A3FAA"))
        c.setLineWidth(2)
        c.line(2*cm, height - 2*cm, width - 2*cm, height - 2*cm)

        # Title
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.HexColor("#1a1a2e"))
        c.drawString(2*cm, height - 3*cm, title)

        # Date
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.grey)
        c.drawRightString(width - 2*cm, height - 3*cm,
                          datetime.now().strftime("Generated: %Y-%m-%d %H:%M"))

        # Sections
        y = height - 4.5*cm
        for section in sections:
            if y < 4*cm:
                c.showPage()
                y = height - 2*cm

            heading = section.get("heading", "")
            lines = section.get("lines", [])

            if heading:
                c.setFont("Helvetica-Bold", 13)
                c.setFillColor(colors.HexColor("#4A3FAA"))
                c.drawString(2*cm, y, heading)
                y -= 0.6*cm
                c.setStrokeColor(colors.HexColor("#ccccee"))
                c.setLineWidth(0.5)
                c.line(2*cm, y, width - 2*cm, y)
                y -= 0.5*cm

            c.setFont("Helvetica", 11)
            c.setFillColor(colors.black)
            for line in lines:
                if y < 4*cm:
                    c.showPage()
                    y = height - 2*cm
                c.drawString(2.5*cm, y, line)
                y -= 0.55*cm
            y -= 0.4*cm

        # Stamp (мокрая печать)
        if stamp:
            col = colors.red if stamp_color == "red" else colors.HexColor("#0055AA")
            c.saveState()
            c.setFont("Helvetica-Bold", 28)
            c.setFillColor(col)
            c.setStrokeColor(col)
            c.setFillAlpha(0.25)
            c.setStrokeAlpha(0.35)
            c.translate(width / 2, height / 2)
            c.rotate(35)
            c.roundRect(-120, -25, 240, 50, 8, stroke=1, fill=0)
            c.setFillAlpha(0.22)
            c.drawCentredString(0, -9, stamp.upper())
            c.restoreState()

        # Footer
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(width / 2, 1.5*cm, f"{title} | AI Self-Learning System")

        c.save()
        return {"status": "ok", "path": out_path, "filename": f"{filename}.pdf"}
