from fpdf import FPDF
from datetime import datetime

class MedicalReport(FPDF):
    def header(self):
        # Header with title and logo
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "", align="R", ln=True)
        self.set_font("Arial", "B", 16)
        self.image('bitxlamarato.png', 145, 18, 50)  # Replace 'logo.png' with a valid path
        self.cell(0, 5, "Informe IPF", ln=True)

    def footer(self):
        # Page footer
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, "Pg 1", align="C")

    def add_patient_info(self, name, id):
        # Patient and test info
        self.set_font("Arial", "", 10)
        self.cell(0, 5, f"Nombre paciente: {name}", ln=True)
        self.cell(0, 5, f"Identificador: {id}", ln=True)
        # Automatically get the current date
        current_date = datetime.now().strftime("%d/%m/%Y")
        self.cell(0, 10, f"Fecha: {current_date}", ln=True)

    def add_section(self, title, data):
        # Section title
        self.set_font("Arial", "B", 11)
        self.set_text_color(220, 50, 50)
        self.cell(0, 10, title, ln=True)
        self.set_text_color(0, 0, 0)

        # Section content
        self.set_font("Arial", "", 9)
        for test, result, units, ref_range in data:
            self.cell(60, 7, test, border=0)
            self.cell(30, 7, f"{result}                                                 {units}", border=0)
            self.cell(50, 7, ref_range, border=0, ln=True)
