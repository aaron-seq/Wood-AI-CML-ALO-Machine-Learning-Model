from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io
import logging
from typing import List, Any

logger = logging.getLogger(__name__)

class ReportService:
    """PDF report generation for CML analysis"""
    
    def generate_pdf_report(
        self, 
        cmls: List[Any],
        include_forecasts: bool = True,
        include_shap: bool = True
    ) -> io.BytesIO:
        """Generate comprehensive PDF report"""
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Build content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("CML Optimization Analysis Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        
        total_cmls = len(cmls)
        eliminations = sum(1 for c in cmls if c.elimination_candidate)
        critical = sum(1 for c in cmls if c.risk_level and 'CRITICAL' in str(c.risk_level))
        
        summary_data = [
            ['Total CMLs', str(total_cmls)],
            ['Elimination Candidates', str(eliminations)],
            ['Critical Risk CMLs', str(critical)],
            ['Potential Cost Savings', f"${eliminations * 1500:,.0f}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Detailed CML List
        story.append(Paragraph("Elimination Candidates", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        candidates = [c for c in cmls if c.elimination_candidate]
        
        if candidates:
            table_data = [['CML ID', 'Facility', 'Risk', 'Remaining Life (yrs)', 'Confidence']]
            
            for cml in candidates[:20]:  # Limit to 20
                table_data.append([
                    cml.cml_id,
                    cml.facility or 'N/A',
                    str(cml.risk_level.value) if cml.risk_level else 'N/A',
                    f"{cml.remaining_life_years:.1f}" if cml.remaining_life_years else 'N/A',
                    f"{cml.ml_confidence*100:.0f}%" if cml.ml_confidence else 'N/A'
                ])
            
            cml_table = Table(table_data, colWidths=[1.2*inch, 1.2*inch, 1*inch, 1.3*inch, 1*inch])
            cml_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(cml_table)
        else:
            story.append(Paragraph("No elimination candidates identified.", styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles['Heading2']))
        
        recommendations = [
            "1. Prioritize elimination of low-risk CMLs with high remaining life",
            "2. Conduct engineering review for CMLs flagged for review",
            "3. Monitor critical risk CMLs with enhanced inspection frequency",
            "4. Implement SME override process for final approval",
            "5. Schedule re-analysis annually or after major process changes"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        logger.info(f"PDF report generated for {total_cmls} CMLs")
        return buffer
