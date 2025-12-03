#!/usr/bin/env python
"""
Seed database with sample CML data

Usage:
    python scripts/seed_database.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import datetime
import logging

from backend.app.core.database import SessionLocal, engine, Base
from backend.app.models.db_models import CML, RiskLevel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def seed_database(excel_file: str = 'data/raw/CML_Optimization_Sample_Data.xlsx'):
    """Seed database with sample data from Excel file"""
    
    logger.info(f"Seeding database from {excel_file}...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    # Read Excel file
    if not os.path.exists(excel_file):
        logger.error(f"File not found: {excel_file}")
        return
    
    df = pd.read_excel(excel_file, sheet_name='CML_Master_Data')
    logger.info(f"Loaded {len(df)} CML records from Excel")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Clear existing data (optional - comment out if you want to keep existing data)
        db.query(CML).delete()
        db.commit()
        logger.info("Cleared existing CML data")
        
        # Insert CMLs
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            try:
                # Map Risk_Level string to enum
                risk_level_str = str(row['Risk_Level']).upper().replace(' ', '_')
                risk_level = RiskLevel[risk_level_str] if risk_level_str in RiskLevel.__members__ else None
                
                cml = CML(
                    cml_id=row['CML_ID'],
                    line_id=row.get('Line_ID'),
                    equipment_id=row.get('Equipment_ID'),
                    facility=row.get('Facility'),
                    system=row.get('System'),
                    commodity=row.get('Commodity'),
                    material_type=row.get('Material_Type'),
                    feature_type=row.get('Feature_Type'),
                    cml_shape=row.get('CML_Shape'),
                    design_thickness_mm=row.get('Design_Thickness_mm'),
                    min_allowable_thickness_mm=row.get('Min_Allowable_Thickness_mm'),
                    corrosion_allowance_mm=row.get('Corrosion_Allowance_mm'),
                    current_thickness_mm=row.get('Current_Thickness_mm'),
                    average_corrosion_rate=row.get('Average_Corrosion_Rate_mm_per_year'),
                    years_in_service=int(row.get('Years_In_Service', 0)),
                    number_of_inspections=int(row.get('Number_of_Inspections', 0)),
                    last_inspection_date=pd.to_datetime(row.get('Last_Inspection_Date')).date() if pd.notna(row.get('Last_Inspection_Date')) else None,
                    first_inspection_date=pd.to_datetime(row.get('First_Inspection_Date')).date() if pd.notna(row.get('First_Inspection_Date')) else None,
                    remaining_life_years=row.get('Remaining_Life_Years'),
                    risk_level=risk_level,
                    isometric_id=row.get('Isometric_ID'),
                    inspection_technique=row.get('Inspection_Technique'),
                    data_quality_score=row.get('Data_Quality_Score'),
                    elimination_candidate=bool(row.get('Elimination_Candidate', 0)),
                    requires_engineering_review=bool(row.get('Requires_Engineering_Review', 0)),
                    inspection_history_dates=row.get('Inspection_History_Dates'),
                    inspection_history_measurements=row.get('Inspection_History_Measurements'),
                    notes=row.get('Notes', '')
                )
                
                db.add(cml)
                successful += 1
                
                if (idx + 1) % 50 == 0:
                    db.commit()
                    logger.info(f"Committed {idx + 1} CMLs...")
                
            except Exception as e:
                logger.error(f"Error inserting CML at row {idx}: {e}")
                failed += 1
        
        db.commit()
        logger.info(f"✅ Database seeded successfully!")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
