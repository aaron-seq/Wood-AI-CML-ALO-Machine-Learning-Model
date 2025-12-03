-- Initialize CML Optimization Database
-- This script is automatically run when PostgreSQL container starts

-- Ensure extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE cml_optimization TO cml_user;

-- Create indexes for better performance (will be created by SQLAlchemy, but included here for reference)
-- Tables will be created by SQLAlchemy Base.metadata.create_all()
