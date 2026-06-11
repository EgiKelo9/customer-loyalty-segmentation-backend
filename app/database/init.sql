-- ============================================================
-- init.sql — Database Customer Loyalty Segmentation API
-- ============================================================

CREATE DATABASE cust_segmentation_db;

DROP TABLE IF EXISTS segmentation_results CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- 1. Users
-- ============================================================
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150) UNIQUE NOT NULL,
    password    VARCHAR(255) NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at  TIMESTAMP NULL
);

-- ============================================================
-- 2. Segmentation Results
-- ============================================================
CREATE TABLE segmentation_results (
    id                SERIAL PRIMARY KEY,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    customer_id       VARCHAR(100) NULL,
    cluster           INTEGER NOT NULL,
    pattern           VARCHAR(50) NOT NULL,
    segment           VARCHAR(100) NOT NULL,
    recommendation    TEXT NOT NULL,
    fuzzy_membership  JSONB NOT NULL,
    lrfm              JSONB NULL,
    source            VARCHAR(50) NOT NULL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_segmentation_results_user_id ON segmentation_results(user_id);
CREATE INDEX idx_segmentation_results_created_at ON segmentation_results(created_at);

-- ============================================================
-- DUMMY DATA SEEDING
-- ============================================================

INSERT INTO users (name, email, password) VALUES
    ('Budi Santoso', 'budi@example.com', crypt('BudiPass123', gen_salt('bf'))),
    ('Ani Rahayu',   'ani@example.com',  crypt('AniPass123', gen_salt('bf'))),
    ('Baraja Putra', 'baraja@example.com', crypt('password123', gen_salt('bf')));
