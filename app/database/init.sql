-- ============================================================
-- init.sql — Database Customer Loyalty Segmentation API
-- ============================================================

CREATE DATABASE cust_segmentation_db;

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
-- DUMMY DATA SEEDING
-- ============================================================

INSERT INTO users (name, email, password) VALUES
    ('Budi Santoso', 'budi@example.com', crypt('BudiPass123', gen_salt('bf'))),
    ('Ani Rahayu',   'ani@example.com',  crypt('AniPass123', gen_salt('bf'))),
    ('Baraja Putra', 'baraja@example.com', crypt('password123', gen_salt('bf')));
