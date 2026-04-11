CREATE EXTENSION IF NOT EXISTS vector;

-- Per-document style profiles with embedding vectors
CREATE TABLE style_profiles (
    id                  BIGSERIAL PRIMARY KEY,
    document_id         TEXT NOT NULL,
    document_title      TEXT NOT NULL,
    author              TEXT,
    lang                TEXT DEFAULT 'en',
    word_count          INTEGER,

    -- Layer 1: deterministic features
    classical_features  JSONB NOT NULL,
    classical_vector    vector(128),

    -- Layer 2: neural style embedding (optional)
    style_embedding     vector(768),

    -- Layer 3: hybrid (optional)
    hybrid_vector       vector(128),

    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE(document_id)
);

-- HNSW indexes (m=16, ef_construction=64 per ExecuFunction pattern)
CREATE INDEX idx_style_classical ON style_profiles
    USING hnsw (classical_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_style_neural ON style_profiles
    USING hnsw (style_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_style_hybrid ON style_profiles
    USING hnsw (hybrid_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_style_author ON style_profiles (author);

-- Author-level aggregated profiles
CREATE TABLE author_profiles (
    id                      BIGSERIAL PRIMARY KEY,
    author                  TEXT UNIQUE NOT NULL,
    doc_count               INTEGER DEFAULT 0,
    mean_classical_vector   vector(128),
    mean_classical_features JSONB,
    mean_style_embedding    vector(768),
    mean_hybrid_vector      vector(128),
    updated_at              TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_author_classical ON author_profiles
    USING hnsw (mean_classical_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
