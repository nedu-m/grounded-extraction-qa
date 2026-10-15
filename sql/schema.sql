-- Diligence data room schema.
--
-- Mirrors the shape of the tooling the role supports: targets, the documents a
-- seller uploads against a checklist, the fields extracted from those documents,
-- and the brand assets generated across the specialty brands.
--
-- Foreign keys are declared, not relied upon. Legacy imports were loaded with
-- enforcement off, which is why `foreign_key_violations` is a check rather than
-- an invariant.

CREATE TABLE entities (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    brand_name TEXT NOT NULL
);

CREATE TABLE documents (
    id          TEXT PRIMARY KEY,
    entity_id   TEXT REFERENCES entities (id),
    category    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    received_at TEXT NOT NULL
);

CREATE TABLE data_room_requirements (
    entity_id TEXT NOT NULL REFERENCES entities (id),
    category  TEXT NOT NULL,
    PRIMARY KEY (entity_id, category)
);

CREATE TABLE extracted_fields (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT REFERENCES documents (id),
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    page        INTEGER
);

CREATE TABLE brand_assets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  TEXT REFERENCES entities (id),
    brand_name TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    status     TEXT NOT NULL
);

CREATE INDEX idx_documents_entity ON documents (entity_id, category);
CREATE INDEX idx_extracted_document ON extracted_fields (document_id);
