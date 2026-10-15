-- Seed data for the SQL checks.
--
-- Fictional. Every defect below is planted on purpose and is asserted by
-- tests/test_sqlqa.py, so a check that stops working shows up as a failing test
-- rather than as a quiet all-clear.
--
-- Planted defects:
--   1. documents  d3 duplicates d2's sha256 against the same target
--   2. document   d9 references entity e9, which does not exist
--   3. extracted  x9 references document d999, which does not exist
--   4. entity     e1 is required to hold tax_returns and does not
--   5. entity     e1 extracts totals that are out by 100,000
--   6. asset      a2 carries another brand's name

INSERT INTO entities (id, name, brand_name) VALUES
    ('e1', 'Northgate Clinical Group', 'Northgate Clinical Group'),
    ('e2', 'Cascade Specialty Holdings', 'Cascade Specialty');

INSERT INTO data_room_requirements (entity_id, category) VALUES
    ('e1', 'articles_of_incorporation'),
    ('e1', 'financial_statements'),
    ('e1', 'tax_returns'),
    ('e2', 'articles_of_incorporation'),
    ('e2', 'financial_statements');

INSERT INTO documents (id, entity_id, category, sha256, received_at) VALUES
    ('d1', 'e1', 'articles_of_incorporation', 'aa11', '2026-01-04T09:00:00Z'),
    ('d2', 'e1', 'financial_statements',    'bb22', '2026-02-11T09:00:00Z'),
    ('d3', 'e1', 'financial_statements',    'bb22', '2026-02-12T09:00:00Z'),
    ('d4', 'e2', 'articles_of_incorporation', 'cc33', '2026-03-01T09:00:00Z'),
    ('d5', 'e2', 'financial_statements',    'dd44', '2026-03-02T09:00:00Z'),
    ('d9', 'e9', 'financial_statements',    'ee55', '2026-03-03T09:00:00Z');

INSERT INTO extracted_fields (document_id, field, value, page) VALUES
    ('d3', 'total_assets',       '13504500', 2),
    ('d3', 'total_liabilities',  '5320400',  2),
    ('d3', 'total_equity',       '8084100',  2),
    ('d3', 'revenue',            '24880000', 3),
    ('d5', 'total_assets',       '13000000', 2),
    ('d5', 'total_liabilities',  '6500000',  2),
    ('d5', 'total_equity',       '6500000',  2),
    ('d5', 'revenue',            '24900000', 3),
    ('d999', 'total_assets',     '999999',   2);

INSERT INTO brand_assets (entity_id, brand_name, asset_type, status) VALUES
    ('e1', 'Northgate Clinical Group', 'logo',        'approved'),
    ('e1', 'Cascade Specialty',        'ad_creative', 'draft'),
    ('e2', 'Cascade Specialty',        'logo',        'approved');
