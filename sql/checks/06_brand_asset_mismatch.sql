-- check: brand_asset_mismatch
-- description: A generated asset carrying a brand name that belongs to another brand. With one parent brand splitting into eight or ten, this is the defect that reaches a patient.
SELECT
    a.id         AS asset_id,
    a.entity_id,
    a.brand_name AS asset_brand,
    e.brand_name AS owning_brand,
    a.asset_type,
    a.status
FROM brand_assets AS a
JOIN entities AS e
    ON e.id = a.entity_id
WHERE a.brand_name <> e.brand_name
ORDER BY a.id;
