-- check: orphaned_extracted_fields
-- description: Extracted values whose parent document does not exist. Nothing reconciles them and nothing will ever clean them up.
SELECT
    ef.id           AS extracted_field_id,
    ef.document_id  AS missing_document_id,
    ef.field,
    ef.value
FROM extracted_fields AS ef
LEFT JOIN documents AS d
    ON d.id = ef.document_id
WHERE d.id IS NULL
ORDER BY ef.id;
