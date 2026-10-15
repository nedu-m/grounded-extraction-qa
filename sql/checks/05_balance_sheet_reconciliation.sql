-- check: balance_sheet_reconciliation
-- description: Reconciling two sources that disagree. The extracted totals for the most recent statements against each target must foot, or the summary built on top of them is wrong.
WITH latest_statements AS (
    SELECT d.id, d.entity_id
    FROM documents AS d
    WHERE d.category = 'financial_statements'
      AND d.entity_id IS NOT NULL
      AND d.received_at = (
          SELECT MAX(peer.received_at)
          FROM documents AS peer
          WHERE peer.entity_id = d.entity_id
            AND peer.category = d.category
      )
),
extracted AS (
    SELECT
        s.entity_id,
        MAX(CASE WHEN ef.field = 'total_assets' THEN CAST(ef.value AS NUMERIC) END) AS total_assets,
        MAX(CASE WHEN ef.field = 'total_liabilities' THEN CAST(ef.value AS NUMERIC) END) AS total_liabilities,
        MAX(CASE WHEN ef.field = 'total_equity' THEN CAST(ef.value AS NUMERIC) END) AS total_equity
    FROM latest_statements AS s
    JOIN extracted_fields AS ef
        ON ef.document_id = s.id
    GROUP BY s.entity_id
)
SELECT
    entity_id,
    total_assets,
    total_liabilities,
    total_equity,
    total_liabilities + total_equity      AS liabilities_and_equity,
    total_assets - (total_liabilities + total_equity) AS out_by
FROM extracted
WHERE total_assets IS NOT NULL
  AND total_liabilities IS NOT NULL
  AND total_equity IS NOT NULL
  AND total_assets <> total_liabilities + total_equity
ORDER BY entity_id;
