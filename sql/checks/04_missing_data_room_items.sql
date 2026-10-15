-- check: missing_data_room_items
-- description: A required checklist item with nothing filed against it. This is the room that reads as complete on the checklist because nobody counts the rows.
SELECT
    r.entity_id,
    e.name   AS entity_name,
    r.category AS missing_category
FROM data_room_requirements AS r
LEFT JOIN entities AS e
    ON e.id = r.entity_id
LEFT JOIN documents AS d
    ON d.entity_id = r.entity_id
   AND d.category = r.category
WHERE d.id IS NULL
ORDER BY r.entity_id, r.category;
