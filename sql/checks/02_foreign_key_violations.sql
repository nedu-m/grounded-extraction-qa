-- check: foreign_key_violations
-- description: Referential integrity, checked by the database rather than by hand. Legacy imports were loaded with enforcement off, so this is a real risk and not a formality.
PRAGMA foreign_key_check;
