ALTER TABLE face_clusters ADD COLUMN average_encoding BLOB;
ALTER TABLE face_clusters ADD COLUMN face_count INTEGER DEFAULT 0;
