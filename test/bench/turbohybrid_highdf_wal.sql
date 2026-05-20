CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS turbohybrid_highdf_docs;
CREATE TABLE turbohybrid_highdf_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO turbohybrid_highdf_docs
SELECT i,
	   format('[%s,%s,%s]', i % 17, (i + 1) % 17, (i + 2) % 17)::vector,
	   to_tsvector('english', 'common term' || (i % 100))
FROM generate_series(1, 70000) i;

SET hybrid.debug_postings_chunk_size = 512;
CREATE INDEX turbohybrid_highdf_docs_idx ON turbohybrid_highdf_docs
USING turbohybrid (
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (bm25_block_max = on, bm25_delta_compaction_threshold = 1);
RESET hybrid.debug_postings_chunk_size;

SELECT
	tq_debug_bm25_term_stats('turbohybrid_highdf_docs_idx'::regclass, 'common')->>'posting_count' = '70000' AS has_70000_postings,
	(tq_debug_bm25_term_stats('turbohybrid_highdf_docs_idx'::regclass, 'common')->>'postings_chunk_count')::int > 1 AS has_multi_page_postings,
	tq_debug_bm25_topk('turbohybrid_highdf_docs_idx'::regclass, to_tsquery('english', 'common'), 10, true)->>'result_count' = '10' AS wand_returns_10;

INSERT INTO turbohybrid_highdf_docs
SELECT i,
	   format('[%s,%s,%s]', i % 17, (i + 1) % 17, (i + 2) % 17)::vector,
	   to_tsvector('english', 'common delta')
FROM generate_series(70001, 71000) i;

VACUUM turbohybrid_highdf_docs;

SELECT
	tq_debug_bm25_stats('turbohybrid_highdf_docs_idx'::regclass)->>'delta_doc_count' = '0' AS compacted_delta,
	(tq_debug_bm25_stats('turbohybrid_highdf_docs_idx'::regclass)->>'compaction_count')::int > 0 AS compacted_once,
	(tq_debug_bm25_stats('turbohybrid_highdf_docs_idx'::regclass)->>'postings_pages')::int > 0 AS has_postings_pages,
	(tq_debug_bm25_stats('turbohybrid_highdf_docs_idx'::regclass)->>'blockmax_pages')::int > 0 AS has_blockmax_pages,
	(tq_debug_bm25_term_stats('turbohybrid_highdf_docs_idx'::regclass, 'common')->>'postings_chunk_count')::int > 1 AS common_still_chunked;
