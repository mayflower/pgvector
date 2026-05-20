# Test turbohybrid BM25 WAL replay across restart.
use strict;
use warnings FATAL => 'all';
use Test::More;

BEGIN
{
	eval {
		require PostgreSQL::Test::Cluster;
		PostgreSQL::Test::Cluster->import();
		1;
	} or plan skip_all => 'PostgreSQL TAP test modules are not available';
}

my $node = PostgreSQL::Test::Cluster->new('turbohybrid_wal');
$node->init;
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION vector;');
$node->safe_psql('postgres', q(
	CREATE TABLE docs (
		id int PRIMARY KEY,
		embedding vector(3),
		body_tsv tsvector
	);
	INSERT INTO docs
	SELECT i,
		   format('[%s,%s,%s]', i % 7, (i + 1) % 7, (i + 2) % 7)::vector,
		   to_tsvector('english', 'common term' || (i % 5))
	FROM generate_series(1, 40) i;
	SET hybrid.debug_postings_chunk_size = 3;
	CREATE INDEX docs_turbohybrid_idx ON docs
	USING turbohybrid (
		embedding vector_cosine_hybrid_ops,
		body_tsv bm25_tsvector_ops
	)
	WITH (bm25_block_max = on, bm25_delta_compaction_threshold = 1);
	RESET hybrid.debug_postings_chunk_size;
));

$node->safe_psql('postgres', q(
	INSERT INTO docs
	SELECT i,
		   format('[%s,%s,%s]', i % 7, (i + 1) % 7, (i + 2) % 7)::vector,
		   to_tsvector('english', 'common delta term' || (i % 3))
	FROM generate_series(41, 55) i;
	VACUUM docs;
	CHECKPOINT;
));

my $before = $node->safe_psql('postgres', q(
	SELECT count(*)
	FROM (
		SELECT id
		FROM docs
		ORDER BY embedding <~> hybrid_query(
			text_query => to_tsquery('english', 'common'),
			dense_k => 0,
			bm25_k => 10
		)
		LIMIT 10
	) s;
));
is($before, '10', 'BM25-only query works before restart');

$node->restart;

my $after_bm25 = $node->safe_psql('postgres', q(
	SELECT count(*)
	FROM (
		SELECT id
		FROM docs
		ORDER BY embedding <~> hybrid_query(
			text_query => to_tsquery('english', 'common'),
			dense_k => 0,
			bm25_k => 10
		)
		LIMIT 10
	) s;
));
is($after_bm25, '10', 'BM25-only query works after restart');

my $after_hybrid = $node->safe_psql('postgres', q(
	SELECT count(*)
	FROM (
		SELECT id
		FROM docs
		ORDER BY embedding <~> hybrid_query(
			vector_query => '[1,2,3]'::vector,
			text_query => to_tsquery('english', 'common'),
			dense_k => 10,
			bm25_k => 10,
			final_k => 10
		)
		LIMIT 10
	) s;
));
is($after_hybrid, '10', 'hybrid query works after restart');

my $stats = $node->safe_psql('postgres', q(
	SELECT
		(tq_debug_bm25_stats('docs_turbohybrid_idx'::regclass)->>'compaction_count')::int > 0
		AND
		(tq_debug_bm25_stats('docs_turbohybrid_idx'::regclass)->>'postings_pages')::int > 0
		AND
		(tq_debug_bm25_stats('docs_turbohybrid_idx'::regclass)->>'blockmax_pages')::int > 0;
));
is($stats, 't', 'compacted BM25 page counters survive restart');

done_testing();
