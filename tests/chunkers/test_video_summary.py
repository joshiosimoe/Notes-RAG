from notes_rag.chunkers.video_summary import chunk_video_summary

PATH = "summaries/dQw4w9WgXcQ.json"


def test_overview_chunk_carries_verdict_tldr_and_takeaways(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Worth watching if you operate clusters at scale." in combined
    assert "Walks through the default scheduler" in combined
    assert "filtering and scoring" in combined


def test_every_chunk_has_video_citation_fields(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert chunk.corpus == "video"
        assert chunk.chunk_type == "summary"
        assert chunk.video_id == "dQw4w9WgXcQ"
        assert chunk.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert chunk.start_seconds is not None


def test_context_prefix_includes_title_and_channel(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    for chunk in chunks:
        assert "How Kubernetes Scheduling Actually Works" in chunk.text
        assert "Some Channel" in chunk.text


def test_all_chunks_have_a_content_hash(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)


def test_source_path_is_recorded(summary_sample):
    chunks = chunk_video_summary(summary_sample, source_path=PATH)
    assert all(chunk.source_path == PATH for chunk in chunks)


def test_missing_sections_still_produces_overview():
    minimal = {
        "video_id": "abc",
        "title": "T",
        "channel": "C",
        "url": "https://example.com",
        "summary": {"verdict": "v", "tldr": "t", "takeaways": [], "sections": []},
    }
    chunks = chunk_video_summary(minimal, source_path="summaries/abc.json")
    assert len(chunks) == 1
    assert "v" in chunks[0].text


def test_section_extraction_produces_distinct_chunks():
    """Verify section extraction and chunking logic works correctly.

    This test uses sections and overview long enough to survive normalization,
    ensuring section chunking logic is exercised and tested directly.
    """
    # Overview text that's long enough to survive normalization (150+ tokens)
    long_overview_verdict = (
        "This is an absolutely essential comprehensive guide for anyone and everyone "
        "working with modern machine learning infrastructure systems. It provides deep "
        "valuable insights into distributed training techniques, advanced model "
        "optimization strategies, and production deployment approaches that are critical "
        "for enterprise ML systems at scale. The comprehensive nature of this guide makes "
        "it indispensable for practitioners of all levels."
    )
    long_overview_tldr = (
        "Covers the complete and detailed lifecycle of ML systems including careful data "
        "preparation, model training with modern distributed frameworks, comprehensive "
        "hyperparameter optimization, thorough performance tuning, and successfully deploying "
        "models at scale to production environments. This includes monitoring strategies, "
        "debugging techniques, and handling real-world edge cases that arise in production."
    )

    summary = {
        "video_id": "test123",
        "title": "ML Infrastructure Guide",
        "channel": "Tech Academy",
        "url": "https://example.com/ml-guide",
        "summary": {
            "verdict": long_overview_verdict,
            "tldr": long_overview_tldr,
            "takeaways": [
                "Distributed training techniques accelerate model convergence significantly.",
                "Model quantization reduces inference latency without sacrificing accuracy.",
            ],
            "sections": [
                {
                    "start_seconds": 120,
                    "title": "Data Pipeline Architecture",
                    "summary": (
                        "This section focuses exclusively on designing and building robust data pipelines "
                        "for modern machine learning workflows. It explains in detail how to ingest data "
                        "from multiple heterogeneous sources, perform sophisticated feature engineering at scale, "
                        "handle missing values and outliers, and normalize features consistently across distributed systems. "
                        "The architecture uses Apache Spark for distributed processing, Pandas for data manipulation, "
                        "and DuckDB for analytical queries. Key implementation details include proper handling of "
                        "imbalanced datasets, implementing multiple data validation checkpoints, ensuring data quality "
                        "before model training, and managing data versioning. The pipeline must handle terabytes of "
                        "data efficiently while maintaining data consistency and lineage tracking throughout the system."
                    ),
                },
                {
                    "start_seconds": 1800,
                    "title": "Advanced Optimization Techniques",
                    "summary": (
                        "This section thoroughly demonstrates advanced optimization methods for training neural networks "
                        "at scale. Advanced methods include gradient accumulation, mixed precision training techniques, "
                        "and learning rate scheduling with warm restarts and cyclical approaches. The instructor shows how "
                        "to use TensorFlow's experimental distributed strategies and PyTorch's DistributedDataParallel for "
                        "efficient multi-GPU training. Performance profiling reveals that gradient checkpointing can reduce "
                        "memory usage by 50 percent while maintaining accuracy. The optimization section covers Adam optimizer "
                        "variants, sophisticated weight decay strategies, and batch normalization tuning for stability. These "
                        "advanced techniques are absolutely essential for production-grade ML systems requiring sub-second "
                        "inference latency and high throughput while maintaining model accuracy across varied input domains."
                    ),
                },
            ],
            "tags": ["ml", "infrastructure"],
        },
    }

    chunks = chunk_video_summary(summary, source_path="summaries/test123.json")

    # Verify we have multiple chunks (at least overview + one section)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    # Find the "Advanced Optimization Techniques" section chunk
    optimization_chunk = None
    for chunk in chunks:
        if chunk.heading == "Advanced Optimization Techniques":
            optimization_chunk = chunk
            break

    assert optimization_chunk is not None, (
        "Could not find 'Advanced Optimization Techniques' chunk. "
        f"Available headings: {[c.heading for c in chunks]}"
    )

    # Verify section-specific attributes
    assert optimization_chunk.start_seconds == 1800, (
        f"Expected start_seconds=1800, got {optimization_chunk.start_seconds}"
    )

    # Verify the section's unique content is present (substring only in this section)
    assert "sub-second inference latency" in optimization_chunk.text, (
        "Section-specific content 'sub-second inference latency' not found in chunk text"
    )
