from pathlib import Path

import pytest
import sqlalchemy as sa
import toml

from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.main import BuildStockQuery
from buildstock_query.query_core import QueryCore
from buildstock_query.report_query import BuildStockReport


def _make_bsq(baseline: sa.Table, upgrades: sa.Table | None = None) -> BuildStockQuery:
    schema_path = Path(__file__).resolve().parents[1] / "buildstock_query" / "db_schema" / "resstock_default.toml"
    bsq = BuildStockQuery.__new__(BuildStockQuery)
    bsq.db_schema = DBSchema.model_validate(toml.load(schema_path))
    bsq.bs_table = baseline
    bsq.up_table = upgrades
    bsq.building_id_column_name = bsq.db_schema.column_names.building_id
    return bsq


def test_baseline_success_report_selects_from_baseline_table() -> None:
    metadata = sa.MetaData()
    baseline = sa.Table("baseline", metadata, sa.Column("building_id", sa.Integer))
    bsq = _make_bsq(baseline)

    query = BuildStockReport(bsq)._get_bs_success_report(get_query_only=True)

    assert "FROM baseline" in query


def test_missing_upgrade_status_preserves_null_for_unmatched_rows() -> None:
    metadata = sa.MetaData()
    baseline = sa.Table("baseline", metadata, sa.Column("building_id", sa.Integer))
    upgrades = sa.Table(
        "upgrades",
        metadata,
        sa.Column("building_id", sa.Integer),
        sa.Column("upgrade", sa.String),
    )
    bsq = _make_bsq(baseline, upgrades)
    status = bsq._up_completed_status_col
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(baseline.insert(), [{"building_id": 1}, {"building_id": 2}])
        connection.execute(upgrades.insert(), [{"building_id": 1, "upgrade": "1"}])
        rows = connection.execute(
            sa.select(status)
            .select_from(baseline.outerjoin(upgrades, baseline.c.building_id == upgrades.c.building_id))
            .order_by(baseline.c.building_id)
        ).scalars().all()

    assert rows == [bsq.db_schema.completion_values.success, None]


def test_missing_table_does_not_swallow_reflection_query_failures() -> None:
    query_core = QueryCore.__new__(QueryCore)
    query_core._tables = {}
    query_core._meta = sa.MetaData()
    query_core._engine = sa.create_engine("sqlite://")
    query_core.db_name = "test"

    with pytest.raises(sa.exc.OperationalError):
        query_core._get_table("missing", missing_ok=True)
