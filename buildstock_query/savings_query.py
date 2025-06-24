from buildstock_query import main
from buildstock_query.aggregate_query import BuildStockAggregate


class BuildStockSavings(BuildStockAggregate):
    """Class for doing savings query (both timeseries and annual)."""

    _bsq: "main.BuildStockQuery"

    def __init__(self, buildstock_query: "main.BuildStockQuery") -> None:
        super().__init__(buildstock_query)
