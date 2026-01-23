import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ignore_golden",
        action="store_true",
        default=False,
        help="Skip golden snapshot tests"
    )

# def pytest_configure(config):
#     # optional: store globally
#     config.ignore_golden = config.getoption("ignore_golden")
