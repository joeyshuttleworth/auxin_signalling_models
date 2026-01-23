
def pytest_addoption(parser):
    parser.addoption(
        "--ignore_golden",
        action="store_true",
        default=False,
        help="Skip golden snapshot tests"
    )
